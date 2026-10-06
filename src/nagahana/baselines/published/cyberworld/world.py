"""CyberWorld's world model: modality encoders, fusion, RSSM, decoders and heads (AS-557).

Modalities (each a set of elements of width d after its own encoder and self-attention)
    vector   a feature vector x [D]: element i = symlog(x_i) w_i + b_i (feature tokens)
    graph    node features [V, f] over an adjacency: symlog, a linear map to d, then a graph-attention
             encoder (the topology encoder)
    text     embeddings [L, d_text] of a frozen language model (CyberWorld's text and multimodal variants
             use Qwen2.5-3B, which is not an approved dependency: the pathway takes precomputed embeddings
             and the encoder that produces them is optional, see TextEncoder)
Every modality has its own self-attention layers; cross-attention fusion (networks.py) joins them into the
embedding e_t that the RSSM posterior reads.

Variants: "vector" uses the vector modalities, "graph" adds the graph modality, "text" and "multimodal"
add the text modality (they need a TextEncoder).

Decoders reconstruct the vector modalities and the node features of a graph modality with a fixed number
of slots, in symlog space with a squared error summed over components (DreamerV3's symlog MSE). Heads:
reward (two-hot over symexp bins), continuation (Bernoulli) and labelled heads (categorical, or Bernoulli
for one class), each an MLP on [h ; z].

Loss (per step, averaged over the batch):
    L = beta_pred (L_decoders + L_reward + L_continue + lambda_labels L_labels) + beta_dyn L_dyn + beta_rep L_rep
with beta_pred = 1, beta_dyn = 0.5, beta_rep = 0.1 and 1 free nat (rssm.py).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nagahana.baselines.published.base import MissingDependency
from nagahana.baselines.published.cyberworld.networks import (
    MLP,
    CrossAttentionFusion,
    FeatureTokens,
    GATEncoder,
    SelfAttentionBlock,
    TwoHot,
    symlog,
)
from nagahana.baselines.published.cyberworld.rssm import RSSM, RSSMState

Variant = Literal["vector", "graph", "text", "multimodal"]


@dataclass(frozen=True)
class Modality:
    """One input modality.

    kind "vector": dim = D. kind "graph": dim = node feature width, slots = fixed node count (0 for graphs
    of varying size, which are not decoded). kind "text": dim = embedding width of the frozen encoder,
    slots = maximum number of embeddings.
    """

    name: str
    kind: Literal["vector", "graph", "text"]
    dim: int
    slots: int = 0
    decode: bool = True


class TextEncoder(Protocol):
    """A frozen text encoder: texts -> embeddings [n, L, d_text] and masks [n, L] (for example Qwen2.5-3B)."""

    def encode(self, texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]: ...


def require_text_encoder(encoder: TextEncoder | None, variant: str) -> TextEncoder:
    """The text and multimodal variants need a text encoder; none is bundled (frozen Qwen2.5-3B weights are
    not an approved dependency)."""
    if encoder is None:
        raise MissingDependency(f"the {variant!r} variant needs a TextEncoder producing frozen language-model embeddings; "
                                "CyberWorld uses Qwen2.5-3B, whose weights are not an approved dependency, so none is bundled")
    return encoder


@dataclass
class WorldModelSettings:
    """Sizes and loss weights (DreamerV3 M-size RSSM: 1,024 deterministic units, 32 x 32 latent; AS-557)."""

    deter: int = 1024
    groups: int = 32
    classes: int = 32
    units: int = 640
    mlp_layers: int = 3
    d_model: int = 128
    attn_heads: int = 4
    attn_layers: int = 1
    fusion_queries: int = 8
    embed: int = 1024
    gat_layers: int = 2
    gat_heads: int = 4
    free_nats: float = 1.0
    beta_pred: float = 1.0
    beta_dyn: float = 0.5
    beta_rep: float = 0.1
    label_weight: float = 1.0
    unimix: float = 0.01

    def validate(self) -> None:
        if min(self.deter, self.groups, self.classes, self.units, self.mlp_layers, self.d_model, self.attn_heads,
               self.fusion_queries, self.embed, self.gat_layers, self.gat_heads) < 1 or self.attn_layers < 0:
            raise ValueError("world-model sizes must be positive (attn_layers >= 0)")
        if self.d_model % self.attn_heads:
            raise ValueError("d_model must be divisible by attn_heads")
        if self.free_nats < 0 or min(self.beta_pred, self.beta_dyn, self.beta_rep, self.label_weight) < 0:
            raise ValueError("loss weights and free nats must be non-negative")
        if not 0.0 <= self.unimix < 1.0:
            raise ValueError("unimix must lie in [0, 1)")


class ModalityEncoder(nn.Module):
    """Elements [N, n, d] and mask [N, n] of one modality (module docstring)."""

    def __init__(self, modality: Modality, s: WorldModelSettings) -> None:
        super().__init__()
        self.modality = modality
        if modality.kind == "vector":
            self.tokens: nn.Module = FeatureTokens(modality.dim, s.d_model)
        elif modality.kind == "graph":
            self.tokens = nn.Linear(modality.dim, s.d_model)
            self.gat = GATEncoder(s.d_model, s.d_model, layers=s.gat_layers, heads=s.gat_heads)
        else:
            self.tokens = nn.Linear(modality.dim, s.d_model)
        self.blocks = nn.ModuleList(SelfAttentionBlock(s.d_model, s.attn_heads) for _ in range(s.attn_layers))

    def forward(self, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.modality
        if m.kind == "vector":
            x = self.tokens(symlog(obs[m.name]))                                   # [N, D, d]
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        elif m.kind == "graph":
            mask = obs[f"{m.name}.mask"]
            x = self.tokens(symlog(obs[f"{m.name}.nodes"]))                        # [N, V, d]
            x = self.gat(x, obs[f"{m.name}.adjacency"], mask)
        else:
            mask = obs[f"{m.name}.mask"]
            x = self.tokens(obs[f"{m.name}.tokens"]) * mask[..., None].to(torch.float32)
        for block in self.blocks:
            x = block(x, mask)
        return x, mask


class WorldModel(nn.Module):
    """Encoders, fusion, RSSM, decoders and heads (module docstring)."""

    def __init__(self, modalities: Sequence[Modality], settings: WorldModelSettings, *, action_dim: int,
                 reward_head: bool, continue_head: bool, label_heads: Mapping[str, int] | None = None) -> None:
        super().__init__()
        settings.validate()
        if not modalities:
            raise ValueError("the world model needs at least one modality")
        names = [m.name for m in modalities]
        if len(set(names)) != len(names):
            raise ValueError("modality names must be unique")
        self.modalities = tuple(modalities)
        self.s = settings
        self.encoders = nn.ModuleDict({m.name: ModalityEncoder(m, settings) for m in modalities})
        self.fusion = CrossAttentionFusion(settings.d_model, settings.attn_heads, settings.fusion_queries, len(modalities), settings.embed)
        self.rssm = RSSM(deter=settings.deter, groups=settings.groups, classes=settings.classes, embed=settings.embed,
                         action_dim=action_dim, units=settings.units, unimix=settings.unimix)
        feat = self.rssm.feature_dim
        self.decoders = nn.ModuleDict()
        for m in modalities:
            if not m.decode or m.kind == "text":
                continue
            if m.kind == "vector":
                self.decoders[m.name] = MLP(feat, settings.units, settings.mlp_layers, m.dim)
            elif m.slots > 0:
                self.decoders[m.name] = MLP(feat, settings.units, settings.mlp_layers, m.slots * m.dim)
        self.twohot = TwoHot()
        self.reward = MLP(feat, settings.units, settings.mlp_layers, self.twohot.n_bins) if reward_head else None
        self.cont = MLP(feat, settings.units, settings.mlp_layers, 1) if continue_head else None
        self.label_heads = nn.ModuleDict({k: MLP(feat, settings.units, settings.mlp_layers, n) for k, n in (label_heads or {}).items()})
        if self.reward is not None:
            # DreamerV3 initialises the reward and critic output layers to zero (prediction 0 at the start).
            last = self.reward.net[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def embed(self, obs: Mapping[str, torch.Tensor], lead: tuple[int, ...]) -> torch.Tensor:
        """Fused embeddings [*lead, E] of observations whose tensors have leading shape `lead`."""
        n = int(np.prod(lead))
        flat = {k: v.reshape(n, *v.shape[len(lead):]) for k, v in obs.items()}
        parts = [self.encoders[m.name](flat) for m in self.modalities]
        e = self.fusion([p[0] for p in parts], [p[1] for p in parts])
        return e.reshape(*lead, -1)

    def decode_loss(self, feat: torch.Tensor, obs: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Summed symlog squared error of every decoder, per step [B, T]."""
        total = torch.zeros(feat.shape[:-1], device=feat.device)
        parts: dict[str, torch.Tensor] = {}
        for m in self.modalities:
            if m.name not in self.decoders:
                continue
            pred = self.decoders[m.name](feat)
            if m.kind == "vector":
                err = ((pred - symlog(obs[m.name])) ** 2).sum(-1)
            else:
                target = symlog(obs[f"{m.name}.nodes"]).flatten(-2)
                mask = obs[f"{m.name}.mask"][..., None].expand(*obs[f"{m.name}.mask"].shape, m.dim).flatten(-2).to(pred.dtype)
                err = (((pred - target) ** 2) * mask).sum(-1)
            parts[f"decoder.{m.name}"] = err.mean()
            total = total + err
        return total, parts

    def head_losses(self, feat: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Reward, continuation and label losses per step [B, T]."""
        total = torch.zeros(feat.shape[:-1], device=feat.device)
        parts: dict[str, torch.Tensor] = {}
        if self.reward is not None and "reward" in batch:
            loss = self.twohot.loss(self.reward(feat), batch["reward"])
            parts["reward"] = loss.mean()
            total = total + loss
        if self.cont is not None and "continue" in batch:
            loss = F.binary_cross_entropy_with_logits(self.cont(feat)[..., 0], batch["continue"], reduction="none")
            parts["continue"] = loss.mean()
            total = total + loss
        for name, head in self.label_heads.items():
            key = f"label.{name}"
            if key not in batch:
                continue
            target = batch[key]
            known = target >= 0
            logits = head(feat)
            if logits.shape[-1] == 1:
                loss = F.binary_cross_entropy_with_logits(logits[..., 0], target.clamp_min(0).to(logits.dtype), reduction="none")
            else:
                loss = F.cross_entropy(logits.movedim(-1, 1), target.clamp_min(0), reduction="none")
            loss = loss * known.to(loss.dtype)
            parts[key] = loss.sum() / known.sum().clamp_min(1)
            total = total + self.s.label_weight * loss
        return total, parts

    def loss(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float], RSSMState]:
        """World-model loss of a batch of sequences [B, T, ...]; returns (loss, metrics, posteriors)."""
        first = batch["first"]
        lead = tuple(first.shape)
        obs = {k: v for k, v in batch.items() if k.split(".")[0] in self.encoders}
        e = self.embed(obs, lead)
        actions = batch.get("action")
        post, prior = self.rssm.observe(e, actions, first.bool())
        feat = post.features()
        dec, dec_parts = self.decode_loss(feat, obs)
        heads, head_parts = self.head_losses(feat, batch)
        dyn, rep, raw_kl = self.rssm.kl_losses(post, prior, self.s.free_nats)
        loss = (self.s.beta_pred * (dec + heads) + self.s.beta_dyn * dyn + self.s.beta_rep * rep).mean()
        metrics = {"loss": float(loss.detach()), "kl": float(raw_kl.mean()), "dyn": float(dyn.mean().detach()),
                   **{k: float(v.detach()) for k, v in {**dec_parts, **head_parts}.items()}}
        return loss, metrics, post

    def label_probs(self, feat: torch.Tensor, name: str) -> torch.Tensor:
        """Probabilities of a label head: [..., 1] (Bernoulli) or [..., n] (categorical), in float64."""
        logits = self.label_heads[name](feat).double()
        return torch.sigmoid(logits) if logits.shape[-1] == 1 else torch.softmax(logits, dim=-1)

    def reward_mean(self, feat: torch.Tensor) -> torch.Tensor:
        assert self.reward is not None
        return self.twohot.mean(self.reward(feat))

    def continue_prob(self, feat: torch.Tensor) -> torch.Tensor:
        assert self.cont is not None
        return torch.sigmoid(self.cont(feat)[..., 0])


def variant_modalities(variant: Variant, *, vector: Sequence[Modality], graph: Modality | None,
                       text: Modality | None) -> list[Modality]:
    """The modalities of a variant (module docstring)."""
    mods = list(vector)
    if variant in ("graph", "multimodal"):
        if graph is None:
            raise ValueError(f"the {variant!r} variant needs a graph modality")
        mods.append(graph)
    if variant in ("text", "multimodal"):
        if text is None:
            raise ValueError(f"the {variant!r} variant needs a text modality")
        mods.append(text)
    return mods


@dataclass
class ReplaySequences:
    """Fixed-capacity replay of transitions for sequence sampling (DreamerV3 uniform replay).

    Arrays are stored per key with a leading time axis; `first` marks episode starts. A sampled sequence
    of length L starts anywhere in the stored range (sequences may cross episode boundaries; `first`
    resets the state there, as in DreamerV3).
    """

    capacity: int
    data: dict[str, np.ndarray] = field(default_factory=dict)
    size: int = 0
    cursor: int = 0

    def add(self, step: Mapping[str, np.ndarray | float | bool]) -> None:
        for k, v in step.items():
            arr = np.asarray(v)
            if k not in self.data:
                self.data[k] = np.zeros((self.capacity, *arr.shape), dtype=arr.dtype)
            self.data[k][self.cursor] = arr
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int, length: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if self.size < length:
            raise ValueError(f"replay holds {self.size} steps, fewer than the sequence length {length}")
        # Valid starts keep a sequence inside the written, time-contiguous region of the ring buffer.
        oldest = self.cursor if self.size == self.capacity else 0
        starts = (oldest + rng.integers(0, self.size - length + 1, size=batch)) % self.capacity
        idx = (starts[:, None] + np.arange(length)[None, :]) % self.capacity      # [B, L]
        out = {k: v[idx] for k, v in self.data.items()}
        out["first"] = out["first"].copy()
        out["first"][:, 0] = True                                                  # a sequence starts from the initial state
        return out
