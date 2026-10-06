"""CyberWorld as a CyberWheel defense agent: Dreamer actor-critic trained on imagined trajectories (AS-557).

Learning loop (DreamerV3, Hafner et al., arXiv:2301.04104), with CyberWorld's budget of 52,500
environment steps:
    1. act in the environments with the current policy (uniform random actions during the prefill),
       filtering the RSSM state online; store (observation, previous action, reward, continuation, first)
    2. after every environment step, run as many updates as keep (replayed steps) / (environment steps)
       at the train ratio; each update samples B sequences of length L
    3. world-model update (world.py)
    4. imagination: from every posterior state of the batch, roll the prior forward H steps with actions
       from the actor; rewards and continuations from the world-model heads
    5. lambda-returns:  R_t = r_{t+1} + gamma c_{t+1} ((1 - lambda) v(s_{t+1}) + lambda R_{t+1}),  R_H = v(s_H)
    6. critic: two-hot cross-entropy towards sg(R_t), plus the same loss towards the slow critic's
       prediction (an exponential moving average of the critic, fraction 0.02 per update)
    7. actor: -sg((R_t - v(s_t)) / max(1, S)) log pi(a_t | s_t) - eta H[pi(. | s_t)], with S an exponential
       moving average (decay 0.99) of the 95th minus the 5th percentile of the returns and eta = 3e-4
    Steps 6 and 7 weight each imagined step by the cumulative product of gamma c (the probability of
    still being in the episode). The actor's distribution mixes 1 % uniform probability.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nagahana.baselines.published.base import BaselineConfig
from nagahana.baselines.published.cyberwheel.agents import AgentSpec, DefenseAgent
from nagahana.baselines.published.cyberwheel.env import DefenseObservation, EnvFactory, VectorEnv
from nagahana.baselines.published.cyberwheel.ppo import CYBERWORLD_REFERENCE
from nagahana.baselines.published.cyberworld.networks import MLP, TwoHot
from nagahana.baselines.published.cyberworld.rssm import RSSMState
from nagahana.baselines.published.cyberworld.world import (
    Modality,
    ReplaySequences,
    TextEncoder,
    Variant,
    WorldModel,
    WorldModelSettings,
    require_text_encoder,
    variant_modalities,
)
from nagahana.baselines.published.neural import resolve_device


@dataclass
class CyberWorldAgentConfig(BaselineConfig):
    """Dreamer training of CyberWorld in CyberWheel (module docstring; AS-557).

    Attributes
    ----------
    variant:
        "vector", "graph", "text" or "multimodal" (world.py).
    world:
        World-model sizes and loss weights.
    env_steps, num_envs, prefill:
        Environment-step budget (52,500 in CyberWorld), parallel environments, random-action prefill.
    train_ratio, batch_size, batch_length:
        Replayed steps per environment step, and the sampled batch shape.
    imag_horizon, gamma, lam, actor_entropy, actor_unimix:
        Imagination and actor-critic settings.
    model_lr, actor_lr, critic_lr, adam_eps, model_clip, ac_clip:
        Adam settings and global gradient-norm clipping.
    slow_critic_fraction, return_decay, return_low, return_high:
        Slow critic and return normalisation.
    replay_capacity, text_slots:
        Replay size and the maximum text embeddings per step (text variants).
    """

    variant: Variant = "graph"
    world: WorldModelSettings = field(default_factory=WorldModelSettings)
    env_steps: int = 52_500
    num_envs: int = 1
    prefill: int = 2_500
    train_ratio: float = 512.0
    batch_size: int = 16
    batch_length: int = 64
    imag_horizon: int = 15
    gamma: float = 1.0 - 1.0 / 333.0
    lam: float = 0.95
    actor_entropy: float = 3e-4
    actor_unimix: float = 0.01
    model_lr: float = 1e-4
    actor_lr: float = 3e-5
    critic_lr: float = 3e-5
    adam_eps: float = 1e-8
    model_clip: float = 1000.0
    ac_clip: float = 100.0
    slow_critic_fraction: float = 0.02
    return_decay: float = 0.99
    return_low: float = 5.0
    return_high: float = 95.0
    replay_capacity: int = 1_000_000
    text_slots: int = 16

    def validate(self) -> None:
        super().validate()
        self.world.validate()
        if min(self.env_steps, self.num_envs, self.batch_size, self.batch_length, self.imag_horizon, self.replay_capacity) < 1:
            raise ValueError("env_steps, num_envs, batch_size, batch_length, imag_horizon and replay_capacity must be >= 1")
        if self.prefill < self.batch_length or self.prefill > self.env_steps:
            raise ValueError("prefill must lie in [batch_length, env_steps]")
        if self.train_ratio <= 0 or not (0 < self.gamma <= 1 and 0 <= self.lam <= 1):
            raise ValueError("train_ratio > 0, gamma in (0, 1] and lam in [0, 1]")
        if not 0 <= self.return_low < self.return_high <= 100:
            raise ValueError("return percentiles must satisfy 0 <= low < high <= 100")


AGENT_SPEC = AgentSpec(name="cyberworld-agent", title="CyberWorld Dreamer defense agent", reference=CYBERWORLD_REFERENCE,
                       environment="cyberwheel", budget_steps=52_500, third_party="cyberworld", assumptions=("AS-557", "AS-558"))


class Actor(nn.Module):
    """Categorical policy over the decoy actions with 1 % uniform mixing."""

    def __init__(self, feat: int, units: int, layers: int, n_actions: int, unimix: float) -> None:
        super().__init__()
        self.net = MLP(feat, units, layers, n_actions)
        self.unimix = unimix
        self.n_actions = n_actions

    def dist(self, feat: torch.Tensor) -> torch.distributions.Categorical:
        probs = (1.0 - self.unimix) * torch.softmax(self.net(feat), dim=-1) + self.unimix / self.n_actions
        return torch.distributions.Categorical(probs=probs)


class CyberWorldAgent(DefenseAgent):
    """CyberWorld trained in CyberWheel (module docstring)."""

    spec: ClassVar[AgentSpec] = AGENT_SPEC
    config_type: ClassVar[type[BaselineConfig]] = CyberWorldAgentConfig
    config: CyberWorldAgentConfig

    def __init__(self, config: BaselineConfig | None = None, *, text_encoder: TextEncoder | None = None,
                 text_source: Callable[[DefenseObservation], str] | None = None) -> None:
        super().__init__(config)
        self.text_encoder = text_encoder
        self.text_source = text_source
        if self.config.variant in ("text", "multimodal"):
            require_text_encoder(text_encoder, self.config.variant)
            if text_source is None:
                raise ValueError(f"the {self.config.variant!r} variant needs a text_source (observation -> text)")
        self.world: WorldModel | None = None
        self.actor: Actor | None = None
        self.critic: MLP | None = None
        self.shape: dict[str, int] = {}
        self._act_state: RSSMState | None = None
        self._prev_action: torch.Tensor | None = None

    def _modalities(self) -> list[Modality]:
        s = self.shape
        vector = [Modality("hosts", "vector", s["n_slots"] * s["n_fields"]), Modality("telemetry", "vector", s["telemetry_dim"])]
        vector = [m for m in vector if m.dim > 0]
        graph = Modality("topology", "graph", s["n_fields"], slots=s["n_slots"], decode=False)
        text = Modality("text", "text", s.get("text_dim", 0) or 1, slots=self.config.text_slots, decode=False)
        return variant_modalities(self.config.variant, vector=vector, graph=graph, text=text)

    def _build(self) -> None:
        cfg = self.config
        self.world = WorldModel(self._modalities(), cfg.world, action_dim=self.shape["n_actions"], reward_head=True, continue_head=True)
        feat = self.world.rssm.feature_dim
        self.actor = Actor(feat, cfg.world.units, cfg.world.mlp_layers, self.shape["n_actions"], cfg.actor_unimix)
        self.critic = MLP(feat, cfg.world.units, cfg.world.mlp_layers, TwoHot().n_bins)
        last = self.critic.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def _obs_arrays(self, o: DefenseObservation) -> dict[str, np.ndarray]:
        """Observation -> arrays of the world model's modalities (one step)."""
        out: dict[str, np.ndarray] = {"hosts": o.hosts.ravel().astype(np.float32)}
        if o.telemetry.size:
            out["telemetry"] = o.telemetry.astype(np.float32)
        if self.config.variant in ("graph", "multimodal"):
            out["topology.nodes"] = o.hosts.astype(np.float32)
            out["topology.adjacency"] = o.adjacency.astype(bool)
            out["topology.mask"] = np.ones(o.n_slots, dtype=bool)
        if self.config.variant in ("text", "multimodal"):
            assert self.text_encoder is not None and self.text_source is not None
            emb, mask = self.text_encoder.encode([self.text_source(o)])
            tok = np.zeros((self.config.text_slots, emb.shape[-1]), dtype=np.float32)
            msk = np.zeros(self.config.text_slots, dtype=bool)
            k = min(self.config.text_slots, emb.shape[1])
            tok[:k], msk[:k] = emb[0, :k], mask[0, :k]
            out["text.tokens"], out["text.mask"] = tok, msk
        return out

    def _tensors(self, arrays: Mapping[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}

    def _imagine(self, start: RSSMState, horizon: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Imagined features [H + 1, N, F], actions [H, N] and their log-probabilities [H, N]."""
        assert self.world is not None and self.actor is not None
        state = start
        feats, actions, logps, ents = [state.features()], [], [], []
        for _ in range(horizon):
            dist = self.actor.dist(state.features().detach())
            a = dist.sample()
            onehot = F.one_hot(a, self.shape["n_actions"]).float()
            state = self.world.rssm.img_step(state, onehot)
            feats.append(state.features())
            actions.append(a)
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
        return torch.stack(feats), torch.stack(logps), torch.stack(ents)

    def _train(self, factory: EnvFactory, rng: np.random.Generator) -> None:
        cfg = self.config
        device = resolve_device(cfg.device)
        envs = VectorEnv(factory, cfg.num_envs, cfg.seed)
        obs = envs.reset()
        o0 = obs[0]
        self.shape = {"n_slots": o0.n_slots, "n_fields": int(o0.hosts.shape[1]), "telemetry_dim": int(o0.telemetry.size),
                      "n_actions": envs.action_space.n}
        if cfg.variant in ("text", "multimodal"):
            self.shape["text_dim"] = self._text_dim(o0)
        self._build()
        assert self.world is not None and self.actor is not None and self.critic is not None
        self.world.to(device)
        self.actor.to(device)
        self.critic.to(device)
        slow_critic = copy.deepcopy(self.critic)
        for p in slow_critic.parameters():
            p.requires_grad_(False)
        twohot = TwoHot().to(device)
        opt_model = torch.optim.Adam(self.world.parameters(), lr=cfg.model_lr, eps=cfg.adam_eps)
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, eps=cfg.adam_eps)
        opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, eps=cfg.adam_eps)
        replay = ReplaySequences(min(cfg.replay_capacity, cfg.env_steps + cfg.num_envs))
        n_act = self.shape["n_actions"]
        state = self.world.rssm.initial(cfg.num_envs, device)
        prev_action = torch.zeros(cfg.num_envs, n_act, device=device)
        first = np.ones(cfg.num_envs, dtype=bool)
        rewards_in = np.zeros(cfg.num_envs)
        conts_in = np.ones(cfg.num_envs)
        return_scale = 1.0
        updates_done = 0
        history: list[dict[str, float]] = []
        steps = 0
        while steps < cfg.env_steps:
            arrays = [self._obs_arrays(o) for o in obs]
            batch_obs = {k: torch.as_tensor(np.stack([a[k] for a in arrays]), device=device) for k in arrays[0]}
            with torch.no_grad():
                reset = torch.as_tensor(first, device=device, dtype=torch.float32)[:, None]
                init = self.world.rssm.initial(cfg.num_envs, device)
                state = RSSMState(reset * init.h + (1 - reset) * state.h, state.logits, reset * init.z + (1 - reset) * state.z)
                emb = self.world.embed(batch_obs, (cfg.num_envs,))
                state, _ = self.world.rssm.obs_step(state, prev_action * (1 - reset), emb)
                if steps < cfg.prefill:
                    act = rng.integers(0, n_act, size=cfg.num_envs)
                else:
                    act = self.actor.dist(state.features()).sample().cpu().numpy()
            for i in range(cfg.num_envs):
                replay.add({**arrays[i], "action": prev_action[i].cpu().numpy().astype(np.float32),
                            "reward": np.float32(rewards_in[i]), "continue": np.float32(conts_in[i]), "first": bool(first[i])})
            obs, rew, term, trunc, infos = envs.step(act.tolist())
            prev_action = F.one_hot(torch.as_tensor(act, device=device), n_act).float()
            steps += cfg.num_envs
            # The transition into an episode end is stored with its final observation (Dreamer convention).
            for i in range(cfg.num_envs):
                if term[i] or trunc[i]:
                    final = self._obs_arrays(infos[i]["final_observation"])
                    replay.add({**final, "action": prev_action[i].cpu().numpy().astype(np.float32),
                                "reward": np.float32(rew[i]), "continue": np.float32(0.0 if term[i] else 1.0), "first": False})
            first = term | trunc
            rewards_in = np.where(first, 0.0, rew)
            conts_in = np.ones(cfg.num_envs)
            if first.any():
                prev_action = prev_action * torch.as_tensor(~first, device=device, dtype=torch.float32)[:, None]
            if steps < cfg.prefill:
                continue
            target_updates = int(steps * cfg.train_ratio / (cfg.batch_size * cfg.batch_length))
            while updates_done < target_updates and replay.size >= cfg.batch_length:
                metrics, return_scale = self._update(replay, rng, device, opt_model, opt_actor, opt_critic, slow_critic, twohot,
                                                     return_scale)
                updates_done += 1
                if updates_done % 50 == 1:
                    history.append({"env_steps": float(steps), "update": float(updates_done), **metrics})
        self.train_report.update({"env_steps": steps, "updates": updates_done, "history": history,
                                  "episodes": envs.stats.summary(), "return_scale": return_scale})

    def _text_dim(self, o: DefenseObservation) -> int:
        assert self.text_encoder is not None and self.text_source is not None
        emb, _ = self.text_encoder.encode([self.text_source(o)])
        return int(emb.shape[-1])

    def _update(self, replay: ReplaySequences, rng: np.random.Generator, device: torch.device, opt_model: torch.optim.Optimizer,
                opt_actor: torch.optim.Optimizer, opt_critic: torch.optim.Optimizer, slow_critic: MLP, twohot: TwoHot,
                return_scale: float) -> tuple[dict[str, float], float]:
        assert self.world is not None and self.actor is not None and self.critic is not None
        cfg = self.config
        sample = replay.sample(cfg.batch_size, cfg.batch_length, rng)
        batch = {k: torch.as_tensor(v, device=device) for k, v in sample.items()}
        batch["first"] = batch["first"].bool()
        loss, metrics, post = self.world.loss(batch)
        opt_model.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.world.parameters(), cfg.model_clip)
        opt_model.step()
        # Imagination from every posterior state of the batch (gradients do not reach the world model).
        start = RSSMState(post.h.detach().flatten(0, 1), post.logits.detach().flatten(0, 1), post.z.detach().flatten(0, 1))
        for p in self.world.parameters():
            p.requires_grad_(False)
        feats, logps, ents = self._imagine(start, cfg.imag_horizon)                  # [H+1, N, F], [H, N], [H, N]
        for p in self.world.parameters():
            p.requires_grad_(True)
        with torch.no_grad():
            rew = self.world.reward_mean(feats[1:])                                  # [H, N]
            cont = self.world.continue_prob(feats[1:])
            first_cont = batch["continue"].flatten(0, 1)[None]
            disc = cfg.gamma * torch.cat([first_cont, cont], dim=0)                  # [H+1, N]
            weight = torch.cumprod(torch.cat([torch.ones_like(first_cont), disc[:-1]], dim=0), dim=0)
            value = twohot.mean(self.critic(feats))                                  # [H+1, N]
            ret = torch.zeros_like(value)
            ret[-1] = value[-1]
            for t in reversed(range(cfg.imag_horizon)):
                ret[t] = rew[t] + disc[t + 1] * ((1 - cfg.lam) * value[t + 1] + cfg.lam * ret[t + 1])
            lo = torch.quantile(ret[:-1].flatten(), cfg.return_low / 100.0)
            hi = torch.quantile(ret[:-1].flatten(), cfg.return_high / 100.0)
            return_scale = cfg.return_decay * return_scale + (1 - cfg.return_decay) * float(hi - lo)
            adv = (ret[:-1] - value[:-1]) / max(1.0, return_scale)
            slow = twohot.mean(slow_critic(feats[:-1]))
        actor_loss = (weight[:-1] * (-adv * logps - cfg.actor_entropy * ents)).mean()
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.ac_clip)
        opt_actor.step()
        critic_logits = self.critic(feats[:-1].detach())
        critic_loss = (weight[:-1] * (twohot.loss(critic_logits, ret[:-1]) + twohot.loss(critic_logits, slow))).mean()
        opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.ac_clip)
        opt_critic.step()
        with torch.no_grad():
            for ps, pc in zip(slow_critic.parameters(), self.critic.parameters(), strict=True):
                ps.mul_(1 - cfg.slow_critic_fraction).add_(cfg.slow_critic_fraction * pc)
        metrics.update({"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                        "imagined_return": float(ret[0].mean()), "entropy": float(ents.detach().mean())})
        return metrics, return_scale

    def reset_state(self) -> None:
        """Forget the filtered state (called at the start of every evaluation episode)."""
        self._act_state = None
        self._prev_action = None

    @torch.no_grad()
    def _act(self, observation: DefenseObservation, greedy: bool) -> int:
        assert self.world is not None and self.actor is not None
        device = resolve_device(self.config.device)
        arrays = self._obs_arrays(observation)
        obs = {k: torch.as_tensor(v[None], device=device) for k, v in arrays.items()}
        if self._act_state is None or self._prev_action is None:
            self._act_state = self.world.rssm.initial(1, device)
            self._prev_action = torch.zeros(1, self.shape["n_actions"], device=device)
        emb = self.world.embed(obs, (1,))
        self._act_state, _ = self.world.rssm.obs_step(self._act_state, self._prev_action, emb)
        dist = self.actor.dist(self._act_state.features())
        a = int(dist.probs.argmax(dim=-1)[0]) if greedy else int(dist.sample()[0])
        self._prev_action = F.one_hot(torch.as_tensor([a], device=device), self.shape["n_actions"]).float()
        return a

    def _export_state(self, directory: Path) -> dict[str, Any]:
        assert self.world is not None and self.actor is not None and self.critic is not None
        torch.save({"world": self.world.state_dict(), "actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
                   directory / "cyberworld.pt")
        return {"weights": "cyberworld.pt", "shape": self.shape}

    def _import_state(self, directory: Path, state: Mapping[str, Any]) -> None:
        self.shape = {k: int(v) for k, v in state["shape"].items()}
        self._build()
        assert self.world is not None and self.actor is not None and self.critic is not None
        tensors = torch.load(directory / str(state["weights"]), map_location="cpu", weights_only=True)
        device = resolve_device(self.config.device)
        for module, key in ((self.world, "world"), (self.actor, "actor"), (self.critic, "critic")):
            module.load_state_dict(tensors[key])
            module.to(device)
            module.eval()

