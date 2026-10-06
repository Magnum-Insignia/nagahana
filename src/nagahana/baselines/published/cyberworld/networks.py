"""Building blocks of CyberWorld (AS-557; DreamerV3 numerics from Hafner, Pasukonis, Ba and Lillicrap,
"Mastering Diverse Domains through World Models", arXiv:2301.04104).

symlog / symexp
    symlog(x) = sign(x) log(1 + |x|),  symexp(y) = sign(y) (exp(|y|) - 1)
    Vector observations are predicted in symlog space, so targets of very different scales share one loss.

Two-hot distributional regression (rewards and values)
    bins B_i = symexp(linspace(-20, 20, 255)); a target y is encoded as weights on its two neighbouring
    bins, w_k = (B_{k+1} - y) / (B_{k+1} - B_k) on k and 1 - w_k on k + 1; the loss is the cross-entropy
    of softmax(logits) against that encoding; the prediction is sum_i softmax(logits)_i B_i.

DreamerV3 MLP layer:  x -> Linear -> LayerNorm -> SiLU.

DreamerV3 GRU (normalised, update bias -1):
    [r, c, u] = LayerNorm(W [x ; h]);  r = sigma(r);  c = tanh(r * c);  u = sigma(u - 1);  h' = u c + (1 - u) h

Graph attention (Velickovic et al., ICLR 2018, arXiv:1710.10903), head k, self-loops included:
    e_ij = LeakyReLU_0.2(a_k^T [W_k h_i ; W_k h_j]),  alpha_ij = softmax_{j in N(i) + i}(e_ij),
    h'_i = ||_k ELU(sum_j alpha_ij W_k h_j)  (hidden layers),  mean_k (...)  (output layer)
over a dense adjacency with a node mask (graphs of one batch are padded to the same node count).

Modality self-attention: a pre-norm Transformer encoder layer per modality (multi-head attention and a
GELU feed-forward, each with a residual connection), with key masks for padded elements.

Cross-attention fusion: Q learned queries attend to the concatenated elements of every modality (each
element carries its modality's learned embedding); the Q outputs are flattened and projected to the
embedding the RSSM posterior reads.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """sign(x) log(1 + |x|)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(y: torch.Tensor) -> torch.Tensor:
    """sign(y) (exp(|y|) - 1), the inverse of symlog."""
    return torch.sign(y) * torch.expm1(torch.abs(y))


class TwoHot(nn.Module):
    """Two-hot encoding over symexp-spaced bins (module docstring)."""

    def __init__(self, n_bins: int = 255, low: float = -20.0, high: float = 20.0) -> None:
        super().__init__()
        self.register_buffer("bins", symexp(torch.linspace(low, high, n_bins, dtype=torch.float64)).float(), persistent=False)

    @property
    def n_bins(self) -> int:
        return int(self.bins.shape[0])

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """Target weights [..., n_bins] of values y [...] (clipped to the bin range)."""
        bins = self.bins
        y = y.clamp(float(bins[0]), float(bins[-1]))
        k = torch.searchsorted(bins, y.contiguous(), right=True).clamp(1, bins.shape[0] - 1) - 1    # lower bin
        lo, hi = bins[k], bins[k + 1]
        w_hi = ((y - lo) / (hi - lo)).clamp(0.0, 1.0)
        out = torch.zeros(*y.shape, bins.shape[0], device=y.device, dtype=y.dtype)
        out.scatter_(-1, k[..., None], (1.0 - w_hi)[..., None])
        out.scatter_add_(-1, (k + 1)[..., None], w_hi[..., None])
        return out

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Cross-entropy of softmax(logits) against twohot(y), per element."""
        return -(self.encode(y.detach()) * F.log_softmax(logits, dim=-1)).sum(dim=-1)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value sum_i softmax(logits)_i B_i."""
        return (F.softmax(logits, dim=-1) * self.bins).sum(dim=-1)


class MLP(nn.Module):
    """`layers` DreamerV3 layers (Linear, LayerNorm, SiLU), then an optional linear output."""

    def __init__(self, d_in: int, units: int, layers: int, d_out: int | None = None) -> None:
        super().__init__()
        mods: list[nn.Module] = []
        prev = d_in
        for _ in range(layers):
            mods += [nn.Linear(prev, units, bias=False), nn.LayerNorm(units, eps=1e-3), nn.SiLU()]
            prev = units
        if d_out is not None:
            mods.append(nn.Linear(prev, d_out))
        self.net = nn.Sequential(*mods)
        self.d_out = d_out if d_out is not None else prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class NormGRUCell(nn.Module):
    """DreamerV3's layer-normalised GRU cell with update bias -1 (module docstring)."""

    def __init__(self, d_in: int, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in + hidden, 3 * hidden, bias=False)
        self.norm = nn.LayerNorm(3 * hidden, eps=1e-3)
        self.hidden = hidden

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        parts = self.norm(self.linear(torch.cat([x, h], dim=-1)))
        r, c, u = parts.split(self.hidden, dim=-1)
        r = torch.sigmoid(r)
        c = torch.tanh(r * c)
        u = torch.sigmoid(u - 1.0)
        out: torch.Tensor = u * c + (1.0 - u) * h
        return out


class GATLayer(nn.Module):
    """One multi-head graph-attention layer on padded dense graphs (module docstring)."""

    def __init__(self, d_in: int, d_head: int, heads: int, *, concat: bool, dropout: float = 0.0) -> None:
        super().__init__()
        self.heads, self.d_head, self.concat = heads, d_head, concat
        self.w = nn.Linear(d_in, heads * d_head, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, d_head))
        self.a_dst = nn.Parameter(torch.empty(heads, d_head))
        self.bias = nn.Parameter(torch.zeros(heads * d_head if concat else d_head))
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)

    @property
    def d_out(self) -> int:
        return self.heads * self.d_head if self.concat else self.d_head

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """x [G, V, d_in], adj [G, V, V] bool (i attends to j where adj[i, j]), node_mask [G, V] -> [G, V, d_out]."""
        g, v, _ = x.shape
        wx = self.w(x).view(g, v, self.heads, self.d_head)                              # [G, V, H, d_h]
        s_dst = (wx * self.a_dst).sum(-1)                                               # [G, V, H] (receiving node i)
        s_src = (wx * self.a_src).sum(-1)                                               # [G, V, H] (sending node j)
        e = F.leaky_relu(s_dst[:, :, None, :] + s_src[:, None, :, :], 0.2)              # [G, V(i), V(j), H]
        eye = torch.eye(v, dtype=torch.bool, device=x.device)[None]
        allowed = (adj | eye) & node_mask[:, None, :] & node_mask[:, :, None]
        allowed = allowed | eye                                                          # padded rows attend to themselves
        e = e.masked_fill(~allowed[..., None], float("-inf"))
        alpha = self.drop(torch.softmax(e, dim=2))                                       # over neighbours j
        out = torch.einsum("gijh,gjhd->gihd", alpha, wx)                                 # [G, V, H, d_h]
        out = out.reshape(g, v, self.heads * self.d_head) if self.concat else out.mean(dim=2)
        return (out + self.bias) * node_mask[..., None].to(out.dtype)


class GATEncoder(nn.Module):
    """Stack of GAT layers: ELU between layers, heads concatenated except at the output (averaged)."""

    def __init__(self, d_in: int, d_out: int, *, layers: int = 2, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("the GAT encoder needs at least one layer")
        mods: list[GATLayer] = []
        prev = d_in
        for i in range(layers):
            last = i == layers - 1
            layer = GATLayer(prev, d_out if last else max(1, d_out // heads), heads, concat=not last, dropout=dropout)
            mods.append(layer)
            prev = layer.d_out
        self.layers = nn.ModuleList(mods)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, adj, node_mask)
            if i < len(self.layers) - 1:
                x = F.elu(x)
        return x


class SelfAttentionBlock(nn.Module):
    """Pre-norm Transformer encoder layer with key masking."""

    def __init__(self, d: int, heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x [B, n, d], mask [B, n] (True = real element)."""
        y = self.norm1(x)
        # Rows without any real element attend to themselves only, so no attention row is empty.
        pad = ~mask
        if pad.all(dim=1).any():
            pad = pad.clone()
            pad[pad.all(dim=1), 0] = False
        a, _ = self.attn(y, y, y, key_padding_mask=pad, need_weights=False)
        x = x + a
        out: torch.Tensor = x + self.ff(self.norm2(x))
        return out * mask[..., None].to(out.dtype)


class FeatureTokens(nn.Module):
    """A vector x [B, D] as D elements: element i is x_i w_i + b_i (one learned embedding per feature)."""

    def __init__(self, n_features: int, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d) / math.sqrt(d))
        self.bias = nn.Parameter(torch.zeros(n_features, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = x[..., None] * self.weight + self.bias
        return out


class CrossAttentionFusion(nn.Module):
    """Q learned queries attend to every modality's elements; output [B, Q * d] -> Linear -> [B, d_out]."""

    def __init__(self, d: int, heads: int, queries: int, n_modalities: int, d_out: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(queries, d) / math.sqrt(d))
        self.modality = nn.Parameter(torch.zeros(n_modalities, d))
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(queries * d, d_out), nn.LayerNorm(d_out, eps=1e-3), nn.SiLU())

    def forward(self, elements: list[torch.Tensor], masks: list[torch.Tensor]) -> torch.Tensor:
        b = elements[0].shape[0]
        kv = torch.cat([e + self.modality[i] for i, e in enumerate(elements)], dim=1)   # [B, n_total, d]
        mask = torch.cat(masks, dim=1)
        pad = ~mask
        if pad.all(dim=1).any():
            pad = pad.clone()
            pad[pad.all(dim=1), 0] = False
        q = self.norm_q(self.queries).expand(b, -1, -1)
        fused, _ = self.attn(q, self.norm_kv(kv), self.norm_kv(kv), key_padding_mask=pad, need_weights=False)
        out: torch.Tensor = self.out(fused.reshape(b, -1))
        return out
