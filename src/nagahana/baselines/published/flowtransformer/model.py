"""FlowTransformer network: input encoding, transformer blocks, classification head, MLP.

A window holds T consecutive flow records; the classified flow is the last one (position T - 1). For flow
element t with scaled numeric fields u_t in R^{D_n} and categorical codes k_{t,c}:

Input encodings (paper Sec. V; "Record Emb. Dense" is the encoding of the best reported rows)
    none                    e_t = [u_t ; onehot(k_{t,1}) ; ... ; onehot(k_{t,C})]
    record_dense            e_t = W [u_t ; onehot(k_t)] + b                       (W: d_e x width)
    record_projection       e_t = W [u_t ; onehot(k_t)]                           (no bias)
    categorical_dense       e_t = [u_t ; W_1 onehot(k_{t,1}) + b_1 ; ... ]        (W_c: d_c x L_c)
    categorical_lookup      e_t = [u_t ; E_1[k_{t,1}] ; ... ]                     (embedding tables)
    categorical_projection  e_t = [u_t ; W_1 onehot(k_{t,1}) ; ... ]              (no bias)

Transformer block (post-norm, as the "basic transformer" of the framework, AS-543)
    a = MHA(e, e, e)                              H heads of width d_h, output projection to d_model
    e = LayerNorm(e + Dropout(a))                 epsilon 1e-6
    e = LayerNorm(e + Dropout(W_2 ReLU(W_1 e)))   W_1: ff_dim x d_model
The decoder variant masks attention to positions <= t (GPT style). No positional encoding is added:
the paper's head comparison is consistent with a position-agnostic encoder, since only the heads that
keep the position of the classified flow (last token, flatten, featurewise) reach high F1 while the
permutation-invariant heads (global average pooling, CLS token) stay near F1 0.33 (AS-543).

Classification heads (applied to the block output h [B, T, d])
    last_token              z = h[:, T - 1]
    flatten                 z = [h_0 ; ... ; h_{T-1}]
    global_average_pooling  z = mean_t h_t over the window's real (unpadded) positions
    cls_token               a learned vector c is prepended; z = h_cls
    featurewise             z = [w^T h_0 + b ; ... ] with a shared map R^d -> R^k per position (k =
                            featurewise_dim), flattened to T * k values (AS-545)
then an MLP (layers of `mlp_sizes`, ReLU, dropout) and one sigmoid unit.

Windows shorter than T at the start of a stream are left-padded; padded positions are masked out of
attention (every position may attend to itself, so no attention row is empty), zeroed before the
flatten and featurewise heads, and excluded from the average.

All Dense kernels use Keras's Glorot-uniform initialisation and zero biases, embedding tables Keras's
uniform(-0.05, 0.05), so the network starts from the framework's initial distribution.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from nagahana.baselines.published.neural import keras_dense_init

ENCODINGS = ("none", "record_dense", "record_projection", "categorical_dense", "categorical_lookup", "categorical_projection")
HEADS = ("last_token", "flatten", "global_average_pooling", "cls_token", "featurewise")


class InputEncoding(nn.Module):
    """Maps (numeric [B, T, D_n], codes [B, T, C]) to flow-element vectors [B, T, d_model]."""

    def __init__(self, kind: str, n_numeric: int, level_counts: list[int], embed_dim: int, categorical_dim: int) -> None:
        super().__init__()
        if kind not in ENCODINGS:
            raise ValueError(f"unknown encoding {kind!r}; known: {ENCODINGS}")
        self.kind = kind
        self.level_counts = list(level_counts)
        width = n_numeric + sum(self.level_counts)
        self.record: nn.Linear | None = None
        self.per_field = nn.ModuleList()
        if kind == "none":
            self.out_dim = width
        elif kind in ("record_dense", "record_projection"):
            self.record = nn.Linear(width, embed_dim, bias=kind == "record_dense")
            self.out_dim = embed_dim
        elif kind == "categorical_lookup":
            for n_levels in self.level_counts:
                emb = nn.Embedding(n_levels, categorical_dim)
                nn.init.uniform_(emb.weight, -0.05, 0.05)
                self.per_field.append(emb)
            self.out_dim = n_numeric + categorical_dim * len(self.level_counts)
        else:
            bias = kind == "categorical_dense"
            for n_levels in self.level_counts:
                self.per_field.append(nn.Linear(n_levels, categorical_dim, bias=bias))
            self.out_dim = n_numeric + categorical_dim * len(self.level_counts)
        keras_dense_init(self)

    def _onehot(self, codes: torch.Tensor) -> list[torch.Tensor]:
        # One indicator block per categorical field: [B, T, L_c] each.
        return [F.one_hot(codes[..., c], n).to(torch.float32) for c, n in enumerate(self.level_counts)]

    def forward(self, numeric: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        if self.kind == "categorical_lookup":
            parts = [numeric, *(emb(codes[..., c]) for c, emb in enumerate(self.per_field))]
            return torch.cat(parts, dim=-1)
        onehots = self._onehot(codes)
        if self.kind in ("none", "record_dense", "record_projection"):
            x = torch.cat([numeric, *onehots], dim=-1)                       # [B, T, width]
            return x if self.record is None else self.record(x)
        return torch.cat([numeric, *(lin(oh) for lin, oh in zip(self.per_field, onehots, strict=True))], dim=-1)


class TransformerBlock(nn.Module):
    """Post-norm multi-head attention block with a ReLU feed-forward layer (see the module docstring)."""

    def __init__(self, d_model: int, ff_dim: int, n_heads: int, head_dim: int, dropout: float) -> None:
        super().__init__()
        self.n_heads, self.head_dim = n_heads, head_dim
        inner = n_heads * head_dim
        self.q = nn.Linear(d_model, inner)
        self.k = nn.Linear(d_model, inner)
        self.v = nn.Linear(d_model, inner)
        self.o = nn.Linear(inner, d_model)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_attn = nn.LayerNorm(d_model, eps=1e-6)
        self.ff_in = nn.Linear(d_model, ff_dim)
        self.ff_out = nn.Linear(ff_dim, d_model)
        self.drop_ff = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(d_model, eps=1e-6)
        keras_dense_init(self)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """x [B, T, d]; allowed [B, 1, T, T] bool (True where query t may attend to key s)."""
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)    # [B, H, T, d_h]
        k = self.k(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)            # [B, H, T, d_h]
        a = self.o(a.transpose(1, 2).reshape(b, t, self.n_heads * self.head_dim))
        x = self.norm_attn(x + self.drop_attn(a))
        y = self.ff_out(F.relu(self.ff_in(x)))
        out: torch.Tensor = self.norm_ff(x + self.drop_ff(y))
        return out


class FlowTransformerNet(nn.Module):
    """Encoding -> transformer blocks -> head -> MLP -> logit of the last flow of each window."""

    def __init__(self, *, n_numeric: int, level_counts: list[int], window: int, encoding: str, embed_dim: int,
                 categorical_dim: int, n_layers: int, ff_dim: int, n_heads: int, head_dim: int, dropout: float,
                 causal: bool, head: str, featurewise_dim: int, mlp_sizes: tuple[int, ...], mlp_dropout: float) -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; known: {HEADS}")
        self.encoding = InputEncoding(encoding, n_numeric, level_counts, embed_dim, categorical_dim)
        d_model = self.encoding.out_dim
        h_dim = head_dim if head_dim > 0 else max(1, d_model // n_heads)
        self.blocks = nn.ModuleList(TransformerBlock(d_model, ff_dim, n_heads, h_dim, dropout) for _ in range(n_layers))
        self.window, self.causal, self.head = window, causal, head
        self.cls = nn.Parameter(torch.zeros(d_model)) if head == "cls_token" else None
        if self.cls is not None:
            nn.init.uniform_(self.cls, -0.05, 0.05)
        self.featurewise = nn.Linear(d_model, featurewise_dim) if head == "featurewise" else None
        z_dim = {"last_token": d_model, "flatten": window * d_model, "global_average_pooling": d_model,
                 "cls_token": d_model, "featurewise": window * featurewise_dim}[head]
        layers: list[nn.Module] = []
        prev = z_dim
        for width in mlp_sizes:
            layers += [nn.Linear(prev, width), nn.ReLU(), nn.Dropout(mlp_dropout)]
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        keras_dense_init(self.mlp)
        if self.featurewise is not None:
            keras_dense_init(self.featurewise)

    def _allowed(self, pad: torch.Tensor) -> torch.Tensor:
        # [B, 1, T, T]: keys that are real (not padding), causal when decoding, and always the diagonal.
        b, t = pad.shape
        allowed = (~pad)[:, None, None, :].expand(b, 1, t, t)
        if self.causal:
            allowed = allowed & torch.tril(torch.ones(t, t, dtype=torch.bool, device=pad.device))
        return allowed | torch.eye(t, dtype=torch.bool, device=pad.device)

    def forward(self, numeric: torch.Tensor, codes: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """numeric [B, T, D_n], codes [B, T, C], pad [B, T] (True = padding) -> logits [B]."""
        h = self.encoding(numeric, codes)                                          # [B, T, d]
        if self.cls is not None:
            h = torch.cat([self.cls.expand(h.shape[0], 1, -1), h], dim=1)          # CLS at position 0
            pad = torch.cat([torch.zeros_like(pad[:, :1]), pad], dim=1)
        allowed = self._allowed(pad)
        for block in self.blocks:
            h = block(h, allowed)
        real = (~pad).to(h.dtype)[..., None]                                       # [B, T, 1]
        if self.head == "last_token":
            z = h[:, -1]
        elif self.head == "cls_token":
            z = h[:, 0]
        elif self.head == "global_average_pooling":
            z = (h * real).sum(dim=1) / real.sum(dim=1).clamp_min(1.0)
        elif self.head == "flatten":
            z = (h * real).flatten(1)
        else:
            assert self.featurewise is not None
            z = (self.featurewise(h) * real).flatten(1)
        logit: torch.Tensor = self.mlp(z)[:, 0]
        return logit
