"""Shared neural-network primitives for every NagaHana component (docs/build-spec.md §5).

One implementation of each primitive, so that every component normalises, attends, encodes time and
loops in the same way:

- `norms`       RMSNorm (pre-norm blocks, AS-32)
- `mlp`         SwiGLU feed-forward (AS-32)
- `numeric`     signed log1p and periodic numeric embeddings (AS-31)
- `positional`  continuous-time rotary encoding, log-Δt bucket bias, clock features, RWSE (D-49, D-50)
- `attention`   multi-head attention with typed boolean masks, additive bias, null keys, K/V return,
                dense and gathered-key forms
- `blocks`      pre-norm self-attention block and decoder-style (self + cross) block
- `loop`        the two-stream weight-tied loop (memory stream once, thinking stream R passes; AS-06)
- `lora`        low-rank adapters for site calibration (AS-26)
"""
