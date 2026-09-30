# Parameter budget (illustrative tiers; not decisions)

Generated with `python -m nagahana.lab.sizing` (src/nagahana/lab/sizing.py).
- **Excluded:** the Generator.
- **Not decided:** every width and depth in `conf/` is still `???`. These three tiers are sizing
  scenarios matched to the hardware range in ARCH #18 ("from server gpus to basic rpi").
- **How counted:** the counts are exact for the stated shapes, taken from real PyTorch modules:
  - the actual reference CVG-AE;
  - standard transformer blocks, checked against 12d² + 13d (encoder) and 16d² + 19d (decoder
    with cross-attention).

| Component | S (edge / small site) | M (enterprise workstation) | L (CII server) |
|---|---:|---:|---:|
| Input layer (field embeddings) | 0.53 M | 2.10 M | 8.41 M |
| CVG-AE (encoder) | 1.08 M | 6.45 M | 51.63 M |
| TSTCT | 3.18 M | 25.33 M | 202.21 M |
| Decoder | 0.08 M | 0.40 M | 3.35 M |
| Memory heads (slow weights) | 0.52 M | 2.10 M | 8.39 M |
| TAAFT (trunk + lenses) | 8.23 M | 58.06 M | 433.55 M |
| Forecaster (dynamics + policy/value) | 1.92 M | 13.62 M | 104.29 M |
| Advisor (policy/value) | 1.25 M | 9.07 M | 69.55 M |
| Verifier (step-label PRM + calibration) | 0.86 M | 6.57 M | 51.44 M |
| **Total (excl. Generator)** | **17.7 M** | **123.7 M** | **932.8 M** |

## Shapes behind each tier
| | S | M | L |
|---|---|---|---|
| relation planes × CVG-AE layers × width | 4 × 2 × 64 | 4 × 3 × 128 | 6 × 4 × 256 |
| latent z (continuous + G×C categorical) | 32 + 4×8 | 64 + 8×16 | 128 + 16×32 |
| TSTCT blocks × width (heads) | 4 × 256 (4) | 8 × 512 (8) | 16 × 1024 (16) |
| TAAFT blocks × width (heads), decoder-style | 6 × 256 (4) | 12 × 512 (8) | 24 × 1024 (16) |
| Forecaster dynamics / Advisor / Verifier-PRM blocks | 2 / 1 / 1 | 4 / 2 / 2 | 8 / 4 / 4 |
| hashed categorical rows × field-embedding width | 16,384 × 32 | 32,768 × 64 | 65,536 × 128 |

Assumptions, each a sizing choice to review:
- 7 node kinds and 4 hyperedge kinds per plane;
- TAAFT cross-attends to the Environment cache (so decoder-style blocks);
- 7 TAAFT lenses, one 2-layer MLP each;
- about 700 ATT&CK actions and about 256 D3FEND actions;
- 14 stage classes, and K = 32 for the hazard head.

## What the numbers say
- **The weights are small.** In fp16 they take 34 MiB (S), 236 MiB (M) and 1.8 GiB (L).
  - S fits comfortably in the RAM of a small single-board computer.
  - M fits one workstation GPU.
  - L needs a server-class GPU for training; inference is modest.
- **TAAFT is the largest part (≈ 45–47 %),** as the adversary-analysis core should be. TSTCT is next (≈ 18–22 %).
- **CVG-AE grows with (node kinds × planes × layers)** because every kind has its own parameters
  (heterogeneity). Sharing one MLP plus kind embeddings would cut it several-fold. That is an option,
  not a decision.
- **Memory is the real constraint, and it is the KV cache, not the weights.** Keys and values cost
  2 · L · d · 2 bytes per cached state update in fp16:

  | | S | M | L |
  |---|---|---|---|
  | Environment (TSTCT) cache per state update | 4 KiB | 16 KiB | 64 KiB |
  | … at 100 updates/s | 0.03 TiB/day | 0.13 TiB/day | 0.51 TiB/day |
  | … at 1,000 updates/s | 0.32 TiB/day | 1.29 TiB/day | 5.15 TiB/day |

  An uncompressed cache cannot be the months-long memory the owner requires (Q-04, Q-19). It must
  be bounded working memory with compression and retrieval behind it. Candidates:
  - Titans-style neural memory;
  - Memorizing-Transformers-style kNN retrieval;
  - an append-only event log to rebuild from.

  This is exactly held decision D-15, with proposals P-02 and P-18. The update rates above are
  illustrative; stage-1 analysis will measure real ones per site.
