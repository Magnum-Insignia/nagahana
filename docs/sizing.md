# Parameters and compute of L (the NagaHana model)

NagaHana is one model, L (owner, 2026-10-02: smaller variants are not part of NagaHana). Its
parameters are counted on the built modules and its compute profile is computed from them.

- **Parameters:** generated with `python -m nagahana.lab.sizing` (src/nagahana/lab/sizing.py), which
  calls `models.nagahana.count_parameters(preset("L"))`: every component is built on the meta
  device (no memory) and its parameters are counted. These exact counts replace the pre-build
  budget of 939.1 M (TAAFT at 24 blocks, 4·d GELU blocks); the build scaled TAAFT to 34 blocks
  (owner, 2026-10-02: "scale up taaft to 500m+") and uses SwiGLU blocks with QK-norm.
- **Excluded:** the Generator (training only, D-40).

## Parameter count

| Component | Parameters | Millions | Share |
|---|---:|---:|---:|
| Input layer (FieldEncoder) | 9,409,792 | 9.41 M | 0.8 % |
| CVG-AE | 117,852,056 | 117.85 M | 10.4 % |
| Decoder | 6,309,600 | 6.31 M | 0.6 % |
| TSTCT | 214,722,872 | 214.72 M | 18.9 % |
| Long-term memory (slow weights) | 7,340,032 | 7.34 M | 0.6 % |
| TAAFT (blocks, lenses, readouts, memory probes) | 515,800,696 | 515.80 M | 45.5 % |
| Forecaster | 115,942,095 | 115.94 M | 10.2 % |
| Advisor | 72,931,585 | 72.93 M | 6.4 % |
| Verifier (process reward, trust and calibration heads) | 73,959,939 | 73.96 M | 6.5 % |
| **Total (Generator excluded)** | **1,134,268,667** | **1,134.3 M** | 100 % |

The Generator, outside the count, has 107,992,101 parameters at the canonical 54-column layout of
the data windows and 92,767,779 at the PCAP adapter's 45 columns. Its count depends on the column
layout because its value bins, embeddings and output heads are per column. (The 87,770,144 in
docs/assumptions/generator.md was measured when the PCAP layout had 40 columns.)

## Shapes (build-spec §4, `preset("L")`)
| | L |
|---|---|
| FieldEncoder: field width; update vector; hashed rows | 128; 256; 65,536 |
| CVG-AE: relation planes × layers × width | 6 × 4 × 256 |
| latent z (continuous + G×C categorical) | 128 + 16×32 (640) |
| TSTCT: blocks × width (heads: spatial / temporal / causal) | 16 × 1024 (16: 4 / 8 / 4) |
| TSTCT keys per query: spatial + temporal + causal | 64 + 512 + 32 |
| TAAFT: decoder-style blocks × width (heads), reading TSTCT's cache | 34 × 1024 (16), block map 34 → 16 |
| TAAFT: adversary slots; hypothesis width; long-term memory probe keys | 16; 256; 32 |
| SwiGLU hidden width (every 1024-wide block) | 2,816 |
| Forecaster / Advisor / Verifier-PRM blocks × width | 8 / 4 / 4 × 1024 |

## What the numbers say
- **The weights are small:** they are stored and served in single precision (fp32, D-54):
  1,134,268,667 × 4 bytes = 4.54 GB = 4.23 GiB. No 16-bit weights are used (the half-precision
  size, 2.11 GiB, is a reference figure only). Outputs (P_inf, posteriors, energies, calibration)
  are computed in float64 (D-54); they are per-trigger tensors of a few KiB and do not change the
  memory account. The Environment and Imagination caches are fp32 as well (D-54 follow-up). Training
  needs server-class accelerators; the weights are about 1.5 % of inference memory.
- **TAAFT is the largest part (45.5 %),** as the adversary-analysis core should be. TSTCT is next
  (18.9 %). Each TAAFT block reads TSTCT's cached keys and values with its own query and output
  projections only, so it has no cross-attention key or value weights of its own.
- **The CVG-AE grows with (node kinds × planes × layers)** because every node kind has its own
  typed MLP in every plane and layer (heterogeneity).
- **Memory is the binding constraint, and it is the KV cache, not the weights.** Keys and values
  are stored in single precision (fp32, D-54) and cost 2 · L · d · 4 bytes per cached state: 128 KiB
  per state, so 1.03 TiB/day at 100 cached states per second and 10.30 TiB/day at 1,000 (each state
  update writes two) if nothing were bounded.

  An uncompressed cache cannot be the months-long memory the owner requires (Q-04, Q-19). The build
  bounds it: log-time buckets per entity, Titans-style long-term memory, and the event log to rebuild
  from (AS-11, standing for held D-15 with proposals P-02 and P-18).

## Compute profile of L (owner, 2026-10-01; built model, 2026-10-02)

Generated with `python -m nagahana.lab.compute` (src/nagahana/lab/compute.py) from the built
modules: parameters from `count_parameters`, shapes from `preset("L")`, small per-row maps (heads,
decoders, lens maps, time encodings) read from the built modules, and the schedule the inference
engine runs (`inference/engine.py`).

How the profile is checked (`tests/test_lab_compute.py`, PyTorch's own count, `FlopCounterMode`):
- **Blocks at L shapes** (meta device): the SelfBlock in the dense and in the gathered layout, the
  TAAFT block with its past, cross and long-term memory keys, and the CVG-AE plane layer. Each
  formula equals PyTorch's count exactly.
- **Every profiled component, composed on the real modules** at the test fixture's widths: input
  layer, CVG-AE, a `TSTCT.step` and a dense thinking pass, a TAAFT thinking pass and descent step,
  `Forecaster.imagine`, the Verifier's process reward and the long-term memory write. Each equals
  PyTorch's count exactly. One known exception is stated in the test: PyTorch evaluates a one-key
  einsum as an elementwise product, which its counter does not see.

**Operating point.** The owner asked for a good case, between light and worst, and chose the heavier
working context. The run-time budgets are the configuration's defaults (tested).

- **Working context:** 4,096 active entities, 512 states per entity, 64 spatial keys and 32 causal
  keys per query; the causal gate scores its cap of 256 candidates.
- **Per state update:**
  - the FieldEncoder over the 54 canonical columns;
  - the CVG-AE on the two positions' local subgraphs, 32 nodes in all, each node in 4 hyperedges
    per plane (one per hyperedge kind);
  - one `TSTCT.step` per position: the memory stream and R = 4 thinking passes. Each head scores
    the 512 keys of the longest head-group set, because the step pads the three key sets to one length.
- **Per Forecaster trigger:**
  - TAAFT over 4,112 tokens (4,096 entities and 16 adversary slots): memory stream, R = 4 thinking
    passes, I = 8 descent steps on the lens energies;
  - imagination: N = 200 routes of K = 12 steps, each step evaluating B = 8 candidates one step
    ahead with their exposure;
  - the Verifier's process reward of every imagined step;
  - the long-term memory write.
- **Server:** a sustained 18,400 state updates per second, one trigger per 60 s window, and
  1.2 × 10¹⁴ FLOP/s of effective compute (the thesis's order-of-magnitude assumption) on four 80 GB
  accelerators, the number the fp32 inference memory needs.
- **Cache precision:** every K/V cache is stored and counted in fp32, 4 bytes per element (D-54
  follow-up, owner 2026-10-02: "Caches fp32 too"): the Environment cache, the Imagination store,
  TAAFT's gathered view of the Environment cache and the Forecaster's route caches (inference runs
  without autocast). Training activations are counted in bf16 (AS-39).
- **Training:**
  - micro-batches of 8 sequences × 2,048 states;
  - TSTCT dense over each window;
  - R = 1 + Poisson(3) clipped at 8, with gradients through the last 2 passes (AS-07);
  - two TAAFT tokens per state update;
  - a corpus of 4 × 10⁹ state updates plus 4 × 10⁹ from Generator variants;
  - Stage 3 makes two passes over them, Stages 4 and 5 one each, and Stage 5 adds 4 × 10⁹
    imagined states;
  - an H100 at 40 % utilisation.

| Figure | Value |
|---|---:|
| **Weights and state** | |
| Parameters (Generator excluded; built model, meta device) | 1,134,268,667 |
| Weights, stored and served in single precision (fp32, 4 B per parameter, D-54) | 4.23 GiB (4.54 GB) |
| … half precision for reference only (not used: no 16-bit weights, D-54) | 2.11 GiB |
| Training state (mixed-precision Adam, 16 B per parameter) | 16.90 GiB |
| **Context window** | |
| TSTCT keys per query (spatial + temporal + causal) | 608 (64 + 512 + 32) |
| Working context | 4,096 entities × 512 states = 2,097,152 cached states |
| Environment cache per cached state (fp32, D-54) | 128 KiB |
| Working Environment cache | 256.00 GiB |
| Time an average entity's 512 states cover at the sustained rate | 57 s |
| TAAFT tokens; self keys; cross keys (32 own + 64 neighbours + 32 memory) | 4,112; 4,120; 128 |
| Imagination store (8 triggers × 4,112 tokens, 272 KiB each) | 8.53 GiB |
| TAAFT working set of one block (gathered cross keys; self-attention scores) | 3.01 GiB; 3.03 GiB |
| Forecaster route caches with the 8 lookahead copies | 8.46 GiB |
| Inference memory in all | 283.26 GiB (304.1 GB) |
| 80 GB accelerators it needs (server) | 4 (320 GB) |
| Retained Environment per day (24 B per update) | 35.53 GiB |
| **Per state update** | |
| Input layer (54 columns) | 7.70 MFLOP |
| CVG-AE (32 nodes, 4 incidences per node and plane) | 1.52 GFLOP |
| TSTCT (2 positions × 16 blocks, memory stream + R 4 thinking passes) | 4.22 GFLOP |
| … of which the thinking passes | 3.02 GFLOP |
| Total, and per parameter | 5.75 GFLOP (5.07 FLOP per parameter) |
| At 18,400 updates/s | 105.8 TFLOP/s |
| **Per Forecaster trigger** (K 12, N 200, R 4, I 8, B 8) | |
| TAAFT over 4,112 tokens (34 blocks, memory stream + R thinking passes, descent) | 31.12 TFLOP |
| Imagination: N K = 2,400 route steps, 8 candidates each | 4.85 TFLOP |
| Verifier (process reward of every imagined step) | 1.59 TFLOP |
| Long-term memory write | 103.08 GFLOP |
| Total, and compute time at 1.2 × 10¹⁴ FLOP/s | 37.66 TFLOP (0.31 s) |
| Advisor solve, on demand (W 64, depth 3, 50 rollouts) | 245.13 TFLOP |
| Server load: updates plus one trigger per 60 s | 89 % of 1.2 × 10¹⁴ FLOP/s |
| **Training** | |
| Activations per micro-batch (8 × 2,048 states, 3 stored passes): TSTCT / TAAFT | 27.38 GiB / 74.11 GiB |
| … with attention scores stored (TSTCT, dense over the window) | 75.38 GiB |
| … with full recomputation: TSTCT / TAAFT | 2.07 GiB / 3.91 GiB |
| Stage 3 / 4 / 5 compute | 2.62e+20 / 3.21e+20 / 4.01e+20 FLOP |
| Total training compute | 9.84e+20 FLOP over 3.2e+10 state updates |
| GPU time at 40 % of an H100's 989 TFLOP/s | 691 GPU-hours |

What the profile says:

- **Both memory and compute bind on the server at this operating point.**
  - Memory sets the number of accelerators. The working Environment cache, stored in fp32 (D-54),
    needs 256 GiB, and inference needs 283 GiB in all (304.1 GB), so four 80 GB accelerators
    (320 GB) are needed; three (240 GB) are not enough. The headroom is 15.9 GB (5 %), so a larger
    working context needs a fifth accelerator or a bounded cache (AS-11). The fp32 weights
    (4.23 GiB, D-54) are 1.5 % of it.
  - The compute figures below are stated at the assumed 1.2 × 10¹⁴ FLOP/s, unchanged by the move
    to four accelerators, so they are conservative for this server.
  - Compute is now close to the budget. Streaming uses 88 % of the server's effective compute and
    triggers 0.5 %, so 89 % in all. A trigger needs 0.31 s of compute, within the server's 0.38 s.
  - The pre-build profile (939.1 M parameters, one TSTCT pass, the reference CVG-AE layer) had 1.77
    GFLOP per update and 27 % load. The built step runs R = 4 thinking passes per new state, which
    are 53 % of an update. The typed CVG-AE layer maps keys and values per incidence pair.
  - R is a run-time budget (D-44). At R = 1 an update costs 3.49 GFLOP and the load is 54 %. A site
    whose sustained rate is near this one can trade passes for headroom; the cache is the same
    whatever R is (AS-06).
- **The working context is short in time for busy entities.**
  - At the sustained rate, 512 states cover about a minute of an average entity's activity.
  - Quiet entities are covered for far longer.
  - Longer horizons come from the retention memory and the event log (D-15), not from the cache.
- **TAAFT dominates a trigger: 83 % of its FLOPs.**
  - Almost all of it is the 34 blocks over the memory stream and four thinking passes. Each token
    attends to every current token (4,120 self keys) and to 128 cross keys.
  - The lens energies and the descent are 1 % of TAAFT. Imagination is 13 % of a trigger.
- **An Advisor solve is on demand and costs about 2.0 s of server compute.** It re-imagines the
  baseline and up to 192 counter sequences.
- **Training is modest for the size.**
  - It processes 3.2 × 10¹⁰ state updates, about 28 per parameter. The compute-optimal ratio of
    Hoffmann et al. 2022 is about 20 tokens per parameter.
  - That is 691 H100-hours, or about 3.6 days on eight accelerators.
  - TSTCT's activations (27.4 GiB) fit beside the training state (16.9 GiB) on one 80 GB
    accelerator. TAAFT's 74.1 GiB do not, so TAAFT trains with block recomputation (3.9 GiB) or
    smaller micro-batches. Attention scores need not be stored.

Sources of the rules: Kaplan et al. 2020 (arXiv:2001.08361, training ≈ 6 N per position); Korthikanti
et al. 2022 (arXiv:2205.05198, activations); Rajbhandari et al. 2020 (arXiv:1910.02054, training
state); Hoffmann et al. 2022 (arXiv:2203.15556, tokens per parameter); NVIDIA H100 datasheet (989
TFLOP/s dense BF16).
