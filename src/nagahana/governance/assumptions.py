"""Engineering assumptions of the L build (owner, 2026-10-02), as code.

Why this exists beside `decisions.py`
-------------------------------------
The owner asked for the full model "with your assumptions of wherever if you think are missing more
details" and for "a small doc, write down all those assumptions, with reasoning & justification,
along with any findings to cite". An assumption is **not** a decision:

- a held decision in `decisions.py` stays HELD; the assumption that stands in for it says so in
  `stands_for`;
- code that relies on an assumption calls `assume("AS-xx")`, which returns the entry and records the
  use, so the report can list exactly which code depends on which assumption;
- `strict_mode(True)` makes every `assume` raise, which reproduces the pre-build "held means blocked"
  behaviour for audits.

The human-readable document is `docs/assumptions.md` (generated from this registry plus the evidence
notes written during the build).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from nagahana.core.errors import DecisionHeld
from nagahana.governance.assumption_types import Assumption
from nagahana.governance.assumptions_build import BUILD_ENTRIES

__all__ = ["Assumption", "all_entries", "assume", "get", "ids", "strict_mode", "uses"]

_A = Assumption
#: AS-01 … AS-41: the lead's assumptions (written here). AS-100 … : the engineers' assumptions,
#: generated from docs/assumptions/*.md by tools/gen_assumptions.py and appended below.
_ENTRIES: tuple[Assumption, ...] = (
    _A("AS-01", "declared-planes", "Six declared relation planes", ("D-04",),
       "connectivity, services, identity, remote_admin, name_resolution, ot_control; formed deterministically "
       "from protocol and port evidence in the data model.",
       "Declared planes are auditable and need no data to learn them; every plane is a relation an analyst "
       "already reasons in. Learned planes stay possible later (flagged as inferred).",
       ("RouteNet builds structure from known facts (Rusek et al., IEEE JSAC 2020, arXiv:1910.01508).",),
       ("graph/", "models/vocab.py")),
    _A("AS-02", "hyperedge-kinds", "Four hyperedge kinds per plane", ("detail",),
       "session (TCP), exchange (UDP/other), group (a multicast member), fan (one initiator → ≥ f responders "
       "within Δ_fan, a scan or sweep as one hyperedge).",
       "A scan is one act touching many hosts; as one hyperedge its structure is visible at once instead of "
       "being smeared over many pairwise edges.",
       ("Feng et al., 'Hypergraph Neural Networks', AAAI 2019 (arXiv:1809.09401).",), ("graph/",)),
    _A("AS-03", "cvgae-attention-aggregation", "CVG-AE: typed attention aggregation on sparse incidence", ("detail",),
       "Node→hyperedge and hyperedge→node segment-softmax attention with HGT-style typed parameters, "
       "cross-plane coupling ω, RWSE and hop bias; sparse scatter ops in plain PyTorch.",
       "Mean aggregation (the reference layer) cannot weight the one suspicious hyperedge among many benign ones.",
       ("Hu et al., 'Heterogeneous Graph Transformer', WWW 2020 (arXiv:2003.01332).",), ("models/cvgae",)),
    _A("AS-04", "reconstruction-target", "Reconstruction target", ("D-11a", "P-20"),
       "Masked likelihood of the *contributing* cells of the update behind each position, plus candidate "
       "hyperedges (observed + sampled negatives).",
       "Only observed values are evidence (D-41); scoring absent cells would teach the model to invent them.",
       (), ("models/decoder", "training/")),
    _A("AS-05", "dynamics-prior", "Latent prior = TSTCT transition model; DreamerV3 KL settings", ("detail",),
       "q(z|obs) from CVG-AE, p(z_next|Environment) from TSTCT; KL balancing β_dyn 0.5 / β_rep 0.1, free bits "
       "1 nat, 1 % unimix on categoricals; posterior mean at inference.",
       "This *is* the problem statement's P(S_{t+1}|S_t), learned with the settings that made RSSM world models "
       "stable across domains.",
       ("Hafner et al., 'Mastering Diverse Domains through World Models' (DreamerV3), arXiv:2301.04104, §3.",),
       ("models/tstct", "training/")),
    _A("AS-06", "two-stream-loop", "Two-stream weight-tied loop", ("D-43 note",),
       "Memory stream once (its K/V are the cache), thinking stream R passes reading those K/V, input "
       "re-injection, no pass embedding.",
       "Makes training and inference compute the same function and makes the cache independent of R (nn/loop.py).",
       ("Dehghani et al., Universal Transformers, arXiv:1807.03819; Geiping et al., arXiv:2502.05171.",
        "tests/test_nn.py: incremental cache path equals the parallel path for R = 3."),
       ("nn/loop.py", "models/tstct", "models/taaft")),
    _A("AS-07", "loop-training", "R sampling and truncated backprop", ("D-43 note",),
       "R = 1 + Poisson(3), clipped to [1, 8], per batch; gradients through the last 2 passes.",
       "A model trained at one fixed R is not known to improve with more passes; random R teaches it to.",
       ("Geiping et al. 2025, §3.3 (random iteration counts, truncated backprop).",), ("training/",)),
    _A("AS-08", "r-per-model", "One R per looped model", ("D-44 note",),
       "TSTCT and TAAFT each take their own R (same default, 4); both are recorded.",
       "The owner's 'the number of passes R' was not explicit; separate budgets cost nothing and keep both readings.",
       (), ("roles/contracts.py",)),
    _A("AS-09", "tstct-head-split", "TSTCT head split", ("detail",),
       "16 heads: 4 spatial, 8 temporal, 4 causal.",
       "Temporal context (512 keys) is the largest and carries low-and-slow evidence; spatial and causal are "
       "sparser (64 / 32 keys).", (), ("models/tstct",)),
    _A("AS-10", "causal-heads", "Causal heads as learned Granger-style gates", ("TSTCT causal design",),
       "Candidates: strictly earlier states of contacted entities within Λ_lag; gate σ(MLP(q,k,φ(Δt))) as "
       "log g bias; top-32; L1 sparsity.",
       "Predictive (Granger) influence is learnable from passive data; it is reported as such, never as "
       "identified causal structure.",
       ("Tank et al., 'Neural Granger Causality', IEEE TPAMI 2021 (arXiv:1802.05842).",), ("models/tstct",)),
    _A("AS-11", "bounded-environment", "Bounded working Environment with log-time buckets", ("D-15", "P-02", "P-18"),
       "Per entity 512 slots in log-time buckets (1 ms … weeks); overflow merges the two most similar slots "
       "with a count n (log n logit bias); Titans-style long-term memory updated once per trigger with α_k = "
       "α (D-36 form); ColumnarUpdates is the durable event log; caches keyed to the model hash.",
       "Volume can compress a bucket but never evict older buckets (D-36 spirit); weeks of history fit "
       "(ai-mod-arch §4); mid-term ages keep reserved slots (lost-in-the-middle, §11c).",
       ("Behrouz et al., 'Titans', arXiv:2501.00663; Liu et al., 'Lost in the Middle', arXiv:2307.03172.",),
       ("memory/",)),
    _A("AS-12", "trigger-policy", "Trigger policy", ("D-02",),
       "Fixed cadence (60 s) plus at most one priority trigger per cadence interval when marginal energy jumps.",
       "Cadence cannot be driven by attack volume; the cap bounds what an attacker can force.", (), ("inference/", "data/")),
    _A("AS-13", "taaft-reads-tstct-kv", "TAAFT reads TSTCT's cached K/V directly", ("detail",),
       "Cross-attention of TAAFT block b uses TSTCT block ⌊b·16/24⌋'s K/V with TAAFT's own query/output "
       "projections; entity tokens read 32 own + 64 neighbour states; Imagination = TAAFT memory-stream K/V "
       "of the last 8 triggers.",
       "The literal design ('TAAFT will work on … TSTCT's kv cache' [A-14]) and no second Environment cache.",
       (), ("models/taaft",)),
    _A("AS-14", "lens-energies", "Hypothesis space and lens energy forms", ("D-11b", "D-24", "D-26"),
       "y per entity (256) + adversary slots; E_total = E_bt + E_game + E_info + E_top + E_time + E_cause + "
       "λΦ_phys with the forms of build-spec §2.7; mechanism design inside E_game, noise features inside "
       "E_info; suspicion floor φ = 0.01.",
       "Each lens becomes a term whose gradient share is an explanation (D-42).",
       ("Du, Li & Mordatch, NeurIPS 2020 (arXiv:2004.06030) on composing energies.",), ("models/taaft",)),
    _A("AS-15", "lambda-phys", "λ_phys", ("D-42 λ",),
       "λ_phys = 0.1 on residuals normalised by their field scale.",
       "Large enough that impossible states dominate the energy, small enough not to swamp the lens terms "
       "on in-bound states (where Φ = 0 anyway).", (), ("models/taaft", "training/")),
    _A("AS-16", "ebt-training", "Energy training: unrolled descent + two readings", ("P-09",),
       "Loss on ŷ_S after S ∈ {2..8} descent steps with learned step size; context dropout 0.1 for the "
       "marginal reading E(∅, y).",
       "The EBT recipe trains the energy through the thinking it is used for; the marginal reading avoids "
       "'continue the attack' bias (DESIGN_LOG P-09).",
       ("Gladstone et al., 'Energy-Based Transformers are Scalable Learners and Thinkers', arXiv:2507.02092; "
        "Ho & Salimans, arXiv:2207.12598.",), ("models/taaft", "training/")),
    _A("AS-17", "adversary-reward", "Adversary reward", ("D-11c",),
       "r^A = Δ kill-chain progress + β·𝟙[infiltration] − κ·exposure (marginal-energy novelty).",
       "A realistic adversary progresses and avoids standing out; a worst-case floor is a separate run setting.",
       (), ("models/forecaster",)),
    _A("AS-18", "infiltration-definition", "Infiltration state", ("D-03a",),
       "An internal entity in a post-initial-access tactic (vocab.INFILTRATION_STAGES); configurable.",
       "Matches 'before compromise is completed' in the problem statement: initial access is the boundary.",
       (), ("models/vocab.py", "data/")),
    _A("AS-19", "stage-classes", "Stage classes", ("detail",),
       "15 classes: 'none' + the 14 ATT&CK Enterprise tactics; ICS tactics map onto them until extended.",
       "Covers the five named in the problem statement and the rest of the matrix.",
       ("https://attack.mitre.org/tactics/enterprise/",), ("models/vocab.py",)),
    _A("AS-20", "technique-slots", "ATT&CK technique action slots", ("detail",),
       "700 slots: known technique IDs mapped by a table, unknown ones hashed.",
       "Order of magnitude of Enterprise + ICS techniques and sub-techniques (sizing).", (), ("models/forecaster",)),
    _A("AS-21", "mppi-routes", "MPC-guided route sampling", ("detail",),
       "N routes × K steps by π_A with an MPPI re-weighting of the top-B actions by one-step lookahead; "
       "duplicates merged (≤ N distinct, D-46).",
       "Model-predictive improvement of a learned policy is the TD-MPC2 recipe; MPPI is its sampling core.",
       ("Williams et al., ICRA 2017 (MPPI); Hansen et al., TD-MPC2, arXiv:2310.16828.",), ("models/forecaster",)),
    _A("AS-22", "coupling-staged", "TAAFT ↔ heads coupling: STAGED", ("D-12",),
       "Stop-gradient into TAAFT in the first part of stage 5, joint fine-tune after.",
       "Penalises analysis and forecasting separately first, as the owner wished, then lets them co-adapt.",
       (), ("models/heads", "training/")),
    _A("AS-23", "disruption-pricing", "Disruption pricing", ("D-03b",),
       "cost = criticality(entity kind) × disruption(action), both tables in config; OT highest.",
       "OT availability first (D-34).", (), ("models/advisor",)),
    _A("AS-24", "risk-measure", "Expected vs worst case", ("D-03c",),
       "Rank by CVaR_0.2 of ΔP_inf over routes; report expected and worst beside it.",
       "CVaR sits between the two and is coherent.", ("Rockafellar & Uryasev, J. Risk 2000.",), ("models/advisor",)),
    _A("AS-25", "verifier-truth", "Verifier mechanics and initial truth", ("P-10", "D-07"),
       "PRM + trust value head + temperature policy head; Brier RLCD reward excluding responded-to; CUSUM, "
       "systematic gap, Page–Hinkley drift; split-conformal thresholds; dataset annotations count as "
       "human-supplied truth for initial training.",
       "Dataset labels were made by people (D-17: human feedback is outside the threat model).",
       ("Lightman et al., arXiv:2305.20050; Damani et al., arXiv:2507.16806; Angelopoulos & Bates, arXiv:2107.07511.",),
       ("models/verifier",)),
    _A("AS-26", "site-adapters", "Site calibration adapters", ("D-13",),
       "LoRA rank 16 on TSTCT/TAAFT attention projections + Verifier temperature; 24 h unlabelled site "
       "traffic + 50 analyst-confirmed alerts; applied only on a HumanCommand.",
       "Small, removable, auditable deltas instead of retraining (ai-mod-arch §3f).",
       ("Hu et al., LoRA, arXiv:2106.09685.",), ("nn/lora.py", "training/")),
    _A("AS-27", "generator-families", "Generator families in v1", ("D-14",),
       "Observability sliding + signature variation + topology variation (deterministic), masked-generative "
       "(MaskGIT-style; autoregressive by order), TabDDPM-style diffusion, JEM-style energy acceptance.",
       "Every family the owner named, ordered so the label-preserving-by-construction ones carry most volume.",
       ("Chang et al., MaskGIT, arXiv:2202.04200; Kotelnikov et al., TabDDPM, arXiv:2209.15421; Grathwohl et "
        "al., JEM, arXiv:1912.03263.",), ("models/generator",)),
    _A("AS-28", "generator-gate", "Generator physics gate enabled", ("P-11",),
       "Accept a variant only if Φ_phys ≤ τ (τ = 1e-3) and hard limits hold.",
       "The owner requires variants not to be impossible [A-17].", (), ("models/generator",)),
    _A("AS-29", "view-flattening", "Decoder view: typed multigraph", ("D-05",),
       "Planes kept as edge types; no merging.", "Lossless and simplest to audit.", (), ("models/decoder",)),
    _A("AS-30", "status-weights", "Status weights", ("D-29",),
       "OBSERVED 1.0, STALE 0.5, LOW_RELIABILITY 0.5 where a numeric weight is needed (physics masks, "
       "evidence counts); the encoder learns its own use of status through embeddings.",
       "Down-weighting without discarding; the learned embedding can override.", (), ("models/inputs", "physics/")),
    _A("AS-31", "numeric-encoding", "Numeric encoding", ("P-22",),
       "Signed log1p + periodic embeddings per column; slot embedding keyed by field ID.",
       "Magnitudes span 10 orders; periodic embeddings resolve fine differences.",
       ("Gorishniy et al., NeurIPS 2022, arXiv:2203.05556.",), ("nn/numeric.py", "models/inputs")),
    _A("AS-32", "block-design", "Block design", ("detail",),
       "Pre-norm RMSNorm, SwiGLU (8/3·d rounded to 256), QK-norm, no biases, learned null key.",
       "The stable large-transformer recipe.",
       ("Xiong et al., arXiv:2002.04745; Shazeer, arXiv:2002.05202; Dehghani et al., arXiv:2302.05442.",), ("nn/",)),
    _A("AS-33", "categorical-fields", "Categorical fields", ("detail",),
       "Encoded by hashed rows h(column, code); decoded into 64 service-class buckets.",
       "Ports are categories, not magnitudes; exact port reconstruction is not needed for the world model.",
       (), ("models/inputs", "models/decoder")),
    _A("AS-34", "label-mapping", "Dataset label → stage/technique mapping", ("detail",),
       "Tables per dataset (CIC-IDS2018, CTU-13, CIC-IoT-2023) mapping attack labels to ATT&CK tactics and "
       "techniques, documented in data/labels.py.",
       "Public datasets label attack families, not ATT&CK steps; the mapping is explicit and reviewable.",
       (), ("data/",)),
    _A("AS-35", "zero-shot-scope", "Zero-shot includes unseen networks", ("D-16",),
       "Novel families and held-out networks both count as zero-shot; reported separately.",
       "Matches the results chapter (P1–P3).", (), ("data/", "evaluation/")),
    _A("AS-36", "noise-features", "Noise features inside E_info", ("D-26",),
       "Per entity: periodogram peak ratio of inter-arrival times and aggregated-variance Hurst estimate.",
       "Beaconing is periodic under jitter; benign aggregates are self-similar.",
       ("Leland et al., IEEE/ACM ToN 1994; Hu et al., BAYWATCH, DSN 2016.",), ("models/taaft",)),
    _A("AS-37", "taaft-view", "TAAFT working view", ("detail",),
       "32 own states + 64 neighbour latest states per entity token; adversary slots read entities by self-attention.",
       "The compute profile's '32 refined states per entity'.", (), ("models/taaft",)),
    _A("AS-38", "decoder-shape", "Decoder conditioning", ("detail",),
       "Shared field head conditioned on role and plane membership; one candidate-edge head per plane.",
       "Fields belong to updates, not planes; edges belong to planes.", (), ("models/decoder",)),
    _A("AS-39", "precision", "Numerical precision", ("detail",),
       "bf16 autocast with fp32 master weights; norms, softmax, energies and time angles in fp32/fp64.",
       "Energy descent and long-time angles are precision-sensitive.", (), ("training/",)),
    _A("AS-40", "physics-on-telemetry-off", "Physics on incoming telemetry stays off", ("D-25",),
       "Not assumed: Target.OBSERVATION still raises.", "D-25 changes the meaning of physics; left to the owner.",
       (), ("physics/term.py",)),
    _A("AS-41", "positions-per-update", "Two positions per update", ("detail",),
       "Initiator and responder states; the service entity is a graph node without its own position.",
       "Compute profile: 2 TSTCT positions per update.", (), ("data/", "models/tstct")),
)

_ENTRIES = _ENTRIES + BUILD_ENTRIES
if len({a.id for a in _ENTRIES}) != len(_ENTRIES):
    raise RuntimeError("duplicate assumption IDs in the registry")
# Slugs of generated entries may collide with hand-written ones; IDs are the primary key, so a slug
# resolves to the first entry that uses it.
_BY_KEY: dict[str, Assumption] = {a.slug: a for a in reversed(_ENTRIES)} | {a.id: a for a in _ENTRIES}
_USES: dict[str, set[str]] = defaultdict(set)
_STRICT = False


def strict_mode(on: bool) -> None:
    """When on, every `assume` raises (audit mode: run as if every assumption were still held)."""
    global _STRICT
    _STRICT = on


def get(key: str) -> Assumption:
    """Look up an assumption by ID or slug (KeyError if unknown)."""
    return _BY_KEY[key]


def assume(key: str, *, by: str = "") -> Assumption:
    """Use an assumption: returns it and records the user (`by`, e.g. the module name)."""
    a = get(key)
    if _STRICT:
        raise DecisionHeld(f"{a.id} ({a.slug}) is an engineering assumption standing for {a.stands_for}; strict mode is on.")
    if by:
        _USES[a.id].add(by)
    return a


def uses() -> dict[str, frozenset[str]]:
    """Which modules used which assumptions in this process."""
    return {k: frozenset(v) for k, v in _USES.items()}


def all_entries() -> tuple[Assumption, ...]:
    """Every assumption, in order."""
    return _ENTRIES


def ids(entries: Iterable[Assumption]) -> tuple[str, ...]:
    return tuple(a.id for a in entries)
