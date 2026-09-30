"""The decision registry: DESIGN_LOG.md §2 (held) and §3 (proposals), as code.

What this is
------------
The single place in code that says, for every design question, whether it is:

- **DECIDED**: the owner decided. Code may rely on `value`.
- **HELD**: the owner has not decided, or is holding it. Code must not choose. It calls
  `require(...)`, which raises `DecisionHeld` [Q-39].
- **PROPOSED**: Engineering proposed it and it is not approved. Code implementing it runs only if its
  ID is enabled in config (`enabled_proposals`) (DESIGN_LOG.md §3).
- **SUPERSEDED**: kept for history. `superseded_by` points to the replacement.

Why it exists
-------------
The owner asked for a codebase that is "fully templated so that we can keep updating with more and
more decisions and logic on the go" [Q-45], and "please don't just default to any without my
permission" [Q-39]. This registry makes that mechanical:
- a held decision blocks exactly the code that depends on it and nothing else;
- deciding it is a one-line status change here, plus the config value in `conf/`;
- `python -m nagahana decisions` lists everything and where the code depends on it
  (see `governance/report.py`).

How to update (the workflow)
----------------------------
1. The owner decides in chat or in a document.
2. Change the entry here: status DECIDED, `value` in plain words, add the source (Q-/A- ID or a
   DESIGN_LOG date).
3. Replace the matching `???` in `conf/` if the decision is a parameter.
4. Update DESIGN_LOG.md, then regenerate `docs/decisions.md`
   (`python -m nagahana decisions --write-docs`).

IDs
---
- `D-01` to `D-11` keep DESIGN_LOG.md §2 numbering. D-03 and D-11 are split into a/b/c.
- `D-12` onward were added on 2026-09-29 from the updated `ai-mod-arch` and the new pipeline
  image [I-01].
- `D-30` onward record foundational decisions made earlier (so the code can cite them).
- `P-xx` are proposals.

Each entry also has a readable `slug`, so code can say
`require("taaft-policy-coupling")` instead of a number.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Iterable
from dataclasses import dataclass

from nagahana.core.errors import DecisionHeld, ProposalNotEnabled


class Status(enum.Enum):
    """Lifecycle of a design question. See the module docstring for the meaning of each."""

    DECIDED = "decided"
    HELD = "held"
    PROPOSED = "proposed"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class Decision:
    """One design question and its current state.

    Attributes
    ----------
    id, slug:
        Stable identifiers. Code may use either.
    title:
        Short name.
    status:
        See `Status`.
    question:
        What has to be decided, in plain words.
    value:
        For DECIDED entries: what was decided, in plain words.
    options:
        Known options (for HELD/PROPOSED). Listing an option does not favour it.
    sources:
        Where it comes from: quote IDs (`Q-`, `A-`, `I-`), DESIGN_LOG dates, ARCHITECTURE.md sections.
    affects:
        Modules that depend on it (informational; `report.py` also finds uses automatically).
    superseded_by:
        For SUPERSEDED entries.
    note:
        Anything a reader must know, e.g. that a DECIDED value is the engineer's reading of the owner's
        words and should be confirmed.
    """

    id: str
    slug: str
    title: str
    status: Status
    question: str
    value: str | None = None
    options: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    affects: tuple[str, ...] = ()
    superseded_by: str | None = None
    note: str | None = None


D, H, P, S = Status.DECIDED, Status.HELD, Status.PROPOSED, Status.SUPERSEDED

_ENTRIES: tuple[Decision, ...] = (
    # ------------------------------------------------------------------ DESIGN_LOG.md §2 (1-11)
    Decision(
        "D-01", "advisor-reads-environment", "Does the Advisor read the Environment?", D,
        "Does the Advisor (formerly Planner) read the Environment, or only Imagination?",
        value="It reads both Environment and Imagination (both KV caches).",
        sources=("A-15", "A-20", "DESIGN_LOG 2026-09-28 §2.1"),
        affects=("memory/access.py", "roles/advisor.py"),
    ),
    Decision(
        "D-02", "forecaster-trigger-policy", "What fires a Forecaster computation?", H,
        "Event-driven triggers could let attack volume drive retention. Fixed cadence plus capped "
        "priority triggers, or something else?",
        options=("fixed cadence", "fixed cadence + capped priority triggers", "other"),
        sources=("DESIGN_LOG §2.2", "Q-33"),
        affects=("memory/retention.py", "roles/forecaster.py"),
    ),
    Decision(
        "D-03a", "attacker-success-criterion", "What counts as the attacker succeeding?", H,
        "Defines the 'infiltration state' behind P_inf(k): a crown jewel reached, or any stage "
        "progression?",
        options=("crown jewel reached", "any ATT&CK stage progression", "per-site definition"),
        sources=("DESIGN_LOG §2.3a",),
        affects=("objectives/rewards.py", "roles/forecaster.py", "roles/advisor.py"),
    ),
    Decision(
        "D-03b", "disruption-pricing", "How is disruption priced per asset?", H,
        "Cost of a counter-measure per asset (OT availability first?).",
        sources=("DESIGN_LOG §2.3b",),
        affects=("objectives/rewards.py", "roles/advisor.py"),
    ),
    Decision(
        "D-03c", "advisor-risk-aggregation", "Expected-case or worst-case optimisation?", H,
        "How the Advisor aggregates over adversary policies.",
        options=("expected case", "worst case", "risk measure, e.g. CVaR"),
        sources=("DESIGN_LOG §2.3c",),
        affects=("roles/advisor.py",),
    ),
    Decision(
        "D-04", "relation-plane-formation", "How are relation planes formed?", H,
        "Declared from physics/protocols, learned, or declared + learned (learned ones flagged as "
        "inferred)?",
        options=("declared", "learned", "declared + learned"),
        sources=("DESIGN_LOG §2.4", "A-03"),
        affects=("graph/planes.py", "graph/builder.py", "models/cvgae"),
    ),
    Decision(
        "D-05", "decoder-view-flattening", "How does the Decoder flatten planes into one view?", H,
        "Typed multigraph, supra-graph (lossless), reducibility-based merging (De Domenico 2015), "
        "or a combination?",
        options=("typed multigraph", "supra-graph", "reducibility merge", "combination"),
        sources=("DESIGN_LOG §2.5",),
        affects=("models/decoder", "roles/decoder.py"),
    ),
    Decision(
        "D-06", "standards-reading", "Which standards does the data model follow?", H,
        "Is 'OSTF' the same as OCSF? Is CSTS the Rahman 2026 substrate (arXiv:2603.23459)?",
        sources=("DESIGN_LOG §2.6", "Q-27"),
        affects=("datamodel/layers.py",),
    ),
    Decision(
        "D-07", "verifier-training-placement", "Where does the Verifier train?", D,
        "Where does Verifier training sit in the training sequence?",
        value="Stage 5 (full training), on human feedback only. In stage 6 it calibrates under human "
        "supervision. It never feeds back automatically.",
        sources=("A-16", "A-21", "A-24", "I-01"),
        affects=("pipeline/stages.py", "roles/verifier.py"),
    ),
    Decision(
        "D-08", "threat-model-document", "Should the threat-model skeleton become a document?", H,
        "Should the POSG adversary model, the threats to NagaHana and the invariants become a "
        "document, and where? Scope is partly settled by D-17.",
        sources=("DESIGN_LOG §2.8",),
    ),
    Decision(
        "D-09", "tech-stack", "Which tech-stack additions are approved?", D,
        "Which libraries beyond the owner's stack (ARCH #18) may be used?",
        value="All candidates proposed on 2026-09-28 plus the owner's stack: "
        "PyTorch family (core, PyG incl. TGN, TorchRL, TorchMetrics), MLflow, DVC, HF Accelerate, "
        "DeepSpeed, JAX + Flax/Equinox, Kafka, NumPy/pandas; LightZero; Diffrax, Optimistix, "
        "jaxtyping; Pyro/NumPyro; BlackJAX; hypothesis; Hydra; ruff/mypy; MkDocs; "
        "Arrow/Polars/DuckDB; scipy/statsmodels; Triton/FlashAttention/CUDA Graphs; and, from the "
        "problem statement, Scapy/PyShark and Streamlit/Flask/CLI.",
        sources=("Q-45", "ARCH #18"),
        note="This is the engineer's reading of 'all the tech stack yours and mine together' [Q-45]. "
        "Please confirm.",
    ),
    Decision(
        "D-10", "repo-location", "Repository name and location", H,
        "This workspace or the Fedora workstation? Repository name?",
        sources=("DESIGN_LOG §2.10",),
        note="Started as `nagahana/` in this workspace at the owner's request on 2026-09-29 [Q-45]. "
        "Name and location can still change; no code depends on it.",
    ),
    Decision(
        "D-11a", "reconstruction-target", "What does CVG-AE reconstruct?", H,
        "Which fields and structures enter the reconstruction likelihood (observed fields only? "
        "edges? next state?)",
        sources=("DESIGN_LOG §2.11",),
        affects=("models/cvgae", "objectives/template.py"),
    ),
    Decision(
        "D-11b", "energy-jobs-v1", "Which energy-model jobs are in v1?", H,
        "TAAFT has the EBT at its core [A-19], and the Generator uses energy-based SSL [A-17]. "
        "Which other uses are in v1: Advisor shaping? novelty signal?",
        sources=("DESIGN_LOG §2.11", "A-17", "A-19"),
        affects=("models/taaft", "models/generator", "objectives/rewards.py"),
    ),
    Decision(
        "D-11c", "adversary-objective", "Adversary objective: realistic or worst-case floor?", H,
        "What the modelled attacker optimises, and the make-up of the policy population Π_A.",
        options=("realistic", "realistic + exploiters", "worst-case floor"),
        sources=("DESIGN_LOG §2.11",),
        affects=("models/heads", "objectives/rewards.py"),
    ),
    # ------------------------------------------------ New with ai-mod-arch 2026-09-29 (12-29)
    Decision(
        "D-12", "taaft-policy-coupling", "How do TAAFT and the policy/value heads couple?", H,
        "Joint autoregressive model, separate transformer + policy/value (separate penalties for "
        "analysis vs forecasting), or staged (separate first, joint fine-tune later)? Also: a "
        "detector-driven forecaster?",
        options=("joint", "separate", "staged"),
        sources=("A-19 item 4",),
        affects=("models/heads", "models/taaft", "pipeline/stages.py"),
        note="The owner asked for careful analysis ('take literature with a pinch of salt'). "
        "Code supports all three behind this gate so experiments can decide.",
    ),
    Decision(
        "D-13", "site-calibration-adapter", "What do site calibration adapters change?", H,
        "Output-only recalibration (temperature, isotonic, conformal thresholds) or small "
        "trainable adapters (low-rank), always applied on human command?",
        options=("temperature scaling", "isotonic", "conformal thresholds", "low-rank adapters"),
        sources=("A-24 stage 5", "I-01 stage 6"),
        affects=("roles/verifier.py",),
    ),
    Decision(
        "D-14", "generator-families-v1", "Which Generator families are in v1?", H,
        "The owner named energy-based SSL, Joint Energy Models, generative training (fill missing "
        "parts), autoregressive and diffusion methods [A-17]. Which first?",
        options=("jem", "masked-generative", "autoregressive", "diffusion"),
        sources=("A-17",),
        affects=("models/generator",),
    ),
    Decision(
        "D-15", "environment-persistence", "What is the durable Environment?", H,
        "Is the TSTCT KV cache the Environment itself, or a working view rebuilt from a durable "
        "event log (see P-02, P-18)?",
        options=("KV cache only", "KV cache as a view over an append-only event log"),
        sources=("A-12", "Q-11", "Q-19", "ARCH §5.2"),
        affects=("memory/kvcache.py", "memory/eventlog.py", "roles/simulator.py"),
    ),
    Decision(
        "D-16", "zero-shot-definition", "What does 'novel' mean in zero-shot validation?", H,
        "Unseen attack families only, or also unseen networks (leave-one-network-out, ARCH §12)?",
        options=("unseen families", "unseen families + unseen networks"),
        sources=("I-01", "A-24 stage 5", "ARCH §12"),
        affects=("pipeline/splits.py",),
    ),
    Decision(
        "D-17", "threat-model-scope", "What is inside the threat model?", D,
        "Is analyst/human feedback part of the threat model?",
        value="No. Human feedback is trusted supplied truth. All telemetry the model receives is "
        "inside the threat model.",
        sources=("A-18", "A-21"),
        affects=("roles/verifier.py", "ingest"),
    ),
    Decision(
        "D-18", "physics-framing", "What is the physics term for?", D,
        "Is physics-informed learning a detector of 'impossible' attacker actions, or a boundary on "
        "the model?",
        value="A boundary: it keeps the model's own outputs (reconstructions, imagined states, "
        "generated variants, predicted effects) inside what is physically possible, so it cannot "
        "hallucinate. It is not an attack detector.",
        sources=("A-08", "Q-05", "Q-41"),
        affects=("physics", "objectives/template.py"),
    ),
    Decision(
        "D-19", "names", "Role and component names", D,
        "Current names of roles and components.",
        value="planner → Advisor, renderer → Forecaster, spatio-temporal-causal transformer → TSTCT "
        "(Topological Spatio-Temporal Causal Transformer), EBT → TAAFT (Topological Anti-Adversary "
        "Foundation Transformer); encoder = CVG-AE (Complex Variational Graph AutoEncoder).",
        sources=("A-10", "A-11"),
        affects=("core/roles.py",),
    ),
    Decision(
        "D-20", "encoder-variational", "Is the encoder variational?", D,
        "Deterministic or variational autoencoder?",
        value="Variational (CVG-AE), to better reach the earlier design objectives and OOD.",
        sources=("A-11",),
        affects=("models/cvgae", "models/variational.py"),
    ),
    Decision(
        "D-21", "verifier-human-gated", "When may the Verifier change the model?", D,
        "May the Verifier feed back live or automatically?",
        value="Never automatically. It observes online, stores human feedback, computes memory drift "
        "(outcome-forecast pairs), and adjusts weights only on a human's supervised command.",
        sources=("A-16", "A-21", "Q-14"),
        affects=("roles/verifier.py",),
    ),
    Decision(
        "D-22", "training-pipeline", "Training pipeline", D,
        "Stages of training.",
        value="Six stages: (1) deep data analysis; (2) preparation with the Generator; (3) SSL "
        "pretraining of CVG-AE, Decoder, TSTCT; (4) SSL pretraining of TAAFT with those frozen; "
        "(5) full training incl. Forecaster/Advisor policy-value agents and Verifier on human "
        "feedback; (6) zero-shot validation with calibration.",
        sources=("I-01", "A-24"),
        affects=("pipeline/stages.py",),
        note="The image [I-01] splits A-24's stage 1 into analysis and preparation; this follows the "
        "image.",
    ),
    Decision(
        "D-23", "data-splits", "Data splits", D,
        "Which data go into which split?",
        value="Training and validation use real + generated data. Zero-shot uses real data only. "
        "Novel and known attacks are evaluated separately.",
        sources=("I-01",),
        affects=("pipeline/splits.py",),
    ),
    Decision(
        "D-24", "mechanism-design-placement", "Where does mechanism design live?", H,
        "The owner lists mechanism design inside TAAFT [A-19]. Should the defender's side of it "
        "(shaping the incentives an attacker faces, e.g. deception) sit in the Advisor?",
        options=("TAAFT", "Advisor", "both (analysis in TAAFT, design in Advisor)"),
        sources=("A-19",),
        affects=("models/taaft/lenses.py", "roles/advisor.py"),
    ),
    Decision(
        "D-25", "physics-on-telemetry", "May physics residuals score incoming telemetry?", H,
        "A record that violates physics signals a faulty sensor or forged telemetry (telemetry is "
        "in the threat model, D-17). Should TAAFT's trust/distrust management use residuals on "
        "observations as a reliability input, or should physics stay strictly a learning boundary "
        "(D-18)?",
        options=("boundary only", "boundary + telemetry reliability input"),
        sources=("A-08", "A-18", "A-19"),
        affects=("physics/term.py", "models/taaft/lenses.py"),
    ),
    Decision(
        "D-26", "noise-analysis", "Noise analysis in TAAFT", H,
        "Separate coloured and white noise? The owner is 'not sure yet'.",
        sources=("A-19",),
        affects=("models/taaft/lenses.py",),
        note="Evidence that it is meaningful: benign aggregate traffic is self-similar (Leland et "
        "al., IEEE/ACM ToN 1994), and malware beaconing shows periodic structure (BAYWATCH, "
        "DSN 2016).",
    ),
    Decision(
        "D-27", "kafka-client", "Kafka client library", H,
        "Which Python Kafka client (confluent-kafka, aiokafka, kafka-python)?",
        sources=("datamodel.md",),
        affects=("ingest/kafka.py",),
    ),
    Decision(
        "D-28", "demo-interface", "Demo interface", H,
        "Streamlit, Flask web app, or CLI for the offline demonstration (problem statement)?",
        options=("Streamlit", "Flask", "CLI"),
        sources=("CLAUDE.md problem statement",),
        affects=("cli.py",),
    ),
    Decision(
        "D-29", "observation-status-weights", "Evidence weights for stale / low-reliability fields", H,
        "Weights w(m) in (0,1) for stale and low-reliability observations; age decay form.",
        sources=("P-03",),
        affects=("datamodel/status.py",),
    ),
    # ------------------------------------------- Foundational decisions (recorded for citation)
    Decision(
        "D-30", "event-driven-state", "State semantics", D, "What is one step of state?",
        value="Each input record is a state update (event-driven). K and N are chosen at runtime.",
        sources=("Q-25", "DESIGN_LOG snapshot"),
    ),
    Decision(
        "D-31", "vocabulary", "Vocabulary", D, "Words for states.",
        value="state model (schema), state (instance), state update, transition, entity state. "
        "Never 'token'.",
        sources=("Q-25",),
    ),
    Decision(
        "D-32", "passive-only", "Collection posture", D, "May NagaHana act on or install into the network?",
        value="Passive only: taps, NICs, integrations. No host agents. No actuation.",
        sources=("Q-03", "Q-13", "datamodel.md"),
    ),
    Decision(
        "D-33", "advisory-only", "Advisor output", D, "What does the Advisor output?",
        value="D3FEND-based counter-measure sequences, at graph or sensor level. Advisory only; "
        "humans decide.",
        sources=("Q-13", "Q-30", "A-15"),
    ),
    Decision(
        "D-34", "ot-in-scope", "OT scope", D, "Is OT in the initial scope?",
        value="Yes: OT is in the initial physics and data scope; CII is the priority.",
        sources=("Q-29", "ARCH 'Very critical aspect'"),
    ),
    Decision(
        "D-35", "memory-regions", "Memory regions", D, "How is memory divided?",
        value="Environment (facts, Simulator), Imagination (belief + forecasts, Forecaster), Monitor "
        "(deviations, Verifier). Advisor memory-less. Belief only in Imagination.",
        sources=("Q-31", "Q-32", "A-12", "A-14"),
    ),
    Decision(
        "D-36", "retention-by-trigger", "Retention", D, "What controls memory retention?",
        value="A regular, consistent schedule per Forecaster trigger; never input volume.",
        sources=("Q-33",),
    ),
    Decision(
        "D-37", "physics-global", "Physics placement", D, "Where does physics apply?",
        value="One shared Φ_phys term, reused across component losses. The Decoder has none of its own.",
        sources=("Q-41", "DESIGN_LOG 2026-09-29"),
    ),
    Decision(
        "D-38", "state-update-features", "State-update features", D, "Which features at minimum?",
        value="Flow-level and packet-level features at least, plus other features from the "
        "observable region.",
        sources=("Q-42",),
    ),
    Decision(
        "D-39", "hypergraph-encoder", "Encoder graph family", D, "Which graph family does the encoder use?",
        value="Heterogeneous multiplex hypergraph GNN (relation planes; typed entities; hyperedges).",
        sources=("Q-43", "A-03"),
    ),
    Decision(
        "D-40", "generator-training-only", "Generator scope", D, "When does the Generator run?",
        value="Training only: data augmentation by event variants; physics-bounded.",
        sources=("A-17", "Q-37"),
    ),
    Decision(
        "D-41", "absence-not-zero", "Missing information", D, "How is absent information treated?",
        value="Absence means 'not supplied', never zero; partially observable modelling.",
        sources=("Q-26",),
    ),
    # --------------------------------------------------------------------------- Proposals (P)
    Decision(
        "P-01", "decoder-provenance-view", "Decoder view with provenance tags", P,
        "View-only typed multigraph; every item tagged observed / believed / forecast; decode only "
        "candidate edges.",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-02", "environment-event-sourcing", "Environment as event sourcing", P,
        "Append-only, hash-chained transition log + periodic snapshots; the model learns transitions.",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-03", "observation-status-taxonomy", "Five observation statuses", P,
        "observed / not supplied / not observable / stale / low reliability, on every field.",
        sources=("DESIGN_LOG §3",),
        note="The principle (absence ≠ zero) is decided (D-41); this taxonomy is the proposal.",
    ),
    Decision(
        "P-04", "datamodel-layering", "Data-model layers L0-L5", P,
        "L0 raw → L1 OCSF event → L2 CSTS entity-relational → L3 status/reliability → L4 OT/CII → "
        "L5 macrostates; one Kafka topic per layer.",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-05", "formal-threat-model", "Formal threat model", P,
        "POSG adversary model; threats to NagaHana mapped to NIST AI 100-2 / MITRE ATLAS (telemetry "
        "only, per D-17); invariants.",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-07", "composition-method", "Compose from refs.md with ablation gates", P,
        "Borrow ideas per function from refs.md; each must beat its own removal (see P-17).",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-08", "parallel-branches-term", "Term for 'MIMD'", P,
        "'Relation-specific parallel branches' (ResNeXt cardinality; R-GCN; HGT).",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-09", "energy-two-readings", "One energy network, two readings", P,
        "Conditional E(z≤t, y) for forecasting; marginal E(∅, s) for Advisor shaping and novelty; "
        "learned by random history dropout (as in classifier-free guidance).",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-10", "rlcd-brier", "Brier-score RLCD reward", P,
        "r_cal = −(p̂ − 1[event])²; responded-to cases excluded (precedent: RLCR, arXiv:2507.16806).",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-11", "generator-physics-gate", "Hard physics gate for Generator variants", P,
        "Accept a variant only if Φ_phys ≤ τ. The principle (Generator is physics-informed) is "
        "decided by A-17; the hard gate is the proposal.",
        sources=("DESIGN_LOG §3", "A-17"),
    ),
    Decision(
        "P-12", "decoder-reads-monitor", "Decoder reads the Monitor region", P,
        "So deviations show in the live view.",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-13", "adversary-type-belief", "Belief over adversary types", P,
        "Example types: opportunistic / targeted / insider (part of P-05).",
        sources=("DESIGN_LOG §3",),
    ),
    Decision(
        "P-14", "ground-truth-world", "Ground-truth world simulator (JAX/Equinox)", P,
        "Simulated IT+OT networks with known hidden state and explicit observation models.",
        sources=("DESIGN_LOG §3 2026-09-29", "Q-44"),
    ),
    Decision(
        "P-15", "information-audit", "Information audit / Bayes ceiling", P,
        "Measure what observables reveal about hidden stage per regime; judge the model by its gap "
        "to the ceiling.",
        sources=("DESIGN_LOG §3 2026-09-29", "Q-44"),
    ),
    Decision(
        "P-16", "mechanistic-analysis", "Mechanistic checks", P,
        "Probes on latents; interventions (drop a plane, perturb a feature).",
        sources=("DESIGN_LOG §3 2026-09-29",),
    ),
    Decision(
        "P-17", "ablation-gates", "Pre-registered ablation gates", P,
        "Each borrowed idea must beat its own removal on temporal splits and cross-dataset transfer.",
        sources=("DESIGN_LOG §3 2026-09-29",),
    ),
    Decision(
        "P-18", "kvcache-as-view", "KV cache as a versioned, rebuildable view", P,
        "Treat KV caches as working memory keyed to the model hash that produced them; rebuild from "
        "the event log after retraining; bound growth by the retention schedule.",
        sources=("2026-09-29 review of A-12",),
    ),
    Decision(
        "P-19", "latent-interface", "One latent interface for all roles", P,
        "Roles exchange latents z in the CVG-AE space; KV caches stay internal; forecasts are "
        "projected back into z with a latent-consistency loss so the Decoder can render imagination.",
        sources=("2026-09-29 review of A-13, A-14",),
    ),
    Decision(
        "P-20", "masked-likelihood", "Reconstruct only observed fields", P,
        "The ELBO likelihood covers observed (contributing) fields only; see D-11a.",
        sources=("2026-09-29 review of A-11", "ARCH §4.2"),
    ),
    Decision(
        "P-21", "ood-beyond-likelihood", "OOD scoring beyond VAE likelihood", P,
        "VAE likelihoods can rank OOD data as more likely (Nalisnick et al., ICLR 2019); pair with "
        "energy or likelihood-ratio scores.",
        sources=("2026-09-29 review of A-11",),
    ),
    Decision(
        "P-22", "elastic-field-embeddings", "Field embeddings keyed by field ID", P,
        "Input layer embeds (field ID, value, status) so fields can be added later without "
        "retraining others: the owner's 'empty/disabled extra neurons' idea [Q-01].",
        sources=("Q-01",),
    ),
    Decision(
        "P-23", "generator-no-leakage", "Split before the Generator trains", P,
        "The Generator trains only on the real training split, so zero-shot families never leak "
        "into generated data (data snooping, Arp et al. 2022).",
        sources=("2026-09-29 review of I-01",),
    ),
)

REGISTRY: dict[str, Decision] = {}
_BY_SLUG: dict[str, Decision] = {}
for _d in _ENTRIES:
    if _d.id in REGISTRY or _d.slug in _BY_SLUG:
        raise RuntimeError(f"Duplicate decision id or slug: {_d.id} / {_d.slug}")
    REGISTRY[_d.id] = _d
    _BY_SLUG[_d.slug] = _d


def get(key: str) -> Decision:
    """Return a decision by ID (`"D-12"`) or slug (`"taaft-policy-coupling"`)."""
    if key in REGISTRY:
        return REGISTRY[key]
    if key in _BY_SLUG:
        return _BY_SLUG[key]
    raise KeyError(f"Unknown decision {key!r}. Known IDs: {', '.join(sorted(REGISTRY))}")


def is_decided(key: str) -> bool:
    """True when the decision is DECIDED."""
    return get(key).status is Status.DECIDED


def require(key: str) -> Decision:
    """Return a DECIDED decision, or raise `DecisionHeld` naming what must be decided.

    Use this at the exact point where code would otherwise have to pick an option. Proposals are
    not decisions: gate them with `require_proposal`.
    """
    d = get(key)
    if d.status is Status.DECIDED:
        return d
    if d.status is Status.PROPOSED:
        raise DecisionHeld(
            f"{d.id} ({d.slug}) is a PROPOSAL, not a decision: {d.question} "
            "Gate proposal code with require_proposal(...) instead."
        )
    raise DecisionHeld(
        f"{d.id} ({d.slug}) is {d.status.value}: {d.question} "
        f"Options noted: {', '.join(d.options) or 'none recorded'}. "
        "Record the owner's decision in governance/decisions.py and DESIGN_LOG.md; do not default."
    )


def require_proposal(key: str, enabled: Collection[str]) -> Decision:
    """Allow proposal code to run only when its ID (or slug) is in `enabled`.

    `enabled` comes from config (`enabled_proposals: [...]`). A DECIDED entry always passes,
    because once the owner approves a proposal its entry is flipped to DECIDED.
    """
    d = get(key)
    if d.status is Status.DECIDED:
        return d
    if d.id in enabled or d.slug in enabled:
        return d
    raise ProposalNotEnabled(
        f"{d.id} ({d.slug}) is a proposal awaiting approval: {d.question} "
        "Enable it explicitly in config (enabled_proposals) for experiments, or ask the owner."
    )


def by_status(status: Status) -> tuple[Decision, ...]:
    """All entries with the given status, in registry order."""
    return tuple(d for d in _ENTRIES if d.status is status)


def all_entries() -> tuple[Decision, ...]:
    """Every entry, in registry order."""
    return _ENTRIES


def ids(entries: Iterable[Decision]) -> tuple[str, ...]:
    """IDs of the given entries (convenience for messages)."""
    return tuple(d.id for d in entries)
