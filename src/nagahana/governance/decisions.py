"""The decision registry: every design question with its status, as code.

What this is
------------
The single place in code that says, for every design question, whether it is:

- DECIDED: code relies on `value`.
- HELD: not decided. A held entry carries a working option (`working`) and the assumption that
  records it (`assumption`, an AS-xx ID). `require(...)` resolves a held entry to the option in force:
  an option passed by the caller, else an option configured for the run (`configure`), else the
  working option. Resolving to the working option records the use of its assumption, so strict
  (audit) mode of `governance/assumptions.py` blocks it exactly as if nothing were assumed. A held
  entry without a working option can only be resolved by a configured option.
- PROPOSED: an engineering proposal. Its code runs only when its ID is enabled for the run
  (`require_proposal` with `enabled_proposals`).
- SUPERSEDED: kept for history. `superseded_by` points to the replacement.

Why it exists
-------------
Nothing undecided is silently defaulted:
- a held decision resolves only to a recorded, auditable option (its assumption) or to an option a
  run configured explicitly; the options admissible for each entry are listed here;
- deciding a question is a one-line status change here;
- `python -m nagahana decisions` lists every entry and the code that depends on it
  (`governance/report.py`), and `settings_snapshot()` gives the options in force for run logs.

How to update
-------------
1. Change the entry: status DECIDED, `value` in plain words, the source added.
2. Regenerate the configuration files (`core/config.py`, `generate_conf`) and `docs/decisions.md`
   (`python -m nagahana decisions --write-docs`).

IDs
---
- `D-01` to `D-11` keep the numbering of the design log; D-03 and D-11 are split into a/b/c.
- `D-12` onward follow in the order they were recorded.
- `P-xx` are proposals.

Each entry also has a readable `slug`, so code can say `require("taaft-policy-coupling")` instead of a
number.
"""

from __future__ import annotations

import contextvars
import enum
from collections.abc import Collection, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType

from nagahana.core.errors import DecisionHeld, InvalidOption, ProposalNotEnabled
from nagahana.governance import assumptions


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
        For DECIDED entries: what was decided. For a held entry returned by `require`: the option in
        force.
    options:
        Known options (for HELD and PROPOSED entries). Listing an option does not favour it.
    sources:
        Where the question comes from: quote IDs (`Q-`, `A-`, `I-`), design-log sections, documents.
    affects:
        Modules that depend on it (informational; `report.py` also finds uses automatically).
    superseded_by:
        For SUPERSEDED entries.
    note:
        Anything a reader must know.
    working:
        For HELD entries: the working option of the build, recorded by `assumption`.
    assumption:
        The AS-xx ID of the assumption that records the working option.
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
    working: str | None = None
    assumption: str | None = None

    @property
    def admissible(self) -> tuple[str, ...]:
        """Options a run may configure: the listed options plus the working option."""
        if self.working is None or self.working in self.options:
            return self.options
        return (*self.options, self.working)


D, H, P, S = Status.DECIDED, Status.HELD, Status.PROPOSED, Status.SUPERSEDED

_ENTRIES: tuple[Decision, ...] = (
    Decision(
        "D-01", "advisor-reads-environment", "Does the Advisor read the Environment?", D,
        "Does the Advisor (formerly Planner) read the Environment, or only Imagination?",
        value="It reads both Environment and Imagination (both KV caches).",
        sources=("A-15", "A-20", "DESIGN_LOG section 2.1"),
        affects=("memory/access.py", "roles/advisor.py"),
    ),
    Decision(
        "D-02", "forecaster-trigger-policy", "What fires a Forecaster computation?", H,
        "Event-driven triggers could let attack volume drive retention. Fixed cadence, fixed cadence plus "
        "capped priority triggers, or another rule?",
        options=("fixed cadence", "fixed cadence + capped priority triggers"),
        working="fixed cadence + capped priority triggers", assumption="AS-12",
        sources=("DESIGN_LOG section 2.2", "Q-33"),
        affects=("memory/retention.py", "roles/forecaster.py", "inference/engine.py"),
    ),
    Decision(
        "D-03a", "attacker-success-criterion", "What counts as the attacker succeeding?", H,
        "Defines the infiltration state behind P_inf(k): a crown jewel reached, any stage progression, "
        "or a per-site definition?",
        options=("crown jewel reached", "any ATT&CK stage progression", "per-site definition",
                 "internal entity in a post-initial-access tactic"),
        working="internal entity in a post-initial-access tactic", assumption="AS-18",
        sources=("DESIGN_LOG section 2.3a",),
        affects=("objectives/rewards.py", "roles/forecaster.py", "roles/advisor.py", "models/vocab.py"),
    ),
    Decision(
        "D-03b", "disruption-pricing", "How is disruption priced per asset?", H,
        "Cost of a counter-measure per asset (OT availability first?).",
        options=("per-asset price table", "criticality by entity kind x disruption weight by action"),
        working="criticality by entity kind x disruption weight by action", assumption="AS-23",
        sources=("DESIGN_LOG section 2.3b",),
        affects=("objectives/rewards.py", "roles/advisor.py", "models/advisor"),
    ),
    Decision(
        "D-03c", "advisor-risk-aggregation", "Expected-case or worst-case optimisation?", H,
        "How the Advisor aggregates over adversary policies.",
        options=("expected case", "worst case", "risk measure, e.g. CVaR"),
        working="risk measure, e.g. CVaR", assumption="AS-24",
        sources=("DESIGN_LOG section 2.3c",),
        affects=("roles/advisor.py", "models/advisor"),
    ),
    Decision(
        "D-04", "relation-plane-formation", "How are relation planes formed?", H,
        "Declared from physics and protocols, learned, or declared + learned (learned ones flagged as "
        "inferred)?",
        options=("declared", "learned", "declared + learned"),
        working="declared", assumption="AS-01",
        sources=("DESIGN_LOG section 2.4", "A-03"),
        affects=("graph/planes.py", "graph/builder.py", "models/cvgae"),
    ),
    Decision(
        "D-05", "decoder-view-flattening", "How does the Decoder flatten planes into one view?", H,
        "Typed multigraph, supra-graph (lossless), reducibility-based merging (De Domenico 2015), "
        "or a combination?",
        options=("typed multigraph", "supra-graph", "reducibility merge", "combination"),
        working="typed multigraph", assumption="AS-29",
        sources=("DESIGN_LOG section 2.5",),
        affects=("models/decoder", "roles/decoder.py"),
    ),
    Decision(
        "D-06", "standards-reading", "Which standards does the data model follow?", H,
        "Is 'OSTF' the same as OCSF? Is CSTS the Rahman 2026 substrate (arXiv:2603.23459)?",
        sources=("DESIGN_LOG section 2.6", "Q-27"),
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
        sources=("DESIGN_LOG section 2.8",),
    ),
    Decision(
        "D-09", "tech-stack", "Which libraries make up the stack?", D,
        "Which libraries may the build use?",
        value="The PyTorch family (core, PyG incl. TGN, TorchRL, TorchMetrics), MLflow, DVC, HF Accelerate, "
        "DeepSpeed, JAX + Flax/Equinox, Kafka, NumPy/pandas; LightZero; Diffrax, Optimistix, "
        "jaxtyping; Pyro/NumPyro; BlackJAX; hypothesis; Hydra; ruff/mypy; MkDocs; "
        "Arrow/Polars/DuckDB; scipy/statsmodels; Triton/FlashAttention/CUDA Graphs; and, from the "
        "problem statement, Scapy/PyShark and Streamlit/Flask/CLI.",
        sources=("Q-45", "ARCH #18"),
    ),
    Decision(
        "D-10", "repo-location", "Repository name and location", H,
        "Where does the repository live, and under which name?",
        sources=("DESIGN_LOG section 2.10",),
        note="The repository started as `nagahana/` [Q-45]. Name and location can still change; no code "
        "depends on them.",
    ),
    Decision(
        "D-11a", "reconstruction-target", "What does CVG-AE reconstruct?", H,
        "Which fields and structures enter the reconstruction likelihood: observed (contributing) fields "
        "only, candidate hyperedges, the next state?",
        options=("contributing fields only", "contributing fields + candidate hyperedges",
                 "contributing fields + candidate hyperedges + next state"),
        working="contributing fields + candidate hyperedges", assumption="AS-04",
        sources=("DESIGN_LOG section 2.11",),
        affects=("models/cvgae", "models/decoder", "objectives/template.py"),
    ),
    Decision(
        "D-11b", "energy-jobs-v1", "Which energy-model jobs are in v1?", H,
        "TAAFT has the EBT at its core [A-19], and the Generator uses energy-based SSL [A-17]. "
        "Which other uses are in v1: Advisor shaping? a novelty signal?",
        options=("TAAFT refinement only", "TAAFT refinement + novelty signal",
                 "TAAFT refinement + novelty signal + Advisor shaping"),
        working="TAAFT refinement + novelty signal + Advisor shaping", assumption="AS-16",
        sources=("DESIGN_LOG section 2.11", "A-17", "A-19"),
        affects=("models/taaft", "models/generator", "objectives/rewards.py"),
        note="Partly settled by D-42: TAAFT's own energy is the sum of the lens terms plus the physics term, "
        "and refinement descends it. Open: whether Advisor shaping and a novelty signal are v1 jobs.",
    ),
    Decision(
        "D-11c", "adversary-objective", "Adversary objective: realistic or worst-case floor?", H,
        "What the modelled attacker optimises, and the make-up of the policy population Pi_A.",
        options=("realistic", "realistic + exploiters", "worst-case floor"),
        working="realistic", assumption="AS-17",
        sources=("DESIGN_LOG section 2.11",),
        affects=("models/heads", "objectives/rewards.py", "models/forecaster"),
    ),
    Decision(
        "D-12", "taaft-policy-coupling", "How do TAAFT and the policy/value heads couple?", H,
        "Joint autoregressive model, separate transformer + policy/value heads (separate penalties for "
        "analysis and forecasting), or staged (separate first, joint fine-tune later)? Also: a "
        "detector-driven forecaster?",
        options=("joint", "separate", "staged"),
        working="staged", assumption="AS-22",
        sources=("A-19 item 4",),
        affects=("models/heads", "models/taaft", "pipeline/stages.py"),
        note="All three readings are implemented by `models/heads/policy_value.head_input`; the run selects "
        "the option in force.",
    ),
    Decision(
        "D-13", "site-calibration-adapter", "What do site calibration adapters change?", H,
        "Output-only recalibration (temperature, isotonic, conformal thresholds) or small "
        "trainable adapters (low-rank), always applied on human command?",
        options=("temperature scaling", "isotonic", "conformal thresholds", "low-rank adapters",
                 "low-rank adapters + temperature scaling"),
        working="low-rank adapters + temperature scaling", assumption="AS-26",
        sources=("A-24 stage 5", "I-01 stage 6"),
        affects=("roles/verifier.py", "nn/lora.py", "training/"),
    ),
    Decision(
        "D-14", "generator-families-v1", "Which Generator families are in v1?", H,
        "[A-17] names energy-based SSL, Joint Energy Models, generative training (fill missing parts), "
        "autoregressive and diffusion methods. Which first?",
        options=("jem", "masked-generative", "autoregressive", "diffusion", "all families"),
        working="all families", assumption="AS-27",
        sources=("A-17",),
        affects=("models/generator",),
    ),
    Decision(
        "D-15", "environment-persistence", "What is the durable Environment?", H,
        "Is the TSTCT KV cache the Environment itself, or a working view rebuilt from a durable "
        "event log (see P-02, P-18)?",
        options=("KV cache only", "KV cache as a view over an append-only event log"),
        working="KV cache as a view over an append-only event log", assumption="AS-11",
        sources=("A-12", "Q-11", "Q-19", "ARCH section 5.2"),
        affects=("memory/kvcache.py", "memory/eventlog.py", "roles/simulator.py"),
    ),
    Decision(
        "D-16", "zero-shot-definition", "What does 'novel' mean in zero-shot validation?", H,
        "Unseen attack families only, or also unseen networks (leave-one-network-out, ARCH section 12)?",
        options=("unseen families", "unseen families + unseen networks"),
        working="unseen families + unseen networks", assumption="AS-35",
        sources=("I-01", "A-24 stage 5", "ARCH section 12"),
        affects=("pipeline/splits.py",),
    ),
    Decision(
        "D-17", "threat-model-scope", "What is inside the threat model?", D,
        "Is analyst or human feedback part of the threat model?",
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
        sources=("A-08", "Q-05", "Q-41", "ADR-0003"),
        affects=("physics", "objectives/template.py"),
    ),
    Decision(
        "D-19", "names", "Role and component names", D,
        "Current names of roles and components.",
        value="planner -> Advisor, renderer -> Forecaster, spatio-temporal-causal transformer -> TSTCT "
        "(Topological Spatio-Temporal Causal Transformer), EBT -> TAAFT (Topological Anti-Adversary "
        "Foundation Transformer); encoder = CVG-AE (Complex Variational Graph AutoEncoder).",
        sources=("A-10", "A-11", "ADR-0002"),
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
        sources=("A-16", "A-21", "Q-14", "ADR-0004"),
        affects=("roles/verifier.py",),
    ),
    Decision(
        "D-22", "training-pipeline", "Training pipeline", D,
        "Stages of training.",
        value="Six stages: (1) deep data analysis; (2) preparation with the Generator; (3) SSL "
        "pretraining of CVG-AE, Decoder, TSTCT; (4) SSL pretraining of TAAFT with those frozen; "
        "(5) full training incl. Forecaster/Advisor policy-value agents and Verifier on human "
        "feedback; (6) zero-shot validation with calibration.",
        sources=("I-01", "A-24", "ADR-0006"),
        affects=("pipeline/stages.py",),
        note="The pipeline image [I-01] splits A-24's stage 1 into analysis and preparation; this follows "
        "the image.",
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
        "Mechanism design is listed inside TAAFT [A-19]. Should the defender's side of it (shaping the "
        "incentives an attacker faces, e.g. deception) sit in the Advisor?",
        options=("TAAFT", "Advisor", "both (analysis in TAAFT, design in Advisor)"),
        working="TAAFT", assumption="AS-14",
        sources=("A-19",),
        affects=("models/taaft/lenses.py", "roles/advisor.py"),
        note="With TAAFT or both, the analysis side sits inside E_game unless the lens 'mechanism-design' "
        "is listed in TAAFTConfig.lenses, which makes it a term of its own (the total energy is the same "
        "either way: the participation part of the adversary's gap moves between the two terms).",
    ),
    Decision(
        "D-25", "physics-on-telemetry", "May physics residuals score incoming telemetry?", H,
        "A record that violates physics signals a faulty sensor or forged telemetry (telemetry is "
        "in the threat model, D-17). Should TAAFT's trust and distrust management use residuals on "
        "observations as a reliability input, or should physics stay strictly a learning boundary "
        "(D-18)?",
        options=("boundary only", "boundary + telemetry reliability input"),
        working="boundary only", assumption="AS-40",
        sources=("A-08", "A-18", "A-19"),
        affects=("physics/term.py", "models/taaft/lenses.py"),
    ),
    Decision(
        "D-26", "noise-analysis", "Noise analysis in TAAFT", H,
        "Separate coloured and white noise, as a lens term of its own or inside the information lens?",
        options=("inside the information lens", "own lens term"),
        working="inside the information lens", assumption="AS-36",
        sources=("A-19",),
        affects=("models/taaft/lenses.py", "models/taaft/noise.py"),
        note="Evidence that it is meaningful: benign aggregate traffic is self-similar (Leland et "
        "al., IEEE/ACM ToN 1994), and malware beaconing shows periodic structure (BAYWATCH, "
        "DSN 2016).",
    ),
    Decision(
        "D-27", "kafka-client", "Kafka client library", H,
        "Which Python Kafka client: confluent-kafka, aiokafka or kafka-python?",
        options=("confluent-kafka", "aiokafka", "kafka-python"),
        sources=("datamodel.md",),
        affects=("ingest/kafka.py",),
    ),
    Decision(
        "D-28", "demo-interface", "Demo interface", H,
        "Streamlit, Flask web app, or CLI for the offline demonstration (problem statement)?",
        options=("Streamlit", "Flask", "CLI"),
        sources=("problem statement",),
        affects=("cli.py",),
    ),
    Decision(
        "D-29", "observation-status-weights", "Evidence weights for stale / low-reliability fields", H,
        "Weights w(m) in (0, 1) for stale and low-reliability observations; age decay form.",
        working="OBSERVED 1.0, STALE 0.5, LOW_RELIABILITY 0.5 where a numeric weight is needed; no age decay",
        assumption="AS-30",
        sources=("P-03",),
        affects=("datamodel/status.py", "models/decoder"),
    ),
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
        value="One shared Phi_phys term, reused across component losses. The Decoder has none of its own.",
        sources=("Q-41", "ADR-0003"),
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
    Decision(
        "D-42", "taaft-energy-from-lenses", "Energy made of lens terms", D,
        "Is TAAFT's energy one lens beside the others, or are the lenses terms of one energy?",
        value="The lenses are terms of the energy. TAAFT's total energy is the sum of the lens "
        "energies plus the physics term: E_total = E_belief-and-trust + E_game + E_information + "
        "E_topology + E_time + E_cause + lambda*Phi_phys. Refinement descends E_total.",
        sources=("ADR-0007", "A-19"),
        affects=("models/taaft/lenses.py", "models/taaft/energy.py", "physics/term.py"),
        note="Not decided: the value of lambda (working value AS-15), and whether the lenses mechanism "
        "design (D-24) and noise (D-26) add terms of their own or sit inside E_game and E_information "
        "(configurable through TAAFTConfig.lenses and the D-24 / D-26 options). Partly settles D-11b. "
        "See ADR-0007.",
    ),
    Decision(
        "D-43", "weight-tied-looping", "Weight-tied looping", D,
        "Do TSTCT and TAAFT loop, and may TAAFT loop back into TSTCT?",
        value="TSTCT and TAAFT each loop on themselves: the same block stack is applied for R "
        "passes with shared weights (more passes = more thinking at no extra parameters). There is "
        "no loop from TAAFT back into TSTCT, because beliefs must never overwrite observed facts in "
        "the Environment.",
        sources=("ADR-0008", "Q-31"),
        affects=("models/tstct/model.py", "models/taaft/model.py"),
        note="Consistent with D-35 (belief only in Imagination). How R is drawn in training and which pass "
        "writes the Environment are recorded as AS-07 and AS-06. See ADR-0008.",
    ),
    Decision(
        "D-44", "loop-budget-runtime", "Loop budget", D,
        "Is the number of passes R fixed in the architecture or set at run time?",
        value="R is a budget set at run time (like K, N and the descent steps), recorded with each "
        "forecast.",
        sources=("ADR-0008", "Q-21"),
        affects=("roles/contracts.py", "roles/forecaster.py", "models/tstct/model.py",
                 "models/taaft/model.py"),
        note="The configuration keeps one R per looped model (TSTCT and TAAFT), recorded as AS-08. "
        "See ADR-0008.",
    ),
    Decision(
        "D-45", "verifier-policy-value-heads", "Verifier policy and value heads", D,
        "What does the Verifier compute, beyond storing feedback and measuring drift?",
        value="The Verifier has a value head that scores how far a forecast or an advice can be "
        "trusted (learned from human feedback) and a policy head that proposes the calibration "
        "correction. A correction is applied only on a supervised human command.",
        sources=("architecture.md section 3.10", "A-16", "A-21"),
        affects=("roles/verifier.py", "models/heads", "models/verifier"),
        note="Consistent with D-07 and D-21. Not resolved by this: the form of the correction "
        "(output-only recalibration or trainable adapters, D-13) and the training signal of the "
        "policy head (a Brier reward is proposal P-10).",
    ),
    Decision(
        "D-46", "forecaster-max-routes", "Meaning of N", D,
        "What does N mean in the Forecaster's imagination?",
        value="N is the maximum number of routes explored in the Forecaster's imagination, since "
        "futures diverge. The model may return fewer distinct routes than N.",
        sources=("ADR-0008", "A-14"),
        affects=("models/heads", "roles/forecaster.py", "roles/contracts.py"),
        note="Refines the meaning of N in D-30; N is still chosen at run time.",
    ),
    Decision(
        "D-47", "group-addresses", "Group addresses", D,
        "How does the data model represent a destination address that names a group of machines "
        "(multicast or broadcast)?",
        value="As an entity of its own kind, `multicast`, for multicast and broadcast destinations: "
        "IPv4 224.0.0.0/4, IPv6 ff00::/8, IPv4 255.255.255.255, and an IPv4 destination sent to the "
        "Ethernet broadcast address that has not been seen sending (a directed broadcast). A group is "
        "neither a machine nor a subnet.",
        sources=("ADR-0009",),
        affects=("datamodel/records.py", "ingest/pcap.py", "lab/sizing.py"),
        note="Why a node of its own: LLMNR/NBT-NS poisoning (ATT&CK T1557.001) works through these "
        "addresses (LLMNR queries go to 224.0.0.252 and ff02::1:3, RFC 4795 section 2; NBT-NS queries to "
        "the subnet broadcast, RFC 1002). The rule 'an IPv4 address ending in .255 is a broadcast' is "
        "dropped: it is wrong for subnets larger than /24. The kind `subnet` stays for real subnets. "
        "See ADR-0009.",
    ),
    Decision(
        "D-48", "entity-identity", "Entity identity", D,
        "When several addresses belong to one machine, is that one entity or several? Where do the "
        "roles and names of entities come from?",
        value="One entity per machine. A machine's IPv6 link-local addresses are merged into it as "
        "aliases, matched through the link-layer (Ethernet) address that ARP shows for its IPv4 "
        "address. Roles and names are inferred from the traffic itself, never from hard-coded "
        "addresses, and every inferred fact is time-stamped with the event time from which it was "
        "known.",
        sources=("ADR-0009",),
        affects=("ingest/pcap.py", "datamodel/columnar.py"),
        note="The PCAP adapter (1.1.0) infers the roles 'dns-server' and 'gateway' and names from DNS "
        "answers and TLS server names. Not decided: merging other addresses of one machine (several "
        "IPv4 addresses, global IPv6 addresses, addresses that change hands such as DHCP leases), "
        "and identity for sources without link-layer addresses (raw-IP captures, flow records). "
        "See ADR-0009.",
    ),
    Decision(
        "D-49", "positional-encodings", "Positional encodings", D,
        "How does the model encode where a state sits: in time, in the graph, in the record schema, "
        "in the future, and across loop passes?",
        value="Time: continuous-time rotary encoding (queries and keys rotated by omega*t, t in seconds, "
        "frequencies log-spaced from about 1 ms to about 1 week) plus a learned bias on log-bucketed "
        "delta-t, on TSTCT temporal and causal heads and on TAAFT's Imagination reads. No index-based "
        "encoding anywhere (an attacker can inflate an index by volume, never time). Space: "
        "random-walk structural encodings per plane, a Graphormer-style attention bias by hop "
        "distance and shared plane, entity-kind and role embeddings; no address-based embedding. "
        "Fields: a learned slot embedding per column; TAAFT adversary-hypothesis slots get learned "
        "slot embeddings. Future step k: the same time rotation with t = k x window length. Loops: "
        "no pass embedding; the input is re-injected every pass.",
        sources=("build-spec sections 2.1-2.8",),
        affects=("nn/positional.py", "models/cvgae", "models/tstct", "models/taaft", "models/forecaster"),
        note="Sources: Su et al. 2021 (arXiv:2104.09864, RoPE); Dwivedi et al. ICLR 2022 "
        "(arXiv:2110.07875, RWSE); Ying et al. NeurIPS 2021 (arXiv:2106.05234, Graphormer); Geiping "
        "et al. 2025 (arXiv:2502.05171, input injection). Angles are computed in float64 relative to "
        "a cache origin: epoch seconds in float32 lose milliseconds.",
    ),
    Decision(
        "D-50", "clock-time-features", "Absolute clock-time features", D,
        "Do states carry absolute clock time (time of day, day of week)?",
        value="Built as a config switch, off when training on lab datasets, and turned on only for "
        "site calibration, where the site's own diurnal rhythm is learned from its normal traffic.",
        sources=("build-spec section 2.1",),
        affects=("nn/positional.py", "models/inputs"),
        note="Why off in training: in CIC-IDS2018 every attack runs at a fixed clock time on a fixed "
        "day, so clock features invite a shortcut (Arp et al., USENIX Security 2022, 'Dos and Don'ts "
        "of Machine Learning in Computer Security').",
    ),
    Decision(
        "D-51", "training-across-time", "Training across time; PCAP flow-state updates", D,
        "Packet-level updates make a fixed-size training window only seconds long, so 60 s triggers rarely "
        "fall inside one. How does training reach across time?",
        value="(1) Train on consecutive windows in time order; each window's TSTCT reads the previous "
        "windows' Environment K/V as read-only memory (Transformer-XL style), exactly as inference reads the "
        "Environment store, so triggers fire wherever they fall. (2) The PCAP adapter emits a state update "
        "when a flow's state changes (start, TCP flag change, every active-timeout interval, end) instead of "
        "one per packet; packet-level features are still carried in each update.",
        sources=("AS-317", "build-spec section 3"),
        affects=("ingest/pcap.py", "data/windows.py", "training/", "models/tstct"),
        note="Dai et al., 'Transformer-XL', ACL 2019 (arXiv:1901.02860), segment-level recurrence with "
        "stop-gradient memory. Refines D-30 for PCAP sources: a record is a flow-state change, not a packet. "
        "Measured on the sample slice (AS-317): 1,024 packet updates spanned 2.6-23.8 s and 4 of 5 windows "
        "had no 60 s trigger.",
    ),
    Decision(
        "D-52", "contact-star", "Contact through fan and group hyperedges", D,
        "When one scanner sweeps many hosts (a fan hyperedge) or a sender addresses a multicast group, are "
        "the swept hosts in contact with each other?",
        value="Star semantics: contact means communication. A fan joins its initiator with each responder; a "
        "group joins the sender with the group entity. Co-members are not in contact with each other; they "
        "are 2 hops apart through the scanner or the group, so TSTCT's hop-2 spatial heads still link them "
        "and TAAFT infers co-victimhood itself.",
        sources=("build-spec section 2.2",),
        affects=("graph/window.py", "models/tstct/masks.py"),
        note="Under clique semantics a single /16 sweep would make every swept host a 1-hop neighbour of "
        "every other, flooding spatial attention with pairs that never exchanged a packet. RWSE on the "
        "local subgraph keeps its clique expansion (structure, not contact).",
    ),
    Decision(
        "D-53", "flow-end-reason", "Flow end reason as a field", D,
        "Flow-state updates (D-51) include idle-end updates that carry no new field value (26 % of the "
        "sample slice). Suppress them, or make the end of a flow evidence?",
        value="Add a categorical field `flow.end_reason` (fin, rst, idle, capture_end), with 'unanswered' "
        "when the responder never sent a packet. Idle-end updates then carry evidence: an unanswered SYN is "
        "a scan signal, and silence after C2 traffic matters for low-and-slow activity.",
        sources=("AS-337", "AS-338"),
        affects=("datamodel/fields.py", "ingest/pcap.py", "data/windows.py"),
        note="Measured on the sample slice: 40,618 of 153,935 flow-state updates were idle-end updates, "
        "mostly unanswered scan SYNs reported 120 s after their last packet. Built as two fields (AS-337, "
        "AS-338): `flow.end_reason` (fin / rst / idle / capture_end; NOT_SUPPLIED while the flow is open) and "
        "`flow.unanswered` (0/1), because 'unanswered' is independent of the end cause and CSV sources know it "
        "without the cause.",
    ),
    Decision(
        "D-54", "precision-policy", "Numerical precision", D,
        "Which numerical precision do weights, compute and outputs use?",
        value="fp32 overall (weights stored and served in fp32; no fp16 weights). Outputs are computed in "
        "fp64: P_inf and hazards, stage and other posteriors, energies (E_total, per-lens energies and "
        "shares), and calibration (temperature, ECE, conformal thresholds). Time angles stay fp64 (D-49). "
        "Training keeps fp32 master weights.",
        sources=("docs/assumptions/precision.md",),
        affects=("models/forecaster", "models/taaft", "models/verifier", "lab/compute.py", "memory/kvcache.py",
                 "memory/environment.py", "memory/imagination.py", "training/carry.py",
                 "conf/model/memory/memory.yaml"),
        note="The Environment and Imagination K/V caches are stored in fp32 as well (lab/compute.py reports "
        "the cache sizes). Training's bf16 matrix compute with fp32 master weights (AS-39) is unchanged.",
    ),
    Decision(
        "D-55", "baseline-libraries", "Libraries for the baseline reproductions", D,
        "May the published baselines be reproduced with the libraries their authors used?",
        value="Yes. scikit-learn, XGBoost and LightGBM form the optional [baselines] extra. The logistic-"
        "regression baselines stay on PyTorch tensors so that they see exactly the features NagaHana sees; "
        "the tree ensembles and the scikit-learn cross-check of the logistic regression use the extra. "
        "Official code of published baselines is fetched by scripts from the authors' repositories into "
        "third_party/, which is not part of the package.",
        sources=("D-09",),
        affects=("baselines/lr", "baselines/published", "evaluation", "pyproject.toml"),
    ),
    Decision(
        "D-56", "statphys-early-warning", "Statistical-physics readouts and early warning", D,
        "Should the energy view of the network state be read out as a thermodynamic trajectory, with "
        "early-warning indicators of an approaching transition?",
        value="Yes, as a full module: energy, entropy and free-energy trajectories of the network state; "
        "Shannon and von Neumann entropies of the multiplex graph; critical-slowing-down indicators "
        "(rising variance and lag-1 autocorrelation before a transition, Scheffer et al., Nature 2009); "
        "wired into the TAAFT readouts, the inference engine and the evaluation.",
        sources=("architecture.md section 6 (energy and entropy growth as an early warning)", "D-42"),
        affects=("statphys", "models/taaft/readouts.py", "models/taaft/model.py", "inference/engine.py", "evaluation"),
    ),
    Decision(
        "D-57", "config-single-source", "Single source of configuration", D,
        "Where does configuration live, and how do configuration files relate to code?",
        value="Typed dataclasses are the single source of configuration. The YAML files under conf/ are "
        "generated from them, carry identical values annotated with their provenance, and are validated "
        "as overrides against the dataclasses. A held decision takes its recorded assumption's option "
        "unless a run configures another admissible option.",
        sources=("design review", "architecture.md section 9 (held decisions are configurable settings)"),
        affects=("core/config.py", "models/config/components.py", "conf/", "governance/decisions.py"),
    ),
    Decision(
        "D-58", "loop-normalised-injection", "Normalised input injection in the weight-tied loop", D,
        "How does the weight-tied loop keep its hidden state stable when R grows beyond training?",
        value="The weight-tied loop re-injects the input with normalisation, so the hidden-state scale does "
        "not grow with the number of passes R; it is validated at 2-4x the training R.",
        sources=("design review", "D-43", "D-44"),
        affects=("nn/loop.py", "models/tstct/model.py", "models/taaft/model.py"),
    ),
    Decision(
        "D-59", "earned-component-names", "Component names describe what the components compute", D,
        "What must the game lens, the physics residuals and the causal heads compute to carry their names?",
        value="The game lens is a game-theoretic equilibrium energy; physics residuals are network-physics "
        "and protocol laws (offload-aware size, capacity, propagation, sequence space, queueing); the causal "
        "heads carry causal-structure constraints validated against simulator ground truth.",
        sources=("design review", "D-18", "D-37", "D-42"),
        affects=("models/taaft/lenses.py", "physics", "models/tstct"),
    ),
    Decision(
        "D-60", "faithful-explanations", "Faithful explanations", D,
        "What must an explanation of a forecast attribute, and how is it validated?",
        value="Explanations attribute the displayed P_inf itself, integrate lens shares over the descent "
        "path, include structure and timing attributions, and are tested for faithfulness by deletion and "
        "insertion.",
        sources=("design review", "D-42"),
        affects=("inference/explain.py", "explain", "models/taaft/energy.py"),
    ),
    Decision(
        "D-61", "optimizer-hybrid-muon", "Hybrid Muon and AdamW optimiser", D,
        "Which optimiser trains the model, on which parameters, under which schedule?",
        value="Training uses torch.optim.Muon (Nesterov momentum, 5 Newton-Schulz steps, match_rms_adamw update "
        "scale) on 2-D hidden weight matrices and AdamW on embeddings, normalisation, biases, scalars and output "
        "heads, with QK-Clip on attention query and key projections and a warmup-stable-decay schedule; "
        "distributed training orthogonalises full gradient matrices.",
        sources=("Liu et al., arXiv:2502.16982", "Kimi K2, arXiv:2507.20534", "Haegele et al., arXiv:2405.18392",
                 "Wen et al., arXiv:2509.02046 (expected gain about 1.1x at 1.2B parameters over well-tuned AdamW)"),
        affects=("training",),
    ),
    Decision(
        "D-62", "stage-numbering", "Numbering of the training stages", D,
        "How are the stages of the training pipeline numbered?",
        value="Data preparation comes first and is not numbered; then Stage 1 Simulator pretraining (CVG-AE, "
        "Decoder, TSTCT), Stage 2 TAAFT pretraining, Stage 3 full training with human feedback, and Stage 4 "
        "zero-shot validation and calibration.",
        sources=("design review", "D-22"),
        affects=("pipeline/stages.py", "training/", "docs"),
        note="Renumbers the stages of D-22 (whose stages 3 to 6 become Stages 1 to 4); the content of each "
        "stage is unchanged.",
    ),
    Decision(
        "D-63", "site-sized-memory", "Working memory sized by the site", D,
        "Does the model change with the size of the monitored network?",
        value="No: every site runs the same L model. Its working memory (the Environment, Imagination and "
        "long-term caches) is sized by the site, as a function of monitored hosts, retention and state rate, "
        "from a single desktop accelerator for a small network up to four 80 GB accelerators at "
        "critical-infrastructure scale.",
        sources=("design review", "D-54"),
        affects=("lab/sizing.py", "lab/compute.py", "memory/", "conf/"),
    ),
    Decision(
        "D-64", "parameter-count-follows-datamodel", "Parameter count follows the data model", D,
        "Is the published parameter count of L fixed?",
        value="The L parameter count changes with the columns of the data model (the input layer and the "
        "Decoder's field heads grow with them); every published copy of the count is updated together.",
        sources=("design review", "P-22"),
        affects=("models/config", "models/nagahana.py", "docs"),
    ),
    Decision(
        "D-65", "verifier-rlhf-rlvr", "Verifier learning from human and verifiable rewards", D,
        "Which reward signals may the Verifier learn from?",
        value="Human-gated reinforcement learning from human feedback (RLHF) and from verifiable rewards "
        "(RLVR) in the Verifier, beside the calibration reward (RLCD); every update runs only under a human "
        "command (D-21).",
        sources=("design review", "D-21", "D-45"),
        affects=("models/verifier", "roles/verifier.py", "training/"),
    ),
    Decision(
        "D-66", "attention-outputs", "Attention maps in the explanations", D,
        "Are attention maps part of a forecast's explanation?",
        value="Per-forecast spatial, temporal and causal attention maps are exposed in the explanations.",
        sources=("design review", "problem statement (attention weights or SHAP values)"),
        affects=("inference/explain.py", "models/tstct", "roles/contracts.py"),
    ),
    Decision(
        "D-67", "pyshark-adapter", "Optional PyShark reader", D,
        "Which packet readers may the PCAP adapter use?",
        value="An optional PyShark reader beside dpkt.",
        sources=("design review", "D-09", "problem statement (Scapy or PyShark)"),
        affects=("ingest/pcap.py", "pyproject.toml"),
    ),
    Decision(
        "D-68", "catboost-baseline", "CatBoost baseline", D,
        "Is CatBoost one of the baseline libraries?",
        value="CatBoost joins the [baselines] extra.",
        sources=("design review", "D-55"),
        affects=("baselines", "pyproject.toml"),
    ),
    Decision(
        "P-01", "decoder-provenance-view", "Decoder view with provenance tags", P,
        "View-only typed multigraph; every item tagged observed / believed / forecast; decode only "
        "candidate edges.",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-02", "environment-event-sourcing", "Environment as event sourcing", P,
        "Append-only, hash-chained transition log + periodic snapshots; the model learns transitions.",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-03", "observation-status-taxonomy", "Five observation statuses", P,
        "observed / not supplied / not observable / stale / low reliability, on every field.",
        sources=("DESIGN_LOG section 3",),
        note="The principle (absence is not zero) is decided (D-41); this taxonomy is the proposal.",
    ),
    Decision(
        "P-04", "datamodel-layering", "Data-model layers L0-L5", P,
        "L0 raw -> L1 OCSF event -> L2 CSTS entity-relational -> L3 status/reliability -> L4 OT/CII -> "
        "L5 macrostates; one Kafka topic per layer.",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-05", "formal-threat-model", "Formal threat model", P,
        "POSG adversary model; threats to NagaHana mapped to NIST AI 100-2 / MITRE ATLAS (telemetry "
        "only, per D-17); invariants.",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-07", "composition-method", "Compose from refs.md with ablation gates", P,
        "Borrow ideas per function from refs.md; each must beat its own removal (see P-17).",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-08", "parallel-branches-term", "Term for 'MIMD'", P,
        "'Relation-specific parallel branches' (ResNeXt cardinality; R-GCN; HGT).",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-09", "energy-two-readings", "One energy network, two readings", P,
        "Conditional E(z<=t, y) for forecasting; marginal E(empty, s) for Advisor shaping and novelty; "
        "learned by random history dropout (as in classifier-free guidance).",
        sources=("DESIGN_LOG section 3",),
        note="D-42 fixes what the energy is made of (lens terms + physics). This proposal is about how it is "
        "conditioned (with or without history); it is not resolved by D-42 and stays a proposal.",
    ),
    Decision(
        "P-10", "rlcd-brier", "Brier-score RLCD reward", P,
        "r_cal = -(p_hat - 1[event])^2; responded-to cases excluded (precedent: RLCR, arXiv:2507.16806).",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-11", "generator-physics-gate", "Hard physics gate for Generator variants", P,
        "Accept a variant only if Phi_phys <= tau. The principle (the Generator is physics-informed) is "
        "decided by A-17; the hard gate is the proposal.",
        sources=("DESIGN_LOG section 3", "A-17"),
    ),
    Decision(
        "P-12", "decoder-reads-monitor", "Decoder reads the Monitor region", P,
        "So deviations show in the live view.",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-13", "adversary-type-belief", "Belief over adversary types", P,
        "Example types: opportunistic / targeted / insider (part of P-05).",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-14", "ground-truth-world", "Ground-truth world simulator (JAX/Equinox)", D,
        "Simulated IT+OT networks with known hidden state and explicit observation models.",
        value="Approved and built as the worldsim package: JAX + Equinox with a NumPy fallback, IT and OT "
        "networks with known hidden state, explicit observation models, and output in the NagaHana data "
        "model, so the pipeline, the information audit (P-15) and the evaluation can be run against worlds "
        "whose ground truth is known.",
        sources=("DESIGN_LOG section 3", "Q-44"),
        affects=("worldsim", "lab/world_sim.py", "lab/info_audit.py", "evaluation"),
    ),
    Decision(
        "P-15", "information-audit", "Information audit / Bayes ceiling", P,
        "Measure what observables reveal about hidden stage per regime; judge the model by its gap "
        "to the ceiling.",
        sources=("DESIGN_LOG section 3", "Q-44"),
    ),
    Decision(
        "P-16", "mechanistic-analysis", "Mechanistic checks", P,
        "Probes on latents; interventions (drop a plane, perturb a feature).",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-17", "ablation-gates", "Pre-registered ablation gates", P,
        "Each borrowed idea must beat its own removal on temporal splits and cross-dataset transfer.",
        sources=("DESIGN_LOG section 3",),
    ),
    Decision(
        "P-18", "kvcache-as-view", "KV cache as a versioned, rebuildable view", P,
        "Treat KV caches as working memory keyed to the model hash that produced them; rebuild from "
        "the event log after retraining; bound growth by the retention schedule.",
        sources=("review of A-12",),
    ),
    Decision(
        "P-19", "latent-interface", "One latent interface for all roles", P,
        "Roles exchange latents z in the CVG-AE space; KV caches stay internal; forecasts are "
        "projected back into z with a latent-consistency loss so the Decoder can render imagination.",
        sources=("review of A-13, A-14",),
    ),
    Decision(
        "P-20", "masked-likelihood", "Reconstruct only observed fields", P,
        "The ELBO likelihood covers observed (contributing) fields only; see D-11a.",
        sources=("review of A-11", "ARCH section 4.2"),
    ),
    Decision(
        "P-21", "ood-beyond-likelihood", "OOD scoring beyond VAE likelihood", P,
        "VAE likelihoods can rank OOD data as more likely (Nalisnick et al., ICLR 2019); pair with "
        "energy or likelihood-ratio scores.",
        sources=("review of A-11",),
    ),
    Decision(
        "P-22", "elastic-field-embeddings", "Field embeddings keyed by field ID", P,
        "Input layer embeds (field ID, value, status) so fields can be added later without retraining "
        "others ([Q-01]: empty or disabled extra neurons).",
        sources=("Q-01",),
    ),
    Decision(
        "P-23", "generator-no-leakage", "Split before the Generator trains", P,
        "The Generator trains only on the real training split, so zero-shot families never leak "
        "into generated data (data snooping, Arp et al. 2022).",
        sources=("review of I-01",),
    ),
)

REGISTRY: dict[str, Decision] = {}
_BY_SLUG: dict[str, Decision] = {}
for _d in _ENTRIES:
    if _d.id in REGISTRY or _d.slug in _BY_SLUG:
        raise RuntimeError(f"Duplicate decision id or slug: {_d.id} / {_d.slug}")
    # Registry self-check at import: a working option belongs to a held entry, comes with the assumption
    # that records it, and that assumption exists; a decided entry carries no working option.
    if _d.working is not None:
        if _d.status is not Status.HELD:
            raise RuntimeError(f"{_d.id}: only a HELD entry may carry a working option")
        if _d.assumption is None:
            raise RuntimeError(f"{_d.id}: a working option needs the assumption that records it")
        assumptions.get(_d.assumption)                       # KeyError if the assumption is unknown
    elif _d.assumption is not None:
        raise RuntimeError(f"{_d.id}: an assumption without a working option")
    REGISTRY[_d.id] = _d
    _BY_SLUG[_d.slug] = _d

#: Options configured for the current run (ID -> option), set by `configure`; context-local.
_CONFIGURED: contextvars.ContextVar[Mapping[str, str]] = contextvars.ContextVar(
    "nagahana_decision_options", default=MappingProxyType({})
)
#: Every resolution of a held entry in this process: ID -> {(option, how it was chosen)}.
_RESOLVED: dict[str, set[tuple[str, str]]] = {}


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


def _check_option(d: Decision, option: str) -> None:
    """Raise `InvalidOption` unless `option` is admissible for the held entry `d`."""
    if option not in d.admissible:
        listed = ", ".join(repr(o) for o in d.admissible) or "none recorded"
        raise InvalidOption(
            f"{d.id} ({d.slug}): option {option!r} is not admissible. Admissible options: {listed}. "
            "Record a new option in governance/decisions.py before configuring it."
        )


@contextmanager
def configure(options: Mapping[str, str]) -> Iterator[Mapping[str, str]]:
    """Configure options of held decisions for the enclosed block (thread- and task-local).

    `options` maps a decision ID or slug to one of its admissible options. Only HELD entries can be
    configured: a decided entry has nothing to configure and a proposal is enabled through
    `require_proposal`, so both raise `InvalidOption`. Nested blocks extend the outer configuration.
    Yields the merged mapping (ID -> option).
    """
    merged = dict(_CONFIGURED.get())
    for key, option in options.items():
        d = get(key)
        if d.status is not Status.HELD:
            raise InvalidOption(f"{d.id} ({d.slug}) is {d.status.value}; only held decisions take a configured option")
        if not isinstance(option, str):
            raise InvalidOption(f"{d.id}: a configured option must be a string, got {type(option).__name__}")
        _check_option(d, option)
        merged[d.id] = option
    previous = _CONFIGURED.set(MappingProxyType(merged))
    try:
        yield MappingProxyType(merged)
    finally:
        _CONFIGURED.reset(previous)


def configured_options() -> dict[str, str]:
    """The run-level options in force (ID -> option), as set by enclosing `configure` blocks."""
    return dict(_CONFIGURED.get())


def option_in_force(key: str, configured: str | None = None) -> str | None:
    """The option `require` would resolve a HELD entry to, without recording anything.

    Precedence: `configured` (the caller's option), then the run-level configuration, then the
    working option. None when the entry has none of them. A non-HELD entry raises `InvalidOption`.
    """
    d = get(key)
    if d.status is not Status.HELD:
        raise InvalidOption(f"{d.id} ({d.slug}) is {d.status.value}; options are in force only for held decisions")
    if configured is not None:
        return configured
    run = _CONFIGURED.get().get(d.id)
    return run if run is not None else d.working


def require(key: str, configured: str | None = None, *, by: str = "") -> Decision:
    """Return the decision with the value code must act on.

    - DECIDED: the entry itself (`configured` must be None: there is nothing to configure).
    - HELD: a copy whose `value` is the option in force (`option_in_force`): the caller's `configured`
      option, else the run-level option of `configure`, else the working option. The option must be
      admissible (`InvalidOption` otherwise). Resolving to the working option records the use of its
      assumption through `governance.assumptions.assume` (strict mode raises `DecisionHeld` there);
      a configured option is the run's explicit choice and is recorded in `resolutions()`. A held
      entry with no option in force raises `DecisionHeld`.
    - PROPOSED: raises `DecisionHeld` (proposals are gated with `require_proposal`).
    - SUPERSEDED: raises `DecisionHeld` naming the replacement.

    `by` names the caller in the assumption usage records.
    """
    d = get(key)
    if d.status is Status.DECIDED:
        if configured is not None:
            raise InvalidOption(f"{d.id} ({d.slug}) is decided; it takes no configured option")
        return d
    if d.status is Status.PROPOSED:
        raise DecisionHeld(
            f"{d.id} ({d.slug}) is a PROPOSAL, not a decision: {d.question} "
            "Gate proposal code with require_proposal(...) instead."
        )
    if d.status is Status.SUPERSEDED:
        raise DecisionHeld(f"{d.id} ({d.slug}) is superseded by {d.superseded_by}; require that entry instead.")
    option = option_in_force(d.id, configured)
    if option is None:
        listed = ", ".join(repr(o) for o in d.admissible) or "none recorded"
        raise DecisionHeld(
            f"{d.id} ({d.slug}) is held and has no working option: {d.question} "
            f"Configure one of the admissible options ({listed}) with decisions.configure(...)."
        )
    _check_option(d, option)
    if option == d.working and configured is None and d.id not in _CONFIGURED.get():
        assert d.assumption is not None                      # guaranteed by the import-time self-check
        assumptions.assume(d.assumption, by=by or __name__)
        how = f"working option ({d.assumption})"
    else:
        how = "configured"
    _RESOLVED.setdefault(d.id, set()).add((option, how))
    return replace(d, value=option)


def require_proposal(key: str, enabled: Collection[str]) -> Decision:
    """Allow proposal code to run only when its ID (or slug) is in `enabled`.

    `enabled` comes from the run configuration (`enabled_proposals`). A DECIDED entry always passes,
    because an approved proposal is flipped to DECIDED.
    """
    d = get(key)
    if d.status is Status.DECIDED:
        return d
    if d.id in enabled or d.slug in enabled:
        return d
    raise ProposalNotEnabled(
        f"{d.id} ({d.slug}) is a proposal awaiting approval: {d.question} "
        "Enable it explicitly for the run (enabled_proposals)."
    )


def working_options() -> dict[str, str]:
    """ID -> working option of every held entry that has one, in registry order."""
    return {d.id: d.working for d in _ENTRIES if d.working is not None}


def settings_snapshot() -> dict[str, str]:
    """ID -> the option in force for every held entry ('unresolved' when none), for run logs."""
    return {d.id: (option_in_force(d.id) or "unresolved") for d in _ENTRIES if d.status is Status.HELD}


def resolutions() -> dict[str, frozenset[tuple[str, str]]]:
    """Which held decisions were resolved in this process, to which option, and how."""
    return {k: frozenset(v) for k, v in _RESOLVED.items()}


def by_status(status: Status) -> tuple[Decision, ...]:
    """All entries with the given status, in registry order."""
    return tuple(d for d in _ENTRIES if d.status is status)


def all_entries() -> tuple[Decision, ...]:
    """Every entry, in registry order."""
    return _ENTRIES


def ids(entries: Iterable[Decision]) -> tuple[str, ...]:
    """IDs of the given entries (convenience for messages)."""
    return tuple(d.id for d in entries)
