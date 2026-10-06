"""Typed configuration of a NagaHana training run: the single source of truth (D-57).

Stage numbering (canonical)
---------------------------
    prep      Data preparation: ingest, analytics, cleaning, splitting, augmentation plan
    stage 1   Simulator pretraining: FieldEncoder, CVG-AE, Decoder, TSTCT (self-supervised)
    stage 2   TAAFT pretraining with the stage-1 modules frozen (self-supervised)
    stage 3   full training of the complete architecture (Forecaster, Advisor, Verifier) with
              training-phase human feedback
    stage 4   zero-shot validation with human-feedback confidence calibration

D-22 lists the same work as six steps; docs/training.md keeps the mapping (analysis and preparation
are the data preparation, steps 3 to 6 are stages 1 to 4).

Convention
----------
Every field is declared with `core.config.setting(default, source=..., doc=...)`: the default is the
build's value, the metadata names the decision (D-xx) or assumption (AS-xx) behind it. The YAML files
in conf/training/ are rendered from these dataclasses (`render_preset`, with the provenance as
comments) and a test compares them with the code, so they cannot drift. A YAML file or `key=value`
overrides given at run time are validated against the dataclasses (`core.config.from_mapping`): unknown
keys, wrong types and `???` on a field with a default raise. Value ranges are checked by each class's
`__post_init__` (`InvariantViolation`).

A site value the build cannot know (the MTU of the physics term) has the default None; the stage that
needs it refuses to run without it (`ConfigMissing`), so it is never defaulted and never silently
skipped (D-37). Held decisions are not fields here: they resolve through `governance.decisions`
(working options in conf/decisions/decisions.yaml; a run may name an override file in
`decisions_file`). Proposals are enabled per run in `enabled_proposals`.

Presets: `run_config("L")` is the full-scale run of the L model (the dataclass defaults);
`run_config("tiny")` is the CPU test fixture (preset("tiny") is a test fixture, not a model).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nagahana.core.config import from_mapping, load_yaml, render_yaml, setting, to_mapping
from nagahana.core.errors import InvariantViolation
from nagahana.models.generator.config import GeneratorPolicy, LangevinConfig

SCHEDULES: tuple[str, ...] = ("wsd", "cosine", "linear", "constant")
DECAY_SHAPES: tuple[str, ...] = ("1-sqrt", "linear", "cosine")
PRECISIONS: tuple[str, ...] = ("bf16", "fp32", "fp16")
NS_DTYPES: tuple[str, ...] = ("bf16", "fp32")
MUON_ADJUSTMENTS: tuple[str, ...] = ("match_rms_adamw", "original")
STRATEGIES: tuple[str, ...] = ("auto", "single", "ddp", "fsdp")
BACKENDS: tuple[str, ...] = ("auto", "nccl", "gloo")
LOGGERS: tuple[str, ...] = ("jsonl", "null", "mlflow")
SOURCE_KINDS: tuple[str, ...] = ("pcap", "cic-flows", "ctu13", "ciciot2023")
TARGET_MIXTURES: tuple[str, ...] = ("balanced", "uniform-families", "explicit", "none")
ADAMW_IMPLEMENTATIONS: tuple[str, ...] = ("auto", "foreach", "fused", "loop")
EVAL_SPLITS: tuple[str, ...] = ("test", "zero_shot")
#: Components whose blocks can be sharding or recomputation units (names of `models.nagahana.COMPONENTS`).
BLOCK_COMPONENTS: tuple[str, ...] = ("tstct", "taaft", "forecaster", "advisor", "verifier")
#: The stages of the canonical numbering, in order.
STAGE_KEYS: tuple[str, ...] = ("prep", "stage1", "stage2", "stage3", "stage4")


def _check(ok: bool, where: str, message: str) -> None:
    # One error type for value-range problems, naming where they are (core.config uses the same).
    if not ok:
        raise InvariantViolation(f"{where}: {message}")


@dataclass(frozen=True)
class OptimConfig:
    """The hybrid optimiser of one training stage (D-61; AS-406, AS-570 ... AS-572, AS-581, AS-599).

    Muon trains the 2-D hidden weight matrices and AdamW everything else (training/muon.py documents the
    group rule). With Moonlight's update scale the two share one learning rate and weight decay.
    """

    lr: float = setting(3e-4, source="AS-406, D-61", doc="peak learning rate, shared by Muon and AdamW (match_rms_adamw scale)")
    weight_decay: float = setting(0.1, source="AS-406, D-61", doc="decoupled weight decay, shared by Muon and AdamW (AdamW: matrices only)")
    betas: tuple[float, float] = setting((0.9, 0.95), source="AS-571", doc="AdamW moment decay rates")
    eps: float = setting(1e-8, source="AS-571", doc="AdamW epsilon")
    adamw_implementation: str = setting("auto", source="detail", doc="AdamW kernel: auto | foreach | fused | loop")
    muon_momentum: float = setting(0.95, source="D-61", doc="Muon momentum mu")
    muon_nesterov: bool = setting(True, source="D-61", doc="Nesterov momentum in Muon")
    muon_ns_steps: int = setting(5, source="D-61", doc="Newton-Schulz iterations of Muon")
    muon_adjust: str = setting("match_rms_adamw", source="D-61", doc="Muon lr adjustment: match_rms_adamw (0.2 sqrt(max(A, B))) | original")
    muon_ns_dtype: str = setting("bf16", source="D-61, AS-39", doc="precision of the Newton-Schulz iteration: bf16 (as torch.optim.Muon) | fp32")
    qk_clip: bool = setting(True, source="D-61, AS-581", doc="QK-Clip after every optimiser step (MuonClip, arXiv:2507.20534)")
    qk_clip_tau: float = setting(100.0, source="AS-581", doc="largest admissible pre-softmax attention logit per head")
    grad_clip: float = setting(1.0, source="AS-406", doc="global gradient-norm clip, once per optimiser step after the reduction")
    schedule: str = setting("wsd", source="D-61, AS-570", doc="learning-rate schedule: wsd (default) | cosine | linear | constant")
    warmup_steps: int = setting(2000, source="AS-406, AS-570", doc="linear warm-up steps")
    decay_fraction: float = setting(0.2, source="AS-570", doc="WSD: share of the planned steps spent decaying (Haegele et al.: 20 %)")
    decay_shape: str = setting("1-sqrt", source="AS-570", doc="WSD decay shape: 1-sqrt (Haegele et al.) | linear | cosine")
    min_lr_ratio: float = setting(0.0, source="AS-570", doc="final learning rate as a share of the peak")
    max_nonfinite_skips: int = setting(0, source="AS-406, AS-572", doc="consecutive non-finite steps tolerated before raising (0 = refuse the first)")

    def __post_init__(self) -> None:
        w = "optim"
        _check(self.lr > 0, w, "lr must be > 0")
        _check(self.weight_decay >= 0, w, "weight_decay must be >= 0")
        _check(all(0.0 <= b < 1.0 for b in self.betas), w, "betas must lie in [0, 1)")
        _check(self.eps > 0, w, "eps must be > 0")
        _check(self.adamw_implementation in ADAMW_IMPLEMENTATIONS, w, f"adamw_implementation must be one of {ADAMW_IMPLEMENTATIONS}")
        _check(0.0 <= self.muon_momentum < 1.0, w, "muon_momentum must lie in [0, 1)")
        _check(1 <= self.muon_ns_steps < 100, w, "muon_ns_steps must lie in [1, 99]")
        _check(self.muon_adjust in MUON_ADJUSTMENTS, w, f"muon_adjust must be one of {MUON_ADJUSTMENTS}")
        _check(self.muon_ns_dtype in NS_DTYPES, w, f"muon_ns_dtype must be one of {NS_DTYPES}")
        _check(self.qk_clip_tau > 0, w, "qk_clip_tau must be > 0")
        _check(self.grad_clip > 0, w, "grad_clip must be > 0")
        _check(self.schedule in SCHEDULES, w, f"schedule must be one of {SCHEDULES}")
        _check(self.warmup_steps >= 0, w, "warmup_steps must be >= 0")
        _check(0.0 < self.decay_fraction <= 1.0, w, "decay_fraction must lie in (0, 1]")
        _check(self.decay_shape in DECAY_SHAPES, w, f"decay_shape must be one of {DECAY_SHAPES}")
        _check(0.0 <= self.min_lr_ratio <= 1.0, w, "min_lr_ratio must lie in [0, 1]")
        _check(self.max_nonfinite_skips >= 0, w, "max_nonfinite_skips must be >= 0")


@dataclass(frozen=True)
class EMAConfig:
    """Exponential moving average of the trained weights (AS-573: built for every stage, off by default)."""

    enabled: bool = setting(False, source="AS-573", doc="keep an EMA of the trained weights")
    decay: float = setting(0.9999, source="AS-573", doc="EMA decay")
    warmup: bool = setting(True, source="AS-573", doc="decay_t = min(decay, (1 + t) / (10 + t))")
    evaluate_with_ema: bool = setting(True, source="AS-573", doc="validate (and export) with the averaged weights")

    def __post_init__(self) -> None:
        _check(0.0 < self.decay < 1.0, "ema", "decay must lie in (0, 1)")


@dataclass(frozen=True)
class EarlyStopConfig:
    """Early stopping on a validation metric (AS-574)."""

    enabled: bool = setting(True, source="AS-574", doc="stop when the monitored metric stops improving")
    monitor: str = setting("val/total", source="AS-574", doc="validation metric watched (a 'val/...' key of the evaluation)")
    mode: str = setting("min", source="AS-574", doc="min or max")
    patience: int = setting(10, source="AS-574", doc="evaluations without improvement before stopping")
    min_delta: float = setting(0.0, source="AS-574", doc="smallest change counted as an improvement")
    restore_best: bool = setting(True, source="AS-574", doc="the stage ends with its best validated weights")

    def __post_init__(self) -> None:
        w = "early_stop"
        _check(self.mode in ("min", "max"), w, "mode must be 'min' or 'max'")
        _check(self.patience >= 1, w, "patience must be >= 1")
        _check(self.min_delta >= 0.0, w, "min_delta must be >= 0")
        _check(self.monitor.startswith("val/"), w, "monitor must name a validation metric ('val/...')")


@dataclass(frozen=True)
class CheckpointConfig:
    """Resumable checkpoints of one stage (AS-576)."""

    every_steps: int = setting(1000, source="AS-576", doc="optimiser steps between resumable checkpoints")
    keep_last: int = setting(3, source="AS-576", doc="newest checkpoints kept per stage (the best weights are kept apart)")

    def __post_init__(self) -> None:
        _check(self.every_steps >= 1, "checkpoint", "every_steps must be >= 1")
        _check(self.keep_last >= 1, "checkpoint", "keep_last must be >= 1")


@dataclass(frozen=True)
class LoopConfig:
    """The optimisation loop of one training stage or of site calibration (AS-572, AS-580, AS-592)."""

    enabled: bool = setting(True, source="D-22", doc="run this stage in train --all")
    max_steps: int = setting(0, source="docs/sizing.md", doc="optimiser steps (0 = bounded by epochs only)")
    epochs: int = setting(1, source="docs/sizing.md", doc="passes over the stage's training segments")
    accumulation: int = setting(1, source="AS-572", doc="micro-batches per optimiser step")
    lanes: int = setting(8, source="docs/sizing.md", doc="windows per micro-batch and rank (8 x 2,048 states)")
    precision: str = setting("bf16", source="AS-39, AS-587", doc="bf16 autocast | fp32 | fp16 autocast with loss scaling; weights fp32")
    eval_every: int = setting(1000, source="AS-574", doc="optimiser steps between validations (0 = at the end only)")
    eval_batches: int = setting(64, source="AS-574", doc="objective calls per validation and rank")
    log_every: int = setting(10, source="detail", doc="optimiser steps between logged metrics")
    generated: bool = setting(True, source="D-23, AS-592", doc="mix accepted Generator variants of training segments")
    perturb: bool = setting(False, source="AS-320", doc="out-of-order augmentation inside the recorded uncertainty")
    balanced_sampling: bool = setting(True, source="AS-325", doc="class-balanced segment order per epoch")
    optim: OptimConfig = setting(OptimConfig(), source="D-61", doc="hybrid Muon and AdamW optimiser")
    ema: EMAConfig = setting(EMAConfig(), source="AS-573", doc="weight averaging")
    early_stop: EarlyStopConfig = setting(EarlyStopConfig(), source="AS-574", doc="early stopping")
    checkpoint: CheckpointConfig = setting(CheckpointConfig(), source="AS-576", doc="resumable checkpoints")

    def __post_init__(self) -> None:
        w = "loop"
        _check(self.max_steps >= 0, w, "max_steps must be >= 0")
        _check(self.epochs >= 1, w, "epochs must be >= 1")
        _check(self.accumulation >= 1, w, "accumulation must be >= 1")
        _check(self.lanes >= 1, w, "lanes must be >= 1")
        _check(self.precision in PRECISIONS, w, f"precision must be one of {PRECISIONS}")
        _check(self.eval_every >= 0, w, "eval_every must be >= 0")
        _check(self.eval_batches >= 1, w, "eval_batches must be >= 1")
        _check(self.log_every >= 1, w, "log_every must be >= 1")


@dataclass(frozen=True)
class SourceSpec:
    """One raw source of the corpus (ingest/pcap.py in flow-state mode, D-51; ingest/csv_flows.py)."""

    kind: str = setting("pcap", source="detail", doc="pcap | cic-flows | ctu13 | ciciot2023")
    path: str = setting("", source="run setting", doc="file of the source")
    network: str = setting("", source="AS-35", doc="monitored network (leave-one-network-out)")
    dataset: str = setting("", source="AS-34", doc="label-mapping table of a CSV source (data.labels.MAPPERS)")
    labeller: str | None = setting(None, source="AS-34", doc="named labeller of a capture (training.runs.LABELLERS)")
    sandboxed: bool = setting(True, source="ADR-0005", doc="the parser runs in a sandboxed process (recorded in the manifest)")
    internal_networks: tuple[str, ...] | None = setting(None, source="AS-308", doc="CIDR blocks of the monitored network (None: adapter default)")
    max_records: int | None = setting(None, source="detail", doc="read at most this many records (None = all)")
    utc_offset_hours: float | None = setting(None, source="AS-301", doc="CSV clock offset from UTC (None: read as UTC)")

    def __post_init__(self) -> None:
        w = f"source {self.path!r}"
        _check(self.kind in SOURCE_KINDS, w, f"kind must be one of {SOURCE_KINDS}")
        _check(bool(self.path) and bool(self.network), w, "path and network must be set")
        if self.kind == "pcap":
            _check(self.labeller is not None, w, "a capture needs a named labeller (labels are dataset facts, AS-34)")
        else:
            _check(bool(self.dataset), w, "a CSV source needs its label-mapping dataset name (AS-34)")
        _check(self.max_records is None or self.max_records > 0, w, "max_records must be > 0 or null")


@dataclass(frozen=True)
class DataConfig:
    """Sources, segments, split policy and sampling (D-23; AS-35, AS-325 ... AS-333, AS-590)."""

    sources: tuple[SourceSpec, ...] = setting((), source="run setting", doc="raw sources of the corpus")
    segment_seconds: float | None = setting(None, source="AS-333", doc="segment length in seconds (None = K * c)")
    novel_families: tuple[str, ...] = setting((), source="D-16, AS-35", doc="attack families held out as novel (zero-shot)")
    held_out_networks: tuple[str, ...] = setting((), source="D-16, AS-35", doc="networks held out as zero-shot")
    purge: int = setting(1, source="AS-327", doc="segments dropped after each cut")
    balance_power: float = setting(1.0, source="AS-325", doc="exponent of the class-balanced sampler (1 = every family equally likely)")

    def __post_init__(self) -> None:
        w = "data"
        _check(self.segment_seconds is None or self.segment_seconds > 0, w, "segment_seconds must be > 0 or null")
        _check(self.purge >= 0, w, "purge must be >= 0")
        _check(0.0 <= self.balance_power <= 1.0, w, "balance_power must lie in [0, 1]")
        paths = [s.path for s in self.sources]
        _check(len(set(paths)) == len(paths), w, "a source path appears twice")
        unknown = set(self.held_out_networks) - {s.network for s in self.sources}
        _check(not (self.sources and unknown), w, f"held-out networks not among the sources: {sorted(unknown)}")


@dataclass(frozen=True)
class AugmentConfig:
    """Generator variants of the training split (D-23, D-40; AS-582, AS-583, AS-592, AS-594, AS-596)."""

    enabled: bool = setting(True, source="D-23", doc="augment the training split with Generator variants")
    generated_to_real: float = setting(1.0, source="docs/sizing.md", doc="variant budget per real training segment (4e9 + 4e9 updates: 1:1)")
    target: str = setting("balanced", source="AS-582", doc="balanced | uniform-families | explicit | none")
    mixture: tuple[tuple[str, float], ...] = setting((), source="AS-582", doc="(family, share) pairs when target is explicit")
    reserve_fraction: float = setting(0.25, source="AS-583", doc="extra variants per family kept for the stage-3 energy re-screen")
    max_rounds: int = setting(4, source="AS-582", doc="top-up rounds per family when variants are rejected")
    fit_steps: int = setting(2000, source="AS-594", doc="optimiser steps of each learned Generator family")
    fit_lr: float = setting(3e-4, source="AS-594", doc="learning rate of the learned Generator families")
    energy_max_windows: int = setting(4, source="AS-421", doc="windows scored per variant by the marginal-energy callable")
    store_variants: bool = setting(False, source="AS-596", doc="also store materialised variant tables (audit)")

    def __post_init__(self) -> None:
        w = "augment"
        _check(self.generated_to_real >= 0.0, w, "generated_to_real must be >= 0")
        _check(self.target in TARGET_MIXTURES, w, f"target must be one of {TARGET_MIXTURES}")
        if self.target == "explicit":
            shares = [s for _, s in self.mixture]
            _check(bool(shares) and all(s >= 0 for s in shares) and abs(sum(shares) - 1.0) < 1e-9, w,
                   "an explicit mixture needs non-negative shares summing to 1")
            fams = [f for f, _ in self.mixture]
            _check(len(set(fams)) == len(fams), w, "an explicit mixture names each family once")
        else:
            _check(not self.mixture, w, "mixture is read only when target is 'explicit'")
        _check(self.reserve_fraction >= 0.0, w, "reserve_fraction must be >= 0")
        _check(self.max_rounds >= 1, w, "max_rounds must be >= 1")
        _check(self.fit_steps >= 0 and self.fit_lr > 0, w, "fit_steps must be >= 0 and fit_lr > 0")
        _check(self.energy_max_windows >= 1, w, "energy_max_windows must be >= 1")


@dataclass(frozen=True)
class DistributedConfig:
    """Multi-process training (AS-576 ... AS-580)."""

    strategy: str = setting("auto", source="AS-577", doc="auto (single when WORLD_SIZE is 1, else fsdp) | single | ddp | fsdp")
    backend: str = setting("auto", source="AS-577", doc="auto (nccl on CUDA, gloo on CPU) | nccl | gloo")
    shard_units: tuple[str, ...] = setting(("tstct", "taaft"), source="AS-577", doc="components whose blocks are FSDP units")
    reshard_after_forward: bool = setting(True, source="AS-577", doc="free gathered block parameters after each forward")
    find_unused_parameters: bool = setting(True, source="AS-577", doc="DDP: stage graphs vary with R and the data")
    recompute: tuple[str, ...] = setting(("taaft",), source="AS-578, docs/sizing.md", doc="components whose blocks recompute activations")
    timeout_s: float = setting(1800.0, source="detail", doc="collective timeout in seconds")

    def __post_init__(self) -> None:
        w = "distributed"
        _check(self.strategy in STRATEGIES, w, f"strategy must be one of {STRATEGIES}")
        _check(self.backend in BACKENDS, w, f"backend must be one of {BACKENDS}")
        bad = [u for u in (*self.shard_units, *self.recompute) if u not in BLOCK_COMPONENTS]
        _check(not bad, w, f"unknown block components {bad}; known: {BLOCK_COMPONENTS}")
        _check(self.timeout_s > 0, w, "timeout_s must be > 0")


@dataclass(frozen=True)
class PrepConfig:
    """Data preparation: ingest, analytics, cleaning, splitting, augmentation plan (D-22; AS-589, AS-590)."""

    enabled: bool = setting(True, source="D-22", doc="run the data preparation in train --all")
    quantiles: tuple[float, ...] = setting((0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0), source="AS-589",
                                           doc="quantiles of each numeric column reported by the analytics")

    def __post_init__(self) -> None:
        _check(all(0.0 <= q <= 1.0 for q in self.quantiles), "prep", "quantiles must lie in [0, 1]")
        _check(list(self.quantiles) == sorted(self.quantiles), "prep", "quantiles must be sorted")


@dataclass(frozen=True)
class Stage1Options:
    """Stage-1 objective weights (AS-407 ... AS-410)."""

    mask_weight: float = setting(2.0, source="AS-407", doc="weight of masked cells in L_rec")
    beta_first: float = setting(0.1, source="AS-408", doc="beta_0 of the first-state KL")
    prior_weights: tuple[float, float] = setting((0.5, 0.5), source="AS-409", doc="(memory-stream, thinking-stream) transition-prior KL weights")
    edge_cap: int = setting(512, source="AS-410", doc="observed hyperedges per plane per step in L_edge")

    def __post_init__(self) -> None:
        _check(self.mask_weight >= 1.0, "stage1.options", "mask_weight must be >= 1")
        _check(self.beta_first >= 0.0 and all(w >= 0.0 for w in self.prior_weights), "stage1.options", "weights must be >= 0")
        _check(self.edge_cap >= 1, "stage1.options", "edge_cap must be >= 1")


@dataclass(frozen=True)
class Stage2Options:
    """Stage-2 weights, hidden-entity ratio and the frozen perception's R (AS-411, AS-412)."""

    masked_entity: float = setting(1.0, source="AS-411", doc="weight of the masked-entity belief loss")
    future_latent: float = setting(1.0, source="AS-411", doc="weight of the future-latent loss")
    contrastive: float = setting(1.0, source="AS-411", doc="weight of the contrastive marginal-energy loss")
    malignity: float = setting(1.0, source="AS-411", doc="weight of the malignity readout loss")
    energy_reg: float = setting(0.01, source="AS-411", doc="energy-magnitude regulariser")
    hidden_ratio: float = setting(0.15, source="AS-411", doc="share of seen entities hidden per window")
    perception_passes: int | None = setting(None, source="AS-412", doc="TSTCT R of the frozen perception (None = the run-time default)")

    def __post_init__(self) -> None:
        w = "stage2.options"
        _check(min(self.masked_entity, self.future_latent, self.contrastive, self.malignity, self.energy_reg) >= 0, w,
               "weights must be >= 0")
        _check(0.0 <= self.hidden_ratio < 1.0, w, "hidden_ratio must lie in [0, 1)")
        _check(self.perception_passes is None or self.perception_passes >= 1, w, "perception_passes must be >= 1")


@dataclass(frozen=True)
class Stage3Options:
    """Stage-3 weights, the STAGED switch and training-phase human feedback (AS-22, AS-412 ... AS-415, AS-595, AS-597)."""

    joint_after: int | None = setting(None, source="AS-22, AS-597", doc="optimiser steps of the stop-gradient phase (None = joint_after_fraction)")
    joint_after_fraction: float = setting(0.5, source="AS-597", doc="share of the planned steps in the stop-gradient phase")
    readout_weight: float = setting(1.0, source="AS-414", doc="weight of TAAFT's supervised readouts")
    advisor_weight: float = setting(1.0, source="AS-413", doc="weight of the Advisor's policy improvement")
    forecaster_hazard: float = setting(1.0, source="build-spec section 3", doc="Forecaster survival NLL weight")
    forecaster_bc_technique: float = setting(1.0, source="build-spec section 3", doc="Forecaster behaviour cloning (technique) weight")
    forecaster_bc_target: float = setting(1.0, source="build-spec section 3", doc="Forecaster behaviour cloning (target) weight")
    forecaster_stage: float = setting(1.0, source="build-spec section 3", doc="Forecaster per-step stage weight")
    forecaster_consistency: float = setting(1.0, source="build-spec section 3", doc="latent consistency weight")
    forecaster_hypothesis: float = setting(1.0, source="build-spec section 3", doc="hypothesis consistency weight")
    forecaster_latent: float = setting(1.0, source="build-spec section 3", doc="back-projected latent weight")
    forecaster_reward: float = setting(1.0, source="build-spec section 3", doc="process reward weight")
    forecaster_value: float = setting(1.0, source="build-spec section 3, AS-253", doc="TD(lambda) value weight")
    verifier_routes: int = setting(4, source="AS-415", doc="routes imagined per trigger for the Verifier's step labels")
    perception_passes: int | None = setting(None, source="AS-412", doc="TSTCT R of the frozen perception (None = the run-time default)")
    feedback: tuple[str, ...] = setting(("verifier-heads",), source="D-07, D-21, D-65, AS-595",
                                        doc="training-phase feedback learners (training.feedback registry); each runs only under a HumanCommand")
    feedback_config: str | None = setting(None, source="D-65", doc="YAML file of the Verifier's FeedbackLearningConfig read by its learners (None = its defaults)")
    rlhf_ledger: str | None = setting(None, source="D-65", doc="feedback ledger holding the analyst preferences of verifier-rlhf")
    rlhf_situations: str | None = setting(None, source="D-65", doc="situations file the preferences of verifier-rlhf refer to")
    rlhf_policy: str = setting("forecaster", source="D-65", doc="policy trained by verifier-rlhf: forecaster | advisor")
    rlhf_per_step: int = setting(16, source="D-65", doc="preferences drawn per micro-batch by verifier-rlhf")

    def __post_init__(self) -> None:
        w = "stage3.options"
        _check(self.joint_after is None or self.joint_after >= 0, w, "joint_after must be >= 0 or null")
        _check(0.0 <= self.joint_after_fraction <= 1.0, w, "joint_after_fraction must lie in [0, 1]")
        weights = [getattr(self, f.name) for f in dataclasses.fields(self) if f.name.startswith("forecaster_")]
        _check(min([self.readout_weight, self.advisor_weight, *weights]) >= 0, w, "weights must be >= 0")
        _check(self.verifier_routes >= 1, w, "verifier_routes must be >= 1")
        _check(self.perception_passes is None or self.perception_passes >= 1, w, "perception_passes must be >= 1")
        _check(len(set(self.feedback)) == len(self.feedback), w, "feedback learners must be distinct")
        _check(not {"verifier-heads", "verifier-rlcd"} <= set(self.feedback), w,
               "verifier-heads and verifier-rlcd train the same Verifier heads: name one of them")
        _check(self.rlhf_policy in ("forecaster", "advisor"), w, "rlhf_policy must be 'forecaster' or 'advisor'")
        _check(self.rlhf_per_step >= 1, w, "rlhf_per_step must be >= 1")
        if "verifier-rlhf" in self.feedback:
            _check(self.rlhf_ledger is not None and self.rlhf_situations is not None, w,
                   "verifier-rlhf needs rlhf_ledger and rlhf_situations")


@dataclass(frozen=True)
class StageLoop1:
    """Stage 1: Simulator pretraining (FieldEncoder, CVG-AE, Decoder, TSTCT; self-supervised)."""

    loop: LoopConfig = setting(LoopConfig(epochs=2, perturb=True), source="docs/sizing.md, AS-320",
                               doc="two passes over the corpus, out-of-order augmentation on")
    options: Stage1Options = setting(Stage1Options(), source="AS-407 ... AS-410", doc="objective weights")


@dataclass(frozen=True)
class StageLoop2:
    """Stage 2: TAAFT pretraining with the stage-1 modules frozen."""

    loop: LoopConfig = setting(LoopConfig(), source="docs/sizing.md", doc="one pass over the corpus")
    options: Stage2Options = setting(Stage2Options(), source="AS-411, AS-412", doc="objective weights")

    def __post_init__(self) -> None:
        _check(not self.loop.perturb, "stage2", "the out-of-order augmentation belongs to stage 1 (AS-320)")


@dataclass(frozen=True)
class StageLoop3:
    """Stage 3: full training of the complete architecture with training-phase human feedback."""

    loop: LoopConfig = setting(LoopConfig(), source="docs/sizing.md", doc="one pass over the corpus")
    options: Stage3Options = setting(Stage3Options(), source="AS-412 ... AS-415", doc="objective weights and feedback")

    def __post_init__(self) -> None:
        _check(not self.loop.perturb, "stage3", "the out-of-order augmentation belongs to stage 1 (AS-320)")


@dataclass(frozen=True)
class Stage4Config:
    """Stage 4: zero-shot validation with human-feedback confidence calibration (D-23; D-13 held; AS-26, AS-419, AS-420, AS-591)."""

    enabled: bool = setting(True, source="D-22", doc="run this stage in train --all")
    tstct_passes: int | None = setting(None, source="D-44", doc="TSTCT R (None = the run-time default)")
    taaft_passes: int | None = setting(None, source="D-44", doc="TAAFT R (None = the run-time default)")
    descent_steps: int | None = setting(None, source="D-44", doc="descent steps S (None = the run-time default)")
    horizon_k: int | None = setting(None, source="D-30", doc="forecast horizon K (None = the model default)")
    routes_n: int | None = setting(None, source="D-46", doc="routes N (None = the model default)")
    threshold: float = setting(0.5, source="AS-419", doc="alert threshold on P_inf(K) for detection metrics")
    eval_splits: tuple[str, ...] = setting(EVAL_SPLITS, source="D-23", doc="real splits evaluated (zero-shot split into known and novel)")
    lanes: int = setting(8, source="docs/sizing.md", doc="windows per batch and rank")
    calibration: tuple[str, ...] = setting(("ml-temperature",), source="D-13, AS-26, AS-591",
                                           doc="confidence calibrators (training.feedback registry); applied only on a HumanCommand")
    feedback_config: str | None = setting(None, source="D-65", doc="YAML file of the Verifier's FeedbackLearningConfig read by its calibrators (None = its defaults)")
    site_calibration: bool = setting(False, source="D-13, AS-26", doc="train LoRA site adapters (needs a HumanCommand at run time)")
    site_sources: tuple[SourceSpec, ...] = setting((), source="AS-26", doc="the site's own traffic for the adapters (unlabelled sources take labeller/dataset 'unlabelled')")
    site_alerts: str | None = setting(None, source="AS-26", doc="CSV of analyst-confirmed alerts: entity (data-model key), time (epoch s), malicious (0/1)")
    site_min_span_s: float = setting(86_400.0, source="AS-26", doc="site traffic required (24 h)")
    site_min_alerts: int = setting(50, source="AS-26", doc="analyst-confirmed alerts required")
    site_allow_short: bool = setting(False, source="AS-26", doc="acknowledge a pilot run below the requirements")
    site_alert_weight: float = setting(1.0, source="AS-420", doc="weight of the alert BCE beside the self-supervised terms")
    site_loop: LoopConfig = setting(
        LoopConfig(max_steps=2000, generated=False, eval_every=0, optim=OptimConfig(lr=1e-4, warmup_steps=100)),
        source="AS-598", doc="optimisation of the site adapters")

    def __post_init__(self) -> None:
        w = "stage4"
        for name in ("tstct_passes", "taaft_passes", "descent_steps", "horizon_k", "routes_n"):
            v = getattr(self, name)
            _check(v is None or v >= 1, w, f"{name} must be >= 1 or null")
        _check(0.0 < self.threshold < 1.0, w, "threshold must lie in (0, 1)")
        _check(bool(self.eval_splits) and all(s in EVAL_SPLITS for s in self.eval_splits), w,
               f"eval_splits must be a non-empty subset of {EVAL_SPLITS}")
        _check(self.lanes >= 1, w, "lanes must be >= 1")
        _check(self.site_min_span_s > 0 and self.site_min_alerts >= 1, w, "site requirements must be positive")
        _check(self.site_alert_weight >= 0, w, "site_alert_weight must be >= 0")
        _check(not self.site_loop.generated, w, "site calibration reads the site's real traffic only")
        _check(len(set(self.calibration)) == len(self.calibration), w, "calibrators must be distinct")


@dataclass(frozen=True)
class AblationConfig:
    """Ablation switches of the L model (P-17; AS-579 in docs/assumptions/training-pipeline.md).

    Every switch defaults to the full design. A run records its switches in the run manifest;
    `training/ablation.py` turns a named variant into switches and a tiered plan (inference-time,
    single-stage retrain from shared upstream checkpoints, or full retrain). Names are validated against
    the model configuration when the run starts (`training.ablation.validate_switches`).
    """

    variant: str = setting("full", source="P-17", doc="name of the ablation variant (training.ablation.build_variants); full = no ablation")
    tier: str = setting("none", source="P-17, AS-579", doc="none (the design) | inference | single-stage | full")
    lenses_off: tuple[str, ...] = setting((), source="D-42, P-17", doc="TAAFT lens terms removed from E_total (names of TAAFTConfig.lenses)")
    tstct_causal_heads: bool = setting(True, source="AS-09, AS-10", doc="TSTCT causal heads contribute (False: their outputs are masked)")
    tstct_temporal_heads: bool = setting(True, source="AS-09", doc="TSTCT temporal heads contribute (False: their outputs are masked)")
    loop_passes: int | None = setting(None, source="D-43, D-44, AS-07", doc="fixed loop passes R for TSTCT and TAAFT (None = the design: sampled in training, run-time budget at inference)")
    environment: bool = setting(True, source="D-51", doc="carry the Environment across windows (False: every window starts empty)")
    longterm_memory: bool = setting(True, source="AS-11, AS-220", doc="TAAFT reads the long-term memory")
    physics: bool = setting(True, source="D-18, D-37", doc="the physics boundary (stage-1 term and TAAFT's physics lens)")
    generator: bool = setting(True, source="D-23, D-40", doc="Generator augmentation of the training split")
    stage2_pretraining: bool = setting(True, source="D-22", doc="run TAAFT pretraining (False: stage 3 starts TAAFT from its initialisation)")
    planes_off: tuple[str, ...] = setting((), source="D-39, AS-01", doc="CVG-AE relation planes removed (names of GraphConfig.planes)")

    def __post_init__(self) -> None:
        w = "ablation"
        _check(bool(self.variant), w, "variant must be named")
        _check(self.tier in ("none", "inference", "single-stage", "full"), w, "tier must be none, inference, single-stage or full")
        _check((self.tier == "none") == (self.variant == "full"), w, "the design (variant 'full') has tier 'none' and only it")
        _check(len(set(self.lenses_off)) == len(self.lenses_off), w, "lenses_off must be distinct")
        _check("physics" not in self.lenses_off, w, "the physics term is ablated with physics=false, not as a lens")
        _check(len(set(self.planes_off)) == len(self.planes_off), w, "planes_off must be distinct")
        _check(self.loop_passes is None or self.loop_passes >= 1, w, "loop_passes must be >= 1 or null")

    @property
    def is_full_design(self) -> bool:
        """True when no switch departs from the design."""
        return self == AblationConfig()


@dataclass(frozen=True)
class TrainingRun:
    """One training run of NagaHana (module docstring)."""

    name: str = setting("nagahana-L", source="run setting", doc="run name (directory and tracking)")
    preset: str = setting("L", source="D-22", doc="model preset: L (the model) | tiny (the test fixture)")
    seed: int = setting(0, source="AS-588", doc="base seed of every random stream of the run")
    run_dir: str = setting("runs/nagahana-L", source="run setting", doc="directory of manifests, checkpoints, logs and reports")
    mtu: float | None = setting(None, source="D-37, AS-402", doc="site MTU in bytes for the physics term (required by the stages; never defaulted)")
    link_bps: float | None = setting(None, source="AS-28", doc="site link rate for the Generator's link-rate bound (None = not checked)")
    enabled_proposals: tuple[str, ...] = setting(("P-11", "P-23"), source="AS-28, AS-367",
                                                 doc="proposals enabled for the run: the physics gate (P-11) and the Generator's no-leakage rule (P-23)")
    decisions_file: str | None = setting(None, source="D-57", doc="override file of held-decision options (None = the working options)")
    logger: str = setting("jsonl", source="D-09", doc="jsonl | null | mlflow")
    tracking_uri: str | None = setting(None, source="D-09", doc="MLflow tracking URI (required for the mlflow logger)")
    deterministic: bool = setting(False, source="AS-588", doc="deterministic kernels (slower on accelerators)")
    device: str | None = setting(None, source="detail", doc="cpu | cuda | None (cuda when available)")
    data: DataConfig = setting(DataConfig(), source="D-23", doc="sources, segments and splits")
    augment: AugmentConfig = setting(AugmentConfig(), source="D-23, D-40", doc="Generator variants of the training split")
    generator: GeneratorPolicy = setting(GeneratorPolicy(), source="D-14, AS-27", doc="the Generator's run-level choices")
    distributed: DistributedConfig = setting(DistributedConfig(), source="AS-577", doc="multi-process training")
    prep: PrepConfig = setting(PrepConfig(), source="D-22", doc="data preparation")
    stage1: StageLoop1 = setting(StageLoop1(), source="D-22", doc="Simulator pretraining")
    stage2: StageLoop2 = setting(StageLoop2(), source="D-22", doc="TAAFT pretraining")
    stage3: StageLoop3 = setting(StageLoop3(), source="D-22", doc="full training with training-phase human feedback")
    stage4: Stage4Config = setting(Stage4Config(), source="D-22", doc="zero-shot validation with confidence calibration")
    ablation: AblationConfig = setting(AblationConfig(), source="P-17", doc="ablation switches (full = the design)")

    def __post_init__(self) -> None:
        w = "run"
        _check(self.preset in ("L", "tiny"), w, "preset must be 'L' (the model) or 'tiny' (the test fixture)")
        _check(bool(self.name) and bool(self.run_dir), w, "name and run_dir must be set")
        _check(self.mtu is None or self.mtu >= 68, w, "mtu must be >= 68 bytes (RFC 791 minimum) or null")
        _check(self.link_bps is None or self.link_bps > 0, w, "link_bps must be > 0 or null")
        _check(self.logger in LOGGERS, w, f"logger must be one of {LOGGERS}")
        _check(self.logger != "mlflow" or self.tracking_uri is not None, w, "the mlflow logger needs tracking_uri")
        _check(self.device in (None, "cpu", "cuda"), w, "device must be cpu, cuda or null")
        from nagahana.governance import decisions

        for p in self.enabled_proposals:
            d = decisions.get(p)
            _check(d.status is decisions.Status.PROPOSED or d.status is decisions.Status.DECIDED, w,
                   f"{p} is {d.status.value}; only proposals are enabled per run")

    def stage_loop(self, stage: int) -> LoopConfig:
        """The LoopConfig of stage 1, 2 or 3."""
        if stage == 1:
            return self.stage1.loop
        if stage == 2:
            return self.stage2.loop
        if stage == 3:
            return self.stage3.loop
        raise KeyError(f"stage {stage} has no optimisation loop (stages 1, 2, 3)")


def run_config(preset: str) -> TrainingRun:
    """The run configuration of a preset: "L" (full scale; the dataclass defaults) or "tiny" (the CPU fixture)."""
    if preset == "L":
        return TrainingRun()
    if preset != "tiny":
        raise KeyError(f"unknown run preset {preset!r}; known: 'L', 'tiny'")
    # One small loop shared by the tiny stages; each stage then sets its own step budget.
    small = LoopConfig(lanes=2, precision="fp32", eval_every=2, eval_batches=2, log_every=1,
                       checkpoint=CheckpointConfig(every_steps=2, keep_last=2), early_stop=EarlyStopConfig(patience=3),
                       optim=OptimConfig(lr=1e-3, warmup_steps=1, muon_ns_dtype="fp32"))
    return TrainingRun(
        name="nagahana-tiny", preset="tiny", seed=0, run_dir="runs/nagahana-tiny", mtu=1500.0, device="cpu",
        data=DataConfig(segment_seconds=300.0),
        augment=AugmentConfig(fit_steps=4, fit_lr=1e-3, energy_max_windows=1, max_rounds=2),
        generator=GeneratorPolicy(energy_hidden=32, energy_blocks=2, langevin=LangevinConfig(levels=3, steps_per_level=4)),
        distributed=DistributedConfig(strategy="single", backend="gloo", recompute=()),
        stage1=StageLoop1(loop=dataclasses.replace(small, max_steps=4, perturb=True)),
        stage2=StageLoop2(loop=dataclasses.replace(small, max_steps=3)),
        stage3=StageLoop3(loop=dataclasses.replace(small, max_steps=3)),
        stage4=Stage4Config(routes_n=3, lanes=2,
                            site_loop=dataclasses.replace(small, max_steps=2, generated=False, eval_every=0,
                                                          optim=OptimConfig(lr=1e-3, warmup_steps=0, muon_ns_dtype="fp32"))),
    )


def config_hash(cfg: TrainingRun) -> str:
    """SHA-256 of the canonical JSON of the configuration (sorted keys): the run's configuration identity."""
    text = json.dumps(to_mapping(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_HEADER = (
    "NagaHana training run, preset {preset} (training/config.py, TrainingRun).",
    "Generated from the typed dataclasses by nagahana.training.config.render_preset (D-57). Do not edit by hand:",
    "change the dataclass default and regenerate (python -m nagahana train-config --preset {preset} --write);",
    "tests/test_training_config.py compares this file with the code. Stages: prep, stage 1 ... 4 (docs/training.md).",
)


def render_preset(preset: str) -> str:
    """The YAML text of a preset, with every field's provenance as a comment."""
    return render_yaml(run_config(preset), header=tuple(h.format(preset=preset) for h in _HEADER))


def write_preset(preset: str, directory: str | Path) -> Path:
    """Write conf/training/<preset>.yaml from the dataclasses; returns the path."""
    path = Path(directory) / f"{preset}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_preset(preset), encoding="utf-8", newline="\n")
    return path


def conf_files() -> dict[str, str]:
    """Relative path under conf/ -> generated text, for the training presets (for core.config.conf_files)."""
    return {f"training/{p}.yaml": render_preset(p) for p in ("L", "tiny")}


def _set_dotted(tree: dict[str, Any], key: str, value: Any) -> None:
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise InvariantViolation(f"override key {key!r} is empty")
    node: Any = tree
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node or not isinstance(node[p], dict):
            raise InvariantViolation(f"override {key!r}: no section {p!r} on the path")
        node = node[p]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise InvariantViolation(f"override {key!r}: no field {parts[-1]!r}")
    node[parts[-1]] = value


def load_run_config(path: str | Path | None = None, *, preset: str | None = None,
                    overrides: Sequence[str] = ()) -> TrainingRun:
    """A run configuration from a YAML file (or a preset), with `dotted.key=value` overrides; validated.

    Exactly one of `path` and `preset` is given. A YAML file is an override of the dataclass defaults:
    keys it leaves out take the defaults. Override values are parsed as YAML (numbers, booleans, null,
    flow lists and mappings).
    """
    import yaml

    if (path is None) == (preset is None):
        raise InvariantViolation("give a config file or a preset, not both or neither")
    base = to_mapping(run_config(preset)) if preset is not None else to_mapping(TrainingRun())
    data: Mapping[str, Any] = load_yaml(path) if path is not None else {}
    tree = _merge(base, data, where="run")
    for item in overrides:
        if "=" not in item:
            raise InvariantViolation(f"override {item!r} must look like key.path=value")
        key, raw = item.split("=", 1)
        _set_dotted(tree, key, yaml.safe_load(raw))
    return from_mapping(TrainingRun, tree)


def _merge(base: dict[str, Any], data: Mapping[str, Any], *, where: str) -> dict[str, Any]:
    """`base` updated by `data`, section by section (unknown keys are left for from_mapping to reject)."""
    out = json.loads(json.dumps(base))
    for k, v in data.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v, where=f"{where}.{k}")
        else:
            out[k] = v
    return out


__all__ = [
    "AblationConfig", "AugmentConfig", "BLOCK_COMPONENTS", "CheckpointConfig", "DataConfig", "DistributedConfig", "EMAConfig",
    "EarlyStopConfig", "LoopConfig", "OptimConfig", "PRECISIONS", "PrepConfig", "SCHEDULES", "STAGE_KEYS", "SourceSpec",
    "Stage1Options", "Stage2Options", "Stage3Options", "Stage4Config", "StageLoop1", "StageLoop2", "StageLoop3",
    "TrainingRun", "conf_files", "config_hash", "load_run_config", "render_preset", "run_config", "write_preset",
]
