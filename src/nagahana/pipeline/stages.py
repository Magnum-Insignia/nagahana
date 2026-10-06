"""The training pipeline as data: data preparation and stages 1 to 4 (D-22; canonical numbering).

Canonical numbering
-------------------
    prep     Data preparation: ingest, analytics, cleaning, splitting, augmentation plan
    1        Simulator pretraining: FieldEncoder, CVG-AE, Decoder, TSTCT (self-supervised)
    2        TAAFT pretraining with the stage-1 modules frozen (self-supervised)
    3        Full training of the complete architecture (Forecaster, Advisor, Verifier) with
             training-phase human feedback
    4        Zero-shot validation with human-feedback confidence calibration

D-22 records the same pipeline in six steps ([I-01]): steps 1 and 2 (deep data analysis; preparation
with the Generator) are the data preparation, steps 3 to 6 are stages 1 to 4 (`d22_steps`).

Why data and not code: each stage's inputs, outputs, trained and frozen modules and splits are
decisions; one declarative table makes the plan reviewable (`python -m nagahana stages`) and lets the
orchestrator (training/orchestrator.py), DVC and the tests share it.

Status, never stale
-------------------
`stage_status` derives a stage's status from the code itself: the implementing module must exist and
define its entry point (checked statically with `ast`, so the report needs no torch import), and the
held decisions the stage depends on are listed with the option in force (`governance.decisions`:
the run's configured option or the recorded working option). Nothing here states a status by hand.

Graph invariants (`check_graph`, tested): every artefact a stage consumes is produced by an earlier
stage; every module a stage freezes was trained by an earlier stage; ids increase along the order.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path

#: Model components (names of `models.nagahana.COMPONENTS`).
MODEL_COMPONENTS: tuple[str, ...] = ("inputs", "cvgae", "decoder", "tstct", "longterm", "taaft", "forecaster", "advisor",
                                     "verifier")


@dataclass(frozen=True)
class StagePlan:
    """One stage of the pipeline.

    Attributes
    ----------
    id: 0 for the data preparation, 1 to 4 for the stages. key, name: stable key and title.
    trains, frozen: design-level component names (CVG-AE, Decoder, TSTCT, TAAFT, ...), as in D-22.
    train_modules, frozen_modules: the model components (`MODEL_COMPONENTS`) the trainer actually trains
        and keeps fixed (the FieldEncoder belongs to the perception front end; the long-term memory is
        trained with TAAFT, which reads it, AS-220).
    splits: data splits read. consumes, produces: artefacts (keys of the orchestrator's run state).
    self_supervised: the objective uses no attack labels. human_feedback: analyst feedback enters,
        always under a HumanCommand (D-21).
    module, entry: the implementing module and its entry point (the status is derived from them).
    decisions: held decisions whose option in force shapes the stage.
    d22_steps: the steps of D-22's six-step list this stage covers.
    """

    id: int
    key: str
    name: str
    trains: tuple[str, ...]
    frozen: tuple[str, ...]
    train_modules: tuple[str, ...]
    frozen_modules: tuple[str, ...]
    splits: tuple[str, ...]
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    self_supervised: bool
    human_feedback: bool
    module: str
    entry: str
    decisions: tuple[str, ...]
    d22_steps: tuple[int, ...]

    @property
    def label(self) -> str:
        """'Data preparation' or 'Stage N'."""
        return "Data preparation" if self.id == 0 else f"Stage {self.id}"

    @property
    def waiting_on(self) -> tuple[str, ...]:
        """Nothing blocks a stage: held decisions resolve to an option in force (see `stage_status`)."""
        return ()


STAGES: tuple[StagePlan, ...] = (
    StagePlan(0, "prep", "Data preparation: ingest, analytics, cleaning, splitting, augmentation plan",
              ("generator",), (), (), (), ("raw",), (),
              ("prepared sources", "analysis report", "split plan", "augmentation plan"), True, False,
              "nagahana.training.prep", "run_prep", ("D-14", "D-16"), (1, 2)),
    StagePlan(1, "pretrain-simulator", "Simulator pretraining: FieldEncoder, CVG-AE, Decoder, TSTCT (self-supervised)",
              ("cvgae", "decoder", "tstct"), (), ("inputs", "cvgae", "decoder", "tstct"),
              ("longterm", "taaft", "forecaster", "advisor", "verifier"), ("train", "val"),
              ("prepared sources", "split plan", "augmentation plan"), ("stage1 checkpoint",), True, False,
              "nagahana.training.stage1", "Stage1Trainer", ("D-04", "D-11a", "D-15"), (3,)),
    StagePlan(2, "pretrain-taaft", "TAAFT pretraining with the stage-1 modules frozen (self-supervised)",
              ("taaft",), ("cvgae", "decoder", "tstct"), ("taaft", "longterm"),
              ("inputs", "cvgae", "decoder", "tstct", "forecaster", "advisor", "verifier"), ("train", "val"),
              ("prepared sources", "split plan", "augmentation plan", "stage1 checkpoint"), ("stage2 checkpoint",),
              True, False, "nagahana.training.stage2", "Stage2Trainer", ("D-11b", "D-24", "D-26"), (4,)),
    StagePlan(3, "full-training", "Full training: Forecaster, Advisor, Verifier with training-phase human feedback",
              ("taaft", "forecaster_heads", "advisor_heads", "verifier"), ("cvgae", "decoder", "tstct"),
              ("taaft", "longterm", "forecaster", "advisor", "verifier"), ("inputs", "cvgae", "decoder", "tstct"),
              ("train", "val"), ("prepared sources", "split plan", "augmentation plan", "stage2 checkpoint"),
              ("stage3 checkpoint", "calibration pairs"), False, True, "nagahana.training.stage3", "Stage3Trainer",
              ("D-12", "D-03a", "D-03b", "D-03c", "D-11c"), (5,)),
    StagePlan(4, "zero-shot", "Zero-shot validation with human-feedback confidence calibration (real data only)",
              ("site_adapter",), ("cvgae", "decoder", "tstct", "taaft", "forecaster_heads", "advisor_heads"),
              (), ("inputs", "cvgae", "decoder", "tstct", "longterm", "taaft", "forecaster", "advisor", "verifier"),
              ("zero_shot_known", "zero_shot_novel"), ("prepared sources", "split plan", "stage3 checkpoint"),
              ("evaluation report", "model outputs", "calibration proposal", "site adapters"), False, True,
              "nagahana.training.stage4", "evaluate_split", ("D-13", "D-16"), (6,)),
)


def get(key: str | int) -> StagePlan:
    """A stage by key ("pretrain-taaft"), number (2), "prep" or 0."""
    for s in STAGES:
        if key in (s.id, s.key):
            return s
    raise KeyError(f"No stage {key!r}; stages: {[s.key for s in STAGES]}")


def check_graph(stages: tuple[StagePlan, ...] = STAGES) -> None:
    """Raise ValueError when the plan breaks a graph invariant (module docstring)."""
    produced: set[str] = {"raw"}
    trained: set[str] = set()
    last = -1
    for s in stages:
        if s.id <= last:
            raise ValueError(f"stage ids must increase: {s.key} after id {last}")
        last = s.id
        missing = [a for a in s.consumes if a not in produced]
        if missing:
            raise ValueError(f"{s.key} consumes {missing}, which no earlier stage produces")
        unknown = [m for m in (*s.train_modules, *s.frozen_modules) if m not in MODEL_COMPONENTS]
        if unknown:
            raise ValueError(f"{s.key}: unknown model components {unknown}")
        if set(s.train_modules) & set(s.frozen_modules):
            raise ValueError(f"{s.key}: a component is both trained and frozen")
        untrained = [m for m in s.frozen_modules if m in ("inputs", "cvgae", "decoder", "tstct") and s.id >= 2
                     and m not in trained]
        if untrained:
            raise ValueError(f"{s.key} freezes {untrained} before any stage trained them")
        produced.update(s.produces)
        trained.update(s.train_modules)


@dataclass(frozen=True)
class StageStatus:
    """What the code says about a stage."""

    stage: StagePlan
    implemented: bool
    detail: str
    options: tuple[tuple[str, str], ...]


def _module_file(module: str) -> Path | None:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin)


def _defines(path: Path, name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(isinstance(n, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef) and n.name == name for n in tree.body)


def stage_status(stage: StagePlan) -> StageStatus:
    """The stage's status derived from the code (module docstring); no heavy import."""
    from nagahana.governance import decisions

    try:
        path = _module_file(stage.module)
    except ModuleNotFoundError:
        path = None
    if path is None:
        implemented, detail = False, f"{stage.module} does not exist"
    elif not _defines(path, stage.entry):
        implemented, detail = False, f"{stage.module} does not define {stage.entry}"
    else:
        implemented, detail = True, f"implemented in {stage.module}.{stage.entry}"
    options: list[tuple[str, str]] = []
    for d_id in stage.decisions:
        d = decisions.get(d_id)
        if d.status is decisions.Status.HELD:
            opt = decisions.option_in_force(d.id)
            options.append((d.id, f"{opt} ({'configured' if d.id in decisions.configured_options() else d.assumption})"
                            if opt is not None else "no option in force"))
        else:
            options.append((d.id, d.status.value))
    return StageStatus(stage=stage, implemented=implemented, detail=detail, options=tuple(options))


def report() -> list[str]:
    """Human-readable lines for `python -m nagahana stages`."""
    lines: list[str] = []
    for s in STAGES:
        st = stage_status(s)
        lines.append(f"{s.label} [{s.key}]: {s.name}")
        lines.append(f"   status: {'implemented' if st.implemented else 'not implemented'} ({st.detail})")
        lines.append(f"   trains: {', '.join(s.train_modules) or '-'} | frozen: {', '.join(s.frozen_modules) or '-'} "
                     f"| splits: {', '.join(s.splits)}")
        lines.append(f"   self-supervised: {s.self_supervised} | human feedback: {s.human_feedback} "
                     f"| D-22 steps: {', '.join(str(x) for x in s.d22_steps)}")
        if st.options:
            lines.append("   held decisions in force: " + "; ".join(f"{d} = {o}" for d, o in st.options))
    return lines


__all__ = ["MODEL_COMPONENTS", "STAGES", "StagePlan", "StageStatus", "check_graph", "get", "report", "stage_status"]
