"""Tiered ablations of the L model: switches, variants, retrain plans and their lineage (P-17; AS-579).

Purpose
-------
Every ablatable component of the design has a typed switch (`config.AblationConfig`), validated against
the model configuration and recorded in the run manifest. A named variant (`VARIANTS`) fixes the
switches and a tier, and `plan_variant` turns it into exactly the work it needs:

    inference      the main run's trained weights, evaluated (stage 4) with the switch applied at run time
    single-stage   the main run's shared upstream checkpoint (the final weights of the stage before the
                   first affected stage), frozen; the affected stage and the stages after it retrained
                   with the switch; then evaluated
    full           stages 1 to 3 retrained from scratch with the switch (architecture-level ablations);
                   then evaluated

The data preparation (sources, splits, augmentation plan) is shared with the main run unless the variant
changes it (no Generator augmentation). `ablation_manifest` lists every variant with its tier, switches,
status and checkpoint lineage (each checkpoint's directory, step and SHA-256 digest, and the upstream
checkpoint it started from), for the evaluation package that scores the variants.

How each switch is realised (AS-579)
------------------------------------
- TAAFT lens off: the TAAFT is built with the lens left out of `TAAFTConfig.lenses` ("ablate a lens by
  removing its name", models/taaft/lenses.py). Inference tier: the trained weights load without the
  lens's parameters (exactly those keys are left over, which is checked); retrain tiers: TAAFT trains
  without the term.
- TSTCT causal or temporal heads off: the heads' outputs are masked before the output projection of
  every TSTCT block (head masking, Michel, Levy and Neubig, NeurIPS 2019, arXiv:1905.10650): the heads
  contribute nothing, the architecture and checkpoints stay identical, in evaluation and in training.
  The causal gates stay available to TAAFT's cause lens, which has its own switch.
- Loop passes: R fixed for TSTCT and TAAFT (1 = no weight-tied looping, D-43), in training and evaluation.
- Environment off: windows do not carry the Environment (D-51); every window starts empty.
- Long-term memory off: TAAFT reads no long-term memory, and the memory does not train.
- Physics off: no physics term in the stage-1 objective nor in TAAFT's E_total (D-18, D-37).
- Generator off: no variants in training (full retrain).
- Stage-2 pretraining off: stage 3 starts TAAFT from its initialisation (single-stage, from stage 1).
- CVG-AE planes off: inference tier: the plane's hyperedges, contacts and structural encodings are
  removed from every window (the plane is unseen); retrain tiers: the model is built without the plane.

Decisions: D-18, D-22, D-23, D-37, D-39, D-42, D-43, D-51. Proposals: P-17. Assumptions: AS-579.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.models.batch import WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.training.assumptions import use
from nagahana.training.config import AblationConfig
from nagahana.training.serialization import atomic_write_text

TIERS: tuple[str, ...] = ("inference", "single-stage", "full")


@dataclass(frozen=True)
class Variant:
    """One ablation variant: switches, tier and the first stage it retrains (None: inference only)."""

    name: str
    tier: str
    switches: AblationConfig
    first_stage: int | None
    description: str


def validate_switches(ab: AblationConfig, cfg: NagaHanaConfig) -> None:
    """Every named lens and plane exists; head ablations leave the type they remove present (AS-579)."""
    use("AS-579", by=__name__)
    unknown = [x for x in ab.lenses_off if x not in cfg.taaft.lenses]
    if unknown:
        raise InvariantViolation(f"ablation: unknown TAAFT lenses {unknown}; the model has {cfg.taaft.lenses}")
    bad = [p for p in ab.planes_off if p not in cfg.graph.planes]
    if bad:
        raise InvariantViolation(f"ablation: unknown planes {bad}; the model has {cfg.graph.planes}")
    if len(ab.planes_off) >= len(cfg.graph.planes):
        raise InvariantViolation("ablation: at least one plane must remain")
    if not ab.tstct_causal_heads and cfg.tstct.causal_heads == 0:
        raise InvariantViolation("ablation: the model has no causal heads to mask")
    if not ab.tstct_temporal_heads and cfg.tstct.temporal_heads == 0:
        raise InvariantViolation("ablation: the model has no temporal heads to mask")


def model_config(cfg: NagaHanaConfig, ab: AblationConfig, *, tier: str) -> NagaHanaConfig:
    """The model configuration of a variant: lenses always, planes only when retraining (module docstring)."""
    out = cfg
    if ab.lenses_off:
        out = dataclasses.replace(out, taaft=dataclasses.replace(out.taaft, lenses=tuple(x for x in out.taaft.lenses
                                                                                          if x not in ab.lenses_off)))
    if ab.planes_off and tier != "inference":
        out = dataclasses.replace(out, graph=dataclasses.replace(out.graph, planes=tuple(p for p in out.graph.planes
                                                                                          if p not in ab.planes_off)))
    return out


def load_without_lenses(model: nn.Module, state: dict[str, torch.Tensor], lenses_off: tuple[str, ...]) -> None:
    """Load trained weights into a model built without some lenses; only those lenses' keys may be left over."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = tuple(f".terms.{x}." for x in lenses_off)
    stray = [k for k in unexpected if not any(a in k for a in allowed)]
    if missing or stray:
        raise InvariantViolation(f"lens ablation: missing {list(missing)[:5]}, unexpected {stray[:5]}")


def head_mask_hooks(model: nn.Module, ab: AblationConfig) -> Callable[[], None]:
    """Mask TSTCT head types before every block's output projection; returns a function removing the hooks."""
    cfg = model.cfg  # type: ignore[attr-defined]
    t = cfg.tstct
    off: list[int] = []
    if not ab.tstct_temporal_heads:
        off += list(range(t.spatial_heads, t.spatial_heads + t.temporal_heads))
    if not ab.tstct_causal_heads:
        off += list(range(t.spatial_heads + t.temporal_heads, t.spatial_heads + t.temporal_heads + t.causal_heads))
    if not off:
        return lambda: None
    from nagahana.training.distributed import stack_blocks

    d_h = t.dim // t.heads
    keep = torch.ones(t.dim)
    for h in off:
        keep[h * d_h:(h + 1) * d_h] = 0.0
    handles = []
    for blk in stack_blocks(model.tstct):  # type: ignore[attr-defined]
        proj = blk.attn.o_proj

        def pre(_m: nn.Module, args: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
            x = args[0]
            return (x * keep.to(device=x.device, dtype=x.dtype),)

        handles.append(proj.register_forward_pre_hook(pre))

    def remove() -> None:
        for h in handles:
            h.remove()

    return remove


def drop_planes(window: WindowBatch, planes: tuple[str, ...], all_planes: tuple[str, ...]) -> WindowBatch:
    """The window with the planes removed (inference-tier plane ablation; module docstring)."""
    if not planes:
        return window
    from nagahana.training.carry import min_max_product

    idx = [all_planes.index(p) for p in planes]
    g = window.graph
    inc = dict(g.incidence)
    kinds = dict(g.hyperedge_kind)
    ents = dict(g.hyperedge_entities)
    for p in planes:
        inc[p] = inc[p][:, :0]
        kinds[p] = kinds[p][:0]
        ents[p] = ents[p][:0]
    rwse = g.rwse.clone()
    rwse[:, idx, :] = 0.0
    graph = dataclasses.replace(g, incidence=inc, hyperedge_kind=kinds, hyperedge_entities=ents, rwse=rwse)
    cp = window.contact_planes.clone()
    cp[..., idx] = math.inf
    c1 = cp.min(dim=-1).values
    eye = torch.eye(c1.shape[-1], dtype=torch.bool, device=c1.device)
    c1 = torch.where(eye.expand_as(c1), torch.zeros_like(c1), c1)
    c2 = torch.stack([min_max_product(c1[b], c1[b]) for b in range(c1.shape[0])]) if c1.shape[0] else c1
    up = window.update_planes.clone()
    up[..., idx] = False
    return dataclasses.replace(window, graph=graph, contact_planes=cp, contact1=c1, contact2=c2, update_planes=up)


@dataclass
class RuntimeSwitches:
    """The run-time side of a variant's switches (evaluation and training)."""

    switches: AblationConfig
    all_planes: tuple[str, ...]
    data_planes_off: tuple[str, ...]

    @property
    def loop_passes(self) -> int | None:
        return self.switches.loop_passes

    @property
    def longterm_memory(self) -> bool:
        return self.switches.longterm_memory

    @property
    def physics(self) -> bool:
        return self.switches.physics

    @property
    def environment(self) -> bool:
        return self.switches.environment

    def apply_window(self, window: WindowBatch) -> WindowBatch:
        return drop_planes(window, self.data_planes_off, self.all_planes)


def runtime_switches(ab: AblationConfig, cfg: NagaHanaConfig, *, tier: str) -> RuntimeSwitches:
    """Run-time switches of a variant (planes are removed from the data only in the inference tier)."""
    return RuntimeSwitches(switches=ab, all_planes=tuple(cfg.graph.planes),
                           data_planes_off=tuple(ab.planes_off) if tier == "inference" else ())


def build_variants(cfg: NagaHanaConfig) -> dict[str, Variant]:
    """Every variant of the model configuration (module docstring)."""
    out: dict[str, Variant] = {}

    def add(name: str, tier: str, first: int | None, desc: str, **sw: Any) -> None:
        out[name] = Variant(name=name, tier=tier, switches=AblationConfig(variant=name, tier=tier, **sw), first_stage=first,
                            description=desc)

    for lens in cfg.taaft.lenses:
        add(f"infer-lens-off-{lens}", "inference", None, f"TAAFT without the {lens} term, trained weights",
            lenses_off=(lens,))
        add(f"retrain-lens-off-{lens}", "single-stage", 2, f"TAAFT retrained without the {lens} term from stage 2",
            lenses_off=(lens,))
    add("infer-no-causal-heads", "inference", None, "TSTCT causal heads masked", tstct_causal_heads=False)
    add("infer-no-temporal-heads", "inference", None, "TSTCT temporal heads masked", tstct_temporal_heads=False)
    add("full-no-causal-heads", "full", 1, "trained without TSTCT causal heads", tstct_causal_heads=False)
    add("full-no-temporal-heads", "full", 1, "trained without TSTCT temporal heads", tstct_temporal_heads=False)
    add("infer-r1", "inference", None, "one loop pass (no weight-tied looping)", loop_passes=1)
    add("full-r1", "full", 1, "trained and evaluated with one loop pass", loop_passes=1)
    add("infer-no-environment", "inference", None, "no Environment carried across windows", environment=False)
    add("full-no-environment", "full", 1, "trained without the Environment carry", environment=False)
    add("infer-no-longterm", "inference", None, "TAAFT reads no long-term memory", longterm_memory=False)
    add("retrain-no-longterm", "single-stage", 2, "TAAFT retrained without long-term memory from stage 2",
        longterm_memory=False)
    add("infer-no-physics", "inference", None, "no physics term in TAAFT's energy", physics=False)
    add("full-no-physics", "full", 1, "trained without the physics boundary", physics=False)
    add("full-no-generator", "full", 1, "trained on real data only (no Generator variants)", generator=False)
    add("retrain-no-stage2", "single-stage", 3, "stage 3 from stage 1 without TAAFT pretraining", stage2_pretraining=False)
    for plane in cfg.graph.planes:
        add(f"infer-plane-off-{plane}", "inference", None, f"the {plane} plane removed from every window",
            planes_off=(plane,))
        add(f"full-plane-off-{plane}", "full", 1, f"CVG-AE built and trained without the {plane} plane",
            planes_off=(plane,))
    for v in out.values():
        validate_switches(v.switches, cfg)
    return out


@dataclass(frozen=True)
class PlanStep:
    """One step of a variant plan."""

    action: str               # "use-checkpoint" | "train" | "evaluate"
    stage: int
    source: str               # "main" (the main run's artefact) or "variant"


@dataclass(frozen=True)
class VariantPlan:
    """Exactly the work a variant needs (module docstring)."""

    variant: Variant
    steps: tuple[PlanStep, ...]
    upstream: tuple[str, int] | None      # ("main", stage) whose final checkpoint the variant starts from
    shares_prep: bool


def plan_variant(v: Variant) -> VariantPlan:
    """The tiered plan of a variant."""
    if v.tier == "inference":
        return VariantPlan(v, (PlanStep("use-checkpoint", 3, "main"), PlanStep("evaluate", 4, "variant")), ("main", 3), True)
    if v.tier == "single-stage":
        assert v.first_stage is not None
        up = v.first_stage - 1
        steps = [PlanStep("use-checkpoint", up, "main")]
        for s in range(v.first_stage, 4):
            if s == 2 and not v.switches.stage2_pretraining:
                continue
            steps.append(PlanStep("train", s, "variant"))
        steps.append(PlanStep("evaluate", 4, "variant"))
        return VariantPlan(v, tuple(steps), ("main", up), True)
    if v.tier == "full":
        steps = [PlanStep("train", s, "variant") for s in (1, 2, 3) if s != 2 or v.switches.stage2_pretraining]
        steps.append(PlanStep("evaluate", 4, "variant"))
        return VariantPlan(v, tuple(steps), None, v.switches.generator)
    raise InvariantViolation(f"unknown tier {v.tier!r}")


def ablation_manifest(main_run_dir: str | Path, variants: dict[str, Variant], lineage: dict[str, dict[str, Any]]) -> Path:
    """Write `<main run>/ablations/manifest.json`: every variant, its tier, switches, plan, status and lineage."""
    path = Path(main_run_dir) / "ablations" / "manifest.json"
    existing: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    rows = {}
    for name, v in sorted(variants.items()):
        plan = plan_variant(v)
        rec = dict(existing.get("variants", {}).get(name, {}))
        rec |= {"tier": v.tier, "description": v.description, "switches": dataclasses.asdict(v.switches),
                "plan": [dataclasses.asdict(s) for s in plan.steps],
                "upstream": list(plan.upstream) if plan.upstream else None, "shares_prep": plan.shares_prep}
        rec.setdefault("status", "planned")
        if name in lineage:
            rec |= lineage[name]
        rows[name] = rec
    data = {"main_run": str(main_run_dir), "variants": rows}
    atomic_write_text(path, json.dumps(data, indent=1, sort_keys=True, default=str))
    return path


__all__ = ["PlanStep", "RuntimeSwitches", "TIERS", "Variant", "VariantPlan", "ablation_manifest", "build_variants",
           "drop_planes", "head_mask_hooks", "load_without_lenses", "model_config", "plan_variant", "runtime_switches",
           "validate_switches"]
