"""Run helpers: a labelled capture → stream segments → one training stage (CLI, smoke script, tests).

Purpose
-------
The stage trainers take a `StreamLoader` and a `StreamBridge`; this module wires them for the common
case of one capture with a known labeller:

    load_capture(path, labeller)        PCAP → flow-state updates (D-51) → labels → PreparedSource
    stream_segments(src, cfg, seconds)  consecutive windows grouped into segments (AS-333)
    run_stage(stage, model, …)          stage 3, 4 or 5 in stream order with the carry (D-51)

Labellers are named explicitly; there is no default labeller (labels are dataset-specific facts, AS-34).
Splitting into train / test / zero-shot is the data pipeline's job (`data.sampling.assign_splits`);
`run_stage` trains on the segments it is given, so the caller decides which split they come from.

Decisions: D-22, D-23, D-51. Assumptions: AS-34, AS-333.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nagahana.core.errors import ConfigMissing
from nagahana.core.modes import RunMode, run_mode
from nagahana.data import labels as L
from nagahana.data.sampling import WindowRecord
from nagahana.data.stream import StreamLoader, plan_stream, segment_records
from nagahana.data.windows import PreparedSource, SourceData, prepare_source
from nagahana.models.config import NagaHanaConfig
from nagahana.models.nagahana import NagaHana
from nagahana.physics.term import PhysicsTerm
from nagahana.roles.contracts import HumanCommand
from nagahana.training.carry import StreamBridge
from nagahana.training.stage3 import Stage3Settings, Stage3Trainer
from nagahana.training.stage4 import Stage4Settings, Stage4Trainer
from nagahana.training.stage5 import Stage5Settings, Stage5Trainer

#: Labellers by name (AS-34): only rule sets that exist; never a default.
LABELLERS = {"cic2018-infiltration-slice": L.label_cic2018_infiltration_slice}


def load_capture(path: str | Path, *, labeller: str, network: str, sandboxed: bool) -> PreparedSource:
    """A capture as a prepared, labelled source (flow-state updates, D-51)."""
    from nagahana.ingest.pcap import PcapSource

    if labeller not in LABELLERS:
        raise ConfigMissing(f"unknown labeller {labeller!r}; known: {sorted(LABELLERS)}")
    cu = PcapSource(path, sandboxed=sandboxed, emit="flow-state").columnar()
    return prepare_source(SourceData(cu, LABELLERS[labeller](cu), network=network))


def stream_segments(src: PreparedSource, cfg: NagaHanaConfig, *, segment_seconds: float, source_index: int = 0,
                    windows: slice | None = None) -> list[WindowRecord]:
    """Segments of consecutive windows (AS-333); `windows` restricts to a run of the stream's windows."""
    stream = plan_stream(src, cfg)
    if windows is not None:
        stream = stream[windows]
    return segment_records(source_index, src, stream, segment_seconds=segment_seconds)


@dataclass
class StageRun:
    """The outcome of one stage run."""

    stage: int
    history: list[dict[str, float]]
    seconds: float
    notes: list[str]


def run_stage(stage: int, model: NagaHana, sources: Sequence[PreparedSource], segments: Sequence[WindowRecord], *,
              steps: int, seed: int, precision: str, lanes: int, physics: PhysicsTerm | None,
              verifier_command: HumanCommand | None = None, joint_after: int | None = None,
              perturb_seed: int | None = None, variants_total: int = 0) -> StageRun:
    """Run stage 3, 4 or 5 for at most `steps` optimiser steps over the segments, in stream order with the carry.

    variants_total: stage 3 only — Generator variants requested from the (training) segments and mixed into
    the loader, one variant segment after each real segment (needs the site MTU; `training/variants.py`).
    """
    cfg = model.cfg
    t0 = time.perf_counter()
    notes: list[str] = []
    with run_mode(RunMode.TRAIN):
        srcs, segs = list(sources), list(segments)
        order: list[int] | None = None
        if variants_total > 0:
            if stage != 3:
                raise ConfigMissing("Generator variants are mixed into stage-3 training only (build-spec §3)")
            from nagahana.training.variants import make_variant_sources

            vs = make_variant_sources(model, srcs, segs, total=variants_total, seed=seed)
            notes.append(f"variants: {len(vs.segments)} accepted of {vs.batch.requested} requested; "
                         f"rejections {vs.batch.rejection_counts()}")
            n_real = len(segs)
            srcs += vs.sources
            segs += vs.segments
            # interleave: real segment, variant segment, real segment, … (the rest in order)
            order = []
            for i in range(max(n_real, len(vs.segments))):
                if i < n_real:
                    order.append(i)
                if i < len(vs.segments):
                    order.append(n_real + i)
        loader = StreamLoader(srcs, segs, cfg, lanes=lanes, perturb_seed=perturb_seed, order=order)
        bridge = StreamBridge(model)
        if stage == 3:
            t3 = Stage3Trainer(model, settings=Stage3Settings.assumed(precision=precision), physics=physics, seed=seed)
            hist = t3.run(loader, bridge, max_steps=steps)
        elif stage == 4:
            t4 = Stage4Trainer(model, settings=Stage4Settings.assumed(precision=precision,
                                                                     perception_passes=cfg.tstct.default_passes),
                               physics=physics, seed=seed)
            hist = t4.run(loader, bridge, max_steps=steps)
        elif stage == 5:
            if joint_after is None:
                raise ConfigMissing("stage 5 needs joint_after (the length of the STAGED stop-gradient phase, AS-22)")
            t5 = Stage5Trainer(model, settings=Stage5Settings.assumed(precision=precision,
                                                                     perception_passes=cfg.tstct.default_passes,
                                                                     joint_after=joint_after),
                               physics=physics, seed=seed, verifier_command=verifier_command)
            hist = t5.run(loader, bridge, max_steps=steps)
            notes += t5.state.notes
        else:
            raise ConfigMissing(f"run_stage covers stages 3, 4 and 5; got {stage} (stage 6: training/stage6.py)")
    return StageRun(stage=stage, history=hist, seconds=time.perf_counter() - t0, notes=notes)


__all__ = ["LABELLERS", "StageRun", "load_capture", "run_stage", "stream_segments"]
