"""End-to-end smoke run: stages 3, 4, 5 and a forensic replay on a slice of a real capture (tiny preset).

Purpose
-------
Prove that the composed model trains and infers end to end on real traffic (not a claim about
accuracy: the tiny preset is a test fixture and a few steps teach it nothing about attacks). Used by
`scripts/smoke_e2e.py` and `tests/test_integration_e2e.py`.

Procedure
---------
1. Cut [t0, t1) out of the capture (`cut_capture`, dpkt) and label it with a named labeller (AS-34).
2. Stream plan of the cut (D-51); pick the first window holding a cadence trigger and the windows
   before it (so the TSTCT carry and the long-term memory are exercised), as one segment.
3. For each stage: run the trainer over the segment in stream order (carry written window by window),
   then take the trigger window's prepared batch and check the stage loss on that **same batch with
   the same random draws** (a fixed seed) before and after a few optimiser steps: finite and lower.
4. Replay a second, shorter cut through the inference engine (RunMode.FORENSIC_REPLAY): at least one
   `ForecastBundle` (its invariants are validated on construction) and a `ForensicReport`.

The tiny preset's optimiser warm-up (2,000 steps) would make a few steps invisible, so the smoke run
uses lr = 1e-3 without warm-up (a test setting, recorded in the report; the presets are unchanged).
"""

from __future__ import annotations

import calendar
import dataclasses
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from nagahana.core.modes import RunMode, run_mode
from nagahana.data.stream import StreamLoader, plan_stream, segment_records
from nagahana.data.windows import PreparedSource
from nagahana.inference.engine import Engine, EngineSettings, TriggerResult
from nagahana.inference.forensic import replay
from nagahana.models.config import NagaHanaConfig, preset
from nagahana.models.nagahana import NagaHana, physics_term
from nagahana.physics.term import PhysicsTerm
from nagahana.roles.contracts import ForensicReport, HumanCommand
from nagahana.training.carry import PreparedBatch, StreamBridge
from nagahana.training.common import generator
from nagahana.training.runs import load_capture, run_stage
from nagahana.training.stage3 import Stage3Settings, Stage3Trainer, stage3_loss
from nagahana.training.stage4 import Stage4Settings, Stage4Trainer, frozen_perception, stage4_step_loss
from nagahana.training.stage5 import Stage5Settings, Stage5Trainer


def utc(day: tuple[int, int, int], h: int, m: int, s: int) -> float:
    """Epoch seconds of a UTC date and time."""
    return float(calendar.timegm((*day, h, m, s)))


def cut_capture(src: Path, dst: Path, t0: float, t1: float) -> int:
    """Copy the packets with t0 ≤ ts < t1 of a time-sorted capture into a new pcap. Returns the packet count."""
    import dpkt  # the `pcap` extra (also used by the PCAP adapter)

    n = 0
    with open(src, "rb") as fh, open(dst, "wb") as out:
        writer = dpkt.pcap.Writer(out, linktype=dpkt.pcap.DLT_EN10MB)
        for ts, buf in dpkt.pcap.Reader(fh):
            if ts < t0:
                continue
            if ts >= t1:
                break
            writer.writepkt(buf, ts=ts)
            n += 1
    return n


def smoke_config(mtu: float) -> NagaHanaConfig:
    """preset("tiny") with the site MTU set and a smoke optimiser (lr 1e-3, no warm-up)."""
    cfg = preset("tiny")
    return dataclasses.replace(cfg, generator=dataclasses.replace(cfg.generator, mtu=mtu),
                               training=dataclasses.replace(cfg.training, lr=1e-3, warmup_steps=0))


@dataclass
class SmokeReport:
    """Timings (seconds), loss checks and outputs of a smoke run."""

    timings: dict[str, float] = field(default_factory=dict)
    stream_losses: dict[int, list[float]] = field(default_factory=dict)
    same_batch: dict[int, tuple[float, float]] = field(default_factory=dict)     # stage → (before, after)
    results: list[TriggerResult] = field(default_factory=list)
    report: ForensicReport | None = None
    engine: Engine | None = None
    notes: list[str] = field(default_factory=list)
    updates: int = 0
    windows: int = 0


def _trigger_segment(src: PreparedSource, cfg: NagaHanaConfig, before: int) -> list:
    """One segment: the first trigger-bearing window of the stream and up to `before` windows before it."""
    stream = plan_stream(src, cfg)
    c = cfg.forecaster.window_seconds
    for k, w in enumerate(stream):
        g = math.ceil(w.trigger_lo / c) * c                                       # first cadence point ≥ the window start
        if g < w.trigger_hi:
            return segment_records(0, src, stream[max(0, k - before):k + 1], segment_seconds=1e12)
    raise RuntimeError("no window of the cut holds a cadence trigger: widen the cut")


def _last_prepared(model: NagaHana, src: PreparedSource, segs: list, cfg: NagaHanaConfig) -> tuple[PreparedBatch, StreamBridge]:
    """Run the segment's windows through a fresh bridge (perception only) and return the last prepared batch."""
    bridge = StreamBridge(model)
    prep = None
    with torch.no_grad():
        for w, lab, ctxs in StreamLoader([src], segs, cfg, lanes=1):
            if prep is not None:
                _, env = model.perceive(prep.window, sample=False, passes=cfg.tstct.default_passes, carry=prep.carry)
                bridge.commit(prep, env)
            prep = bridge.prepare(w, lab, ctxs)
    assert prep is not None
    return prep, bridge


def _same_batch(fn: Callable[[int], torch.Tensor], step: Callable[[torch.Tensor], None], steps: int) -> tuple[float, float]:
    """Loss on one batch with fixed random draws (seed 1234) before and after `steps` optimiser steps."""
    before = float(fn(1234).detach())
    for i in range(steps):
        step(fn(100 + i))
    after = float(fn(1234).detach())
    return before, after


def same_batch_check(stage: int, model: NagaHana, src: PreparedSource, segs: list, *, physics: PhysicsTerm, seed: int,
                     steps: int) -> tuple[float, float]:
    """Stage loss on the segment's last (trigger) window with fixed draws, before and after `steps` steps."""
    cfg = model.cfg
    prep, bridge = _last_prepared(model, src, segs, cfg)
    passes = cfg.tstct.default_passes
    if stage == 3:
        t3 = Stage3Trainer(model, settings=Stage3Settings.assumed(precision="fp32"), physics=physics, seed=seed)

        def f3(s: int) -> torch.Tensor:
            return stage3_loss(model, prep, settings=t3.settings, physics=physics, gen=generator(s))[0]

        def s3(loss: torch.Tensor) -> None:
            t3.optim.step(loss)

        return _same_batch(f3, s3, steps)
    if stage == 4:
        t4 = Stage4Trainer(model, settings=Stage4Settings.assumed(precision="fp32", perception_passes=passes),
                           physics=physics, seed=seed)
        lat, env = frozen_perception(model, prep, passes=passes, precision="fp32")

        def f4(s: int) -> torch.Tensor:
            return stage4_step_loss(model, prep, lat, env, settings=t4.settings, physics=physics, gen=generator(s),
                                    negative="swap", longterm=bridge.longterm_states(prep, env), past=None)[0]

        def s4(loss: torch.Tensor) -> None:
            t4.optim.step(loss)
            t4.verify_frozen()                                                     # the perceptors never move

        return _same_batch(f4, s4, steps)
    t5 = Stage5Trainer(model, settings=Stage5Settings.assumed(precision="fp32", perception_passes=passes, joint_after=1),
                       physics=physics, seed=seed, verifier_command=None)
    lat5, env5 = frozen_perception(model, prep, passes=passes, precision="fp32")

    def f5(s: int) -> torch.Tensor:
        t5.gen = generator(s)
        return t5.main_loss(prep, lat5, env5, bridge.longterm_states(prep, env5))[0]

    def s5(loss: torch.Tensor) -> None:
        t5.optim.step(loss)

    return _same_batch(f5, s5, steps)


def run_smoke(capture: Path, workdir: Path, *, train_cut: tuple[float, float], replay_cut: tuple[float, float],
              labeller: str, mtu: float, seed: int, steps: int, windows_before: int, attribution_samples: int,
              routes_n: int) -> SmokeReport:
    """The smoke procedure of the module docstring. Returns timings, loss checks and the replay outputs."""
    rep = SmokeReport()
    t = time.perf_counter()
    workdir.mkdir(parents=True, exist_ok=True)
    train_pcap, replay_pcap = workdir / "train-cut.pcap", workdir / "replay-cut.pcap"
    cut_capture(capture, train_pcap, *train_cut)
    cut_capture(capture, replay_pcap, *replay_cut)
    rep.timings["cut"] = time.perf_counter() - t
    torch.manual_seed(seed)
    cfg = smoke_config(mtu)
    model = NagaHana(cfg)
    physics = physics_term(cfg)
    t = time.perf_counter()
    src = load_capture(train_pcap, labeller=labeller, network="smoke", sandboxed=False)
    segs = _trigger_segment(src, cfg, windows_before)
    rep.updates, rep.windows = len(src.order), len(plan_stream(src, cfg))
    rep.timings["ingest"] = time.perf_counter() - t
    cmd = HumanCommand("smoke-test", "update-weights", "smoke run of the Verifier's training path", time.time())
    for stage in (3, 4, 5):
        # ---- stream-order run over the segment (carry), then the same-batch check
        t = time.perf_counter()
        # stage 3 mixes Generator variants of the segment into a second lane (D-40; training split only)
        run = run_stage(stage, model, [src], segs, steps=steps, seed=seed, precision="fp32", lanes=2 if stage == 3 else 1,
                        physics=physics, verifier_command=cmd if stage == 5 else None, joint_after=1,
                        variants_total=2 if stage == 3 else 0)
        rep.stream_losses[stage] = [h.get(f"stage{stage}/total", float("nan")) for h in run.history]
        rep.notes += run.notes
        rep.timings[f"stage{stage}_stream"] = time.perf_counter() - t
        t = time.perf_counter()
        with run_mode(RunMode.TRAIN):
            rep.same_batch[stage] = same_batch_check(stage, model, src, segs, physics=physics, seed=seed, steps=steps)
        rep.timings[f"stage{stage}_same_batch"] = time.perf_counter() - t
    # ---- forensic replay of the second cut
    t = time.perf_counter()
    settings = EngineSettings.assumed(model, network="smoke")
    settings = dataclasses.replace(settings, attribution_samples=attribution_samples,
                                   budgets=dataclasses.replace(settings.budgets, routes_n=routes_n))
    report, engine = replay(model, replay_pcap, settings=settings, physics=physics, seed=seed, sandboxed=False)
    rep.results, rep.report, rep.engine = list(engine.results), report, engine
    rep.timings["replay"] = time.perf_counter() - t
    rep.notes += engine.notes
    return rep


__all__ = ["SmokeReport", "cut_capture", "run_smoke", "smoke_config", "utc"]
