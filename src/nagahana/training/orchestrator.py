"""The training orchestrator: the stage graph run end to end, resumable at every stage (D-22).

Purpose
-------
`Orchestrator.run(["prep", "stage1", ..., "stage4"])` runs the pipeline of pipeline/stages.py on one
configuration (training/config.py):

    prep     data preparation (training/prep.py): sources, analytics, splits, augmentation plan
    stage1   Simulator pretraining from the initialisation          -> checkpoints/stage1/final.ckpt
    stage2   TAAFT pretraining from the stage-1 weights             -> checkpoints/stage2/final.ckpt
    stage3   full training from the stage-2 weights                 -> checkpoints/stage3/final.ckpt
    stage4   evaluation of the stage-3 weights (test and zero-shot), calibration proposals, site adapters

Dependencies are checked before a stage starts (`requirements`): the stage graph's consumed artefacts
must exist and verify (the upstream final checkpoint with its SHA-256 sidecar and model hash; the data
preparation, re-verified digest by digest). A stage whose final checkpoint exists is complete and is not
run again; a stage with resumable checkpoints and no final one continues from its latest checkpoint
(`resume`). Each stage records in the run manifest (training/manifest.py) its status, its checkpoint
lineage (the upstream checkpoint it started from, its own final checkpoint, digests and model hashes)
and its metrics.

Stage pools of training data (D-23, AS-575, AS-583, AS-590, AS-592):
    stages 1, 2   pretraining scope: train = pretrain-train segments + their variants (physics-gated);
                  validation = pretrain-validation segments (real)
    stage 3       full scope: train = full-train segments + their variants, re-screened with TAAFT's
                  marginal energy when `energy_acceptance` is "from-stage-3"; validation = full validation
                  segments (real)
    stage 4       test and zero-shot segments (real), evaluated separately

Human commands (D-21): stage 3's feedback learners need a HumanCommand("update-weights"); stage 4's
site adapters need a HumanCommand("update-site-adapter") and applying a calibration needs a
HumanCommand("apply-calibration"); the orchestrator never makes one up: they come from the caller
(the CLI's --approver / --reason).

Ablation variants (training/ablation.py): a variant's switches act through the model configuration,
the stage programs (fixed R, no long-term memory, no physics), the bridge (no Environment carry), the
data (planes removed at evaluation) and the plan (stages skipped or reused from the main run).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nagahana.core.config import decision_options_from_mapping, load_yaml, to_mapping
from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.core.modes import RunMode, run_mode
from nagahana.data.sampling import Role, WindowRecord
from nagahana.evaluation.predictions import save_outputs
from nagahana.governance import decisions
from nagahana.models.config import NagaHanaConfig, preset
from nagahana.models.nagahana import NagaHana, latent_space_hash, latent_space_id, model_hash, physics_term
from nagahana.physics.term import PhysicsTerm
from nagahana.roles.contracts import HumanCommand
from nagahana.training.ablation import head_mask_hooks, load_without_lenses, model_config, runtime_switches
from nagahana.training.augment import AugmentedData, reindex, select_pool
from nagahana.training.carry import StreamBridge
from nagahana.training.checkpoint import gather_model_state
from nagahana.training.config import TrainingRun
from nagahana.training.distributed import DistInfo, barrier, init_distributed, shutdown
from nagahana.training.engine import Engine, StageData, StageResult
from nagahana.training.manifest import RunManifest
from nagahana.training.prep import PrepResult, load_prep, run_prep
from nagahana.training.randomness import configure_determinism, derive_seed, seed_everything
from nagahana.training.runlog import make_logger
from nagahana.training.serialization import load_state, save_state, sha256_file
from nagahana.training.stage1 import Stage1Program
from nagahana.training.stage2 import Stage2Program
from nagahana.training.stage3 import Stage3Program
from nagahana.training.stage4 import (
    EvalSettings,
    SiteCalibrationSettings,
    evaluate_split,
    propose_calibration,
    split_loader,
)

FINAL = "final.ckpt"
STAGE_KEYS: tuple[str, ...] = ("prep", "stage1", "stage2", "stage3", "stage4")


def state_hash(state: dict[str, torch.Tensor]) -> str:
    """`models.nagahana.model_hash` of a state dict (names, dtypes, shapes, bytes in sorted order)."""
    import hashlib

    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        x = t.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        if x.numel():
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_cfg_of(cfg: TrainingRun) -> NagaHanaConfig:
    """The model configuration of a run: the preset, the site physics values, the ablation's lenses and planes."""
    base = preset(cfg.preset)
    base = dataclasses.replace(base, generator=dataclasses.replace(base.generator, mtu=cfg.mtu, link_bps=cfg.link_bps))
    return model_config(base, cfg.ablation, tier=cfg.ablation.tier)


class Orchestrator:
    """Runs the stage graph of one configuration (module docstring)."""

    def __init__(self, cfg: TrainingRun, *, feedback_command: HumanCommand | None = None,
                 site_command: HumanCommand | None = None, calibration_command: HumanCommand | None = None,
                 upstream: dict[int, str] | None = None) -> None:
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.feedback_command = feedback_command
        self.site_command = site_command
        self.calibration_command = calibration_command
        #: stage -> path of a final checkpoint to start from instead of this run's own (ablation lineage)
        self.upstream = dict(upstream or {})
        self.info: DistInfo | None = None
        self.prep: PrepResult | None = None
        self.results: dict[str, Any] = {}
        self.model_cfg = model_cfg_of(cfg)

    # ------------------------------------------------------------------ context
    @contextlib.contextmanager
    def _context(self) -> Iterator[DistInfo]:
        info = init_distributed(self.cfg.distributed, device=self.cfg.device)
        self.info = info
        seed_everything(derive_seed(self.cfg.seed, "global", info.rank))
        configure_determinism(self.cfg.deterministic)
        options: dict[str, str] = {}
        if self.cfg.decisions_file is not None:
            options = decision_options_from_mapping(load_yaml(self.cfg.decisions_file))
        try:
            with decisions.configure(options):
                yield info
        finally:
            shutdown(info)
            self.info = None

    def _new_model(self) -> NagaHana:
        torch.manual_seed(derive_seed(self.cfg.seed, "init"))
        model = NagaHana(self.model_cfg)
        assert self.info is not None
        return model.to(self.info.device)

    def _physics(self) -> PhysicsTerm | None:
        if not self.cfg.ablation.physics:
            return None
        if self.model_cfg.generator.mtu is None:
            raise ConfigMissing("run.mtu (the site MTU of the physics term) is required to train; it is never defaulted")
        return physics_term(self.model_cfg)

    # ------------------------------------------------------------------ artefacts
    def final_path(self, stage: int) -> Path:
        return self.run_dir / "checkpoints" / f"stage{stage}" / FINAL

    def _upstream_path(self, stage: int) -> Path:
        """The final checkpoint stage `stage` starts from: the previous stage's (stage 2 may be skipped)."""
        prev = stage - 1
        if stage == 3 and not self.cfg.ablation.stage2_pretraining:
            prev = 1
        if prev in self.upstream:
            return Path(self.upstream[prev])
        return self.final_path(prev)

    def requirements(self, key: str) -> list[str]:
        """Missing artefacts of a stage (empty = ready); verifies what exists."""
        missing: list[str] = []
        if key == "prep":
            if not self.cfg.data.sources:
                missing.append("data.sources (no source configured)")
            return missing
        if not (self.run_dir / "prep" / "splits.state").is_file() and self.prep is None:
            missing.append("the data preparation (train --prep)")
        stage = int(key[-1])
        if stage in (2, 3, 4):
            up = self._upstream_path(stage if stage != 4 else 4)
            if not up.is_file():
                missing.append(f"the upstream final checkpoint {up}")
        return missing

    def _write_final(self, stage: int, model: NagaHana, engine: Engine | None, *, upstream: Path | None,
                     result: StageResult | None) -> dict[str, Any]:
        """Gather and write the stage's final weights (rank 0); returns the lineage record."""
        assert self.info is not None
        if engine is not None:
            state = gather_model_state(model, engine.objective, self.info)
        else:
            state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()} if self.info.is_main else None
        record: dict[str, Any] = {}
        path = self.final_path(stage)
        if self.info.is_main:
            assert state is not None
            header = {"config": model.cfg.name, "stage": f"stage{stage}", "step": result.steps if result else 0,
                      "model_hash": state_hash(state), "latent_space_id": latent_space_id(model.cfg),
                      "latent_space_hash": latent_space_hash(model.cfg)}
            digest = save_state(path, {"state_dict": state, "extra": {}}, kind="nagahana-model", meta=header)
            record = {"final": str(path), "final_sha256": digest, "model_hash": header["model_hash"],
                      "upstream": str(upstream) if upstream is not None else None,
                      "upstream_sha256": sha256_file(upstream) if upstream is not None else None}
        barrier(self.info)
        return record

    def _load_upstream(self, model: NagaHana, path: Path) -> None:
        """Load a final checkpoint (verified before deserialisation; lens ablations drop exactly those lenses)."""
        blob, header = load_state(path, kind="nagahana-model")
        if header["latent_space_hash"] != latent_space_hash(model.cfg):
            raise InvariantViolation(f"{path}: latent space {header['latent_space_id']} differs from the model's (P-19)")
        if state_hash(blob["state_dict"]) != header["model_hash"]:
            raise InvariantViolation(f"{path}: weights do not match their recorded hash")
        state = {k: v.to(next(model.parameters()).device) for k, v in blob["state_dict"].items()}
        if self.cfg.ablation.lenses_off:
            load_without_lenses(model, state, tuple(self.cfg.ablation.lenses_off))
        else:
            model.load_state_dict(state)

    # ------------------------------------------------------------------ data pools
    def _pool(self, scope: str, stage: int, model: NagaHana) -> StageData:
        assert self.prep is not None
        prep = self.prep
        role_train = Role.TRAIN
        real_train = prep.splits.records(scope, role_train)
        val = prep.splits.records(scope, Role.VAL)
        sources = list(prep.sources)
        train: list[WindowRecord] = list(real_train)
        loop = self.cfg.stage_loop(stage)
        if prep.augmented is not None and loop.generated and self.cfg.ablation.generator:
            ids = {r.id for r in real_train}
            pool = prep.augmented
            keep_fn = None
            if stage == 3 and self.cfg.generator.energy_acceptance in ("from-stage-3", "always"):
                keep_fn = self._energy_screen(model, real_train)
            sel = select_pool(pool, keep=keep_fn)
            sel = AugmentedData(sources=[s for s, g in zip(sel.sources, sel.segments, strict=True) if g.derived_from in ids],
                                segments=[g for g in sel.segments if g.derived_from in ids], plan=sel.plan,
                                variants=[v for v, g in zip(sel.variants, sel.segments, strict=True) if g.derived_from in ids])
            sel = reindex(sel, len(sources))
            sources += sel.sources
            train += sel.segments
        limits = prep.splits.limits(scope)
        for g in train:
            if g.origin == "generated":
                limits[g.id] = limits.get(g.derived_from or "", -math.inf)
        return StageData(sources=sources, train=train, val=val, label_limits=limits)

    def _energy_screen(self, model: NagaHana, real_train: Sequence[WindowRecord]) -> Any:
        """JEM-style acceptance by TAAFT's marginal energy, calibrated on real training segments (AS-361, AS-583)."""
        from nagahana.models.generator.acceptance import EnergyAcceptance
        from nagahana.training.variants import marginal_energy_fn, segment_input

        assert self.prep is not None
        energy = marginal_energy_fn(model, passes=model.cfg.taaft.default_passes, descent_steps=model.cfg.taaft.descent_steps,
                                    max_windows=self.cfg.augment.energy_max_windows)
        acc = EnergyAcceptance(energy, quantiles=model.cfg.generator.energy_quantiles)
        reals = [segment_input(self.prep.sources[r.source], r, model.cfg.forecaster.n_techniques)[0] for r in real_train]
        if len(reals) < 2:
            return None
        acc.calibrate_on(reals)
        return lambda v: acc.accepts(float(energy(v.updates)))

    # ------------------------------------------------------------------ stages
    def _engine(self, program: Any, stage: int, model: NagaHana, data: StageData) -> Engine:
        assert self.info is not None
        loop = self.cfg.stage_loop(stage)
        sw = self.cfg.ablation
        logger = make_logger(self.cfg.logger, run_dir=self.run_dir, is_main=self.info.is_main, tracking_uri=self.cfg.tracking_uri)
        bridge_factory = (lambda: StreamBridge(model, carry_enabled=sw.environment))
        return Engine(program, data=data, loop=loop, info=self.info, seed=self.cfg.seed, run_dir=str(self.run_dir),
                      logger=logger, recompute=self.cfg.distributed.recompute, shard_units=self.cfg.distributed.shard_units,
                      dist_cfg=self.cfg.distributed, bridge_factory=bridge_factory, perturb=loop.perturb)

    def _program(self, stage: int, model: NagaHana, physics: PhysicsTerm | None) -> Any:
        sw = self.cfg.ablation
        if stage == 1:
            return Stage1Program(model, options=self.cfg.stage1.options, physics=physics, fixed_passes=sw.loop_passes)
        if stage == 2:
            return Stage2Program(model, options=self.cfg.stage2.options, physics=physics, fixed_passes=sw.loop_passes,
                                 use_longterm=sw.longterm_memory)
        return Stage3Program(model, options=self.cfg.stage3.options, physics=physics, optim=self.cfg.stage3.loop.optim,
                             command=self.feedback_command, info=self.info, fixed_passes=sw.loop_passes,
                             use_longterm=sw.longterm_memory)

    def run_stage(self, stage: int, *, resume: bool, stop_after: int | None = None) -> dict[str, Any]:
        """Train stage 1, 2 or 3 (dependencies checked; a completed stage is not rerun)."""
        assert self.info is not None
        key = f"stage{stage}"
        if self.final_path(stage).is_file():
            return {"status": "complete", "final": str(self.final_path(stage))}
        missing = self.requirements(key)
        if missing:
            raise InvariantViolation(f"{key} cannot start; missing: {missing}")
        model = self._new_model()
        upstream = None
        if stage > 1:
            upstream = self._upstream_path(stage)
            self._load_upstream(model, upstream)
        undo_heads = head_mask_hooks(model, self.cfg.ablation)
        try:
            physics = self._physics()
            program = self._program(stage, model, physics)
            data = self._pool("pretrain" if stage in (1, 2) else "full", stage, model)
            engine = self._engine(program, stage, model, data)
            with run_mode(RunMode.TRAIN):
                result = engine.run(resume=resume, stop_after=stop_after)
            if result.interrupted:
                return {"status": "interrupted", "steps": result.steps, "notes": result.notes}
            record = self._write_final(stage, model, engine, upstream=upstream, result=result)
        finally:
            undo_heads()
        out = {"status": "complete", "steps": result.steps, "epochs": result.epochs_completed,
               "stopped_early": result.stopped_early, "best_metric": result.best_metric, "best_step": result.best_step,
               "validations": result.validations[-5:], "notes": result.notes + getattr(program, "notes", [])} | record
        return out

    def run_stage4(self) -> dict[str, Any]:
        """Evaluate the stage-3 weights on the test and zero-shot splits; calibration; site adapters."""
        assert self.info is not None and self.prep is not None
        missing = self.requirements("stage4")
        if missing:
            raise InvariantViolation(f"stage4 cannot start; missing: {missing}")
        s4 = self.cfg.stage4
        model = self._new_model()
        up = self._upstream_path(4)
        self._load_upstream(model, up)
        switches = runtime_switches(self.cfg.ablation, model_cfg_of(dataclasses.replace(self.cfg, ablation=dataclasses.replace(
            self.cfg.ablation, planes_off=()))), tier=self.cfg.ablation.tier)
        undo_heads = head_mask_hooks(model, self.cfg.ablation)
        out_dir = self.run_dir / "evaluation"
        record: dict[str, Any] = {"upstream": str(up), "upstream_sha256": sha256_file(up), "splits": {}}
        try:
            settings = EvalSettings.for_model(model, threshold=s4.threshold, seed=self.cfg.seed, tstct_passes=s4.tstct_passes,
                                              taaft_passes=s4.taaft_passes, descent_steps=s4.descent_steps,
                                              horizon_k=s4.horizon_k, routes_n=s4.routes_n)
            meta_of = {r.id: {"dataset": self.prep.sources[r.source].data.dataset, "network": r.network}
                       for r in self.prep.segments}
            physics = self._physics()
            all_pairs: dict[str, tuple[list[float], list[float]]] = {}
            for split in s4.eval_splits:
                role = Role.TEST if split == "test" else Role.ZERO_SHOT
                segs = self.prep.splits.records("full", role)
                mine = segs[self.info.rank::self.info.world]
                loader = split_loader(self.prep.sources, mine, self.prep.splits.full, model.cfg, role=role, lanes=s4.lanes)
                with run_mode(RunMode.EVALUATE):
                    rep = evaluate_split(model, loader, StreamBridge(model, carry_enabled=self.cfg.ablation.environment),
                                         settings=settings, split=split, meta_of=meta_of, switches=switches, physics=physics,
                                         device=self.info.device)
                from nagahana.training.distributed import gather_objects

                reps = gather_objects(self.info, rep)
                if self.info.is_main:
                    merged = _merge_reports(reps, threshold=s4.threshold)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    if merged["outputs"] is not None:
                        save_outputs(merged["outputs"], out_dir / f"{split}.npz")
                    record["splits"][split] = {"groups": merged["groups"], "triggers_seen": merged["triggers"],
                                               "outputs": str(out_dir / f"{split}.npz") if merged["outputs"] is not None else None}
                    for fam, (p, y) in merged["calibration"].items():
                        all_pairs.setdefault(fam, ([], []))
                        all_pairs[fam][0].extend(p)
                        all_pairs[fam][1].extend(y)
            if self.info.is_main:
                from nagahana.training.feedback import CalibrationPairs

                pairs = CalibrationPairs(pairs={f: (torch.tensor(p, dtype=torch.float64), torch.tensor(y, dtype=torch.float64))
                                                for f, (p, y) in all_pairs.items() if p})
                proposals = propose_calibration(model, pairs, s4.calibration)
                record["calibration"] = [dataclasses.asdict(p) | {"temperatures": dict(p.temperatures)} for p in proposals]
                record["calibration_applied"] = False
                if proposals and self.calibration_command is not None:
                    from nagahana.models.verifier.reports import TemperatureState
                    from nagahana.training.stage4 import apply_site_temperature

                    state = TemperatureState()
                    for prop in proposals:
                        state = apply_site_temperature(prop, self.calibration_command, state=state)
                    (out_dir / "temperatures.json").write_text(json.dumps({"temperatures": dict(state.temperatures),
                                                                          "applied_by": [c.approver for c in state.applied_by]},
                                                                         indent=1), encoding="utf-8")
                    record["calibration_applied"] = True
                (out_dir / "report.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
            if s4.site_calibration:
                record["site"] = self._site_calibration(model)
        finally:
            undo_heads()
        barrier(self.info)
        return record

    def _site_calibration(self, model: NagaHana) -> dict[str, Any]:
        """LoRA site adapters on the configured site traffic, under the human command (AS-26)."""
        from nagahana.training.stage4 import SiteCalibrator

        if self.site_command is None:
            return {"status": "not run: no HumanCommand('update-site-adapter') was given (D-21)"}
        s4 = self.cfg.stage4
        settings = SiteCalibrationSettings.assumed(model, precision=s4.site_loop.precision, allow_short=s4.site_allow_short,
                                                   optim=s4.site_loop.optim)
        settings = dataclasses.replace(settings, min_span_s=s4.site_min_span_s, min_alerts=s4.site_min_alerts,
                                       alert_weight=s4.site_alert_weight)
        cal = SiteCalibrator(model, settings=settings, physics=self._physics(), seed=self.cfg.seed, command=self.site_command,
                             total_steps=s4.site_loop.max_steps)
        return {"status": "attached", "adapters": cal.adapter_paths}

    # ------------------------------------------------------------------ the run
    def run(self, keys: Sequence[str], *, resume: bool = True, stop_after: int | None = None) -> dict[str, Any]:
        """Run the given stages in order (module docstring); returns the per-stage records."""
        unknown = [k for k in keys if k not in STAGE_KEYS]
        if unknown:
            raise ConfigMissing(f"unknown stages {unknown}; known: {STAGE_KEYS}")
        with self._context() as info:
            manifest = RunManifest(self.run_dir, self.cfg, info=info) if info.is_main else None
            for key in STAGE_KEYS:
                if key not in keys:
                    continue
                if key == "stage2" and not self.cfg.ablation.stage2_pretraining:
                    self.results[key] = {"status": "skipped (ablation: no stage-2 pretraining)"}
                    continue
                t0 = time.perf_counter()
                if key == "prep":
                    if (self.run_dir / "prep" / "splits.state").is_file() and resume:
                        self.prep = load_prep(self.cfg, self.model_cfg, run_dir=self.run_dir)
                        rec: dict[str, Any] = {"status": "complete (restored)"}
                    else:
                        with run_mode(RunMode.TRAIN):
                            self.prep = run_prep(self.cfg, self.model_cfg, run_dir=self.run_dir, is_main=info.is_main)
                        rec = {"status": "complete", "notes": self.prep.notes}
                    rec["digests"] = self.prep.digests
                else:
                    if self.prep is None:
                        self.prep = load_prep(self.cfg, self.model_cfg, run_dir=self.run_dir)
                    rec = self.run_stage4() if key == "stage4" else self.run_stage(int(key[-1]), resume=resume,
                                                                                  stop_after=stop_after)
                rec["seconds"] = time.perf_counter() - t0
                self.results[key] = rec
                if manifest is not None:
                    manifest.stage(key, rec)
                    manifest.write()
                if rec.get("status") == "interrupted":
                    break
        return self.results


def _merge_reports(reps: Sequence[Any], *, threshold: float) -> dict[str, Any]:
    """Merge the ranks' split reports: concatenated outputs, recomputed group metrics and pairs."""
    import pandas as pd

    from nagahana.evaluation.predictions import ForecastPredictions, ModelOutputs, StagePredictions, TimeToEventPredictions
    from nagahana.training.stage4 import group_metrics

    reps = [r for r in reps if r is not None]
    groups_pairs: dict[str, tuple[list[float], list[float]]] = {}
    for r in reps:
        for g, (p, y) in r.pairs.items():
            groups_pairs.setdefault(g, ([], []))
            groups_pairs[g][0].extend(p)
            groups_pairs[g][1].extend(y)
    groups = {g: group_metrics(p, y, threshold=threshold) for g, (p, y) in groups_pairs.items()}
    outs = [r.outputs for r in reps if r.outputs is not None]
    merged: ModelOutputs | None = None
    if outs:
        merged = ModelOutputs(model=outs[0].model, protocol=outs[0].protocol, seed=outs[0].seed, config=outs[0].config)
        fs = [o.forecast for o in outs if o.forecast is not None]
        if fs:
            k = fs[0].p_inf.shape[1]
            n_r = max(f.ensemble.shape[1] for f in fs if f.ensemble is not None)
            ens = [np.pad(f.ensemble, ((0, 0), (0, n_r - f.ensemble.shape[1]), (0, 0))) for f in fs if f.ensemble is not None]
            wts = [np.pad(f.ensemble_weight, ((0, 0), (0, n_r - f.ensemble_weight.shape[1]))) for f in fs
                   if f.ensemble_weight is not None]
            merged.forecast = ForecastPredictions(
                p_inf=np.concatenate([f.p_inf for f in fs]), window_seconds=fs[0].window_seconds,
                event_step=np.concatenate([f.event_step for f in fs]), observed_steps=np.concatenate([f.observed_steps for f in fs]),
                meta=pd.concat([f.meta for f in fs], ignore_index=True),
                hazard=np.concatenate([f.hazard for f in fs if f.hazard is not None]) if all(f.hazard is not None for f in fs) else None,
                ensemble=np.concatenate(ens) if len(ens) == len(fs) else None,
                ensemble_weight=np.concatenate(wts) if len(wts) == len(fs) else None)
            del k
        ss = [o.stage for o in outs if o.stage is not None]
        if ss:
            merged.stage = StagePredictions(probs=np.concatenate([s.probs for s in ss]), label=np.concatenate([s.label for s in ss]),
                                            stage_names=ss[0].stage_names, meta=pd.concat([s.meta for s in ss], ignore_index=True))
        ts = [o.time_to_event for o in outs if o.time_to_event is not None]
        if ts:
            merged.time_to_event = TimeToEventPredictions(
                risk=np.concatenate([t.risk for t in ts]), survival=np.concatenate([t.survival for t in ts]),
                time_grid=ts[0].time_grid, event_time=np.concatenate([t.event_time for t in ts]),
                event_observed=np.concatenate([t.event_observed for t in ts]), meta=pd.concat([t.meta for t in ts], ignore_index=True))
    calib: dict[str, tuple[list[float], list[float]]] = {}
    for r in reps:
        if r.calibration is None:
            continue
        for fam, (p, y) in r.calibration.pairs.items():
            calib.setdefault(fam, ([], []))
            calib[fam][0].extend(p.tolist())
            calib[fam][1].extend(y.tolist())
    return {"groups": groups, "outputs": merged, "triggers": sum(r.triggers_seen for r in reps), "calibration": calib}


def describe_config(cfg: TrainingRun) -> dict[str, Any]:
    """The configuration as plain data (manifests, the CLI's dry run)."""
    return to_mapping(cfg)


__all__ = ["FINAL", "Orchestrator", "STAGE_KEYS", "describe_config", "model_cfg_of", "state_hash"]
