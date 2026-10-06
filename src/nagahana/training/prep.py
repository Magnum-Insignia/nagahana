"""Data preparation: ingest, analytics, cleaning, splitting and the augmentation plan (D-22 steps 1 and 2).

Purpose
-------
"Deep data analysis of raw open datasets, documented [Q-02]" and "Preparation: cleaning, splitting,
curation of attacks, normalisation, augmentation with the Generator" (architecture section 5). The
steps, in order:

1. Ingest every configured source through its adapter (ingest/pcap.py in flow-state mode, D-51;
   ingest/csv_flows.py) with its label mapping (data/labels.py, AS-34).
2. Cleaning (AS-589): exact duplicate records (the same raw-record SHA-256, a record that was written
   twice) are dropped, keeping the first; nothing else is altered: values that are not measurements are
   already NOT_SUPPLIED (AS-302) and absence stays absence (D-41). Normalisation is the input layer's own
   (signed log1p and periodic embeddings, AS-31), so no statistic of any split enters the data.
3. Analytics (AS-589): per source and in total, written to `prep/analysis.json`: records, time span,
   entities by kind, label coverage and families, the contributing share of every column by status,
   quantiles of every numeric column over its contributing cells, the reorder uncertainty, windows,
   segments and cadence triggers, and the digests of the raw files and of the prepared content. No
   physics residual scores the telemetry (D-25 held, AS-40).
4. Splitting (AS-590): segments (AS-333), the full split plan (60 % train, 20 % test, 20 % validation;
   zero-shot = novel families and held-out networks, D-16 held -> AS-35) and, nested in its training
   split, the pretraining plan of stages 1 and 2 (70 % / 30 %); both validated (D-23, P-23), written
   to `prep/splits.state`.
5. The Generator (D-40, D-14 option in force): the tool-class table of fingerprints and the learned
   families (codec, masked model, diffusion, energy-SSL), fitted on training segments only (AS-594,
   AS-367), written to `prep/generator.state`; then the class- and family-balanced augmentation plan of
   the training split (training/augment.py, AS-582), written to `prep/augmentation.state` with every
   variant's provenance and digest (AS-596).

`run_prep` returns everything later stages need; `load_prep` restores it in another process, re-ingesting
the sources and verifying every digest, and regenerating the variants from the stored plan and Generator
state (verified variant by variant).

Decisions: D-14 (held), D-16 (held), D-22, D-23, D-25 (held), D-40, D-41, D-51. Assumptions: AS-34, AS-35,
AS-302, AS-333, AS-367, AS-582, AS-589, AS-590, AS-594, AS-596.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.core.modes import RunMode, run_mode
from nagahana.data.sampling import Role, WindowRecord
from nagahana.data.stream import plan_stream
from nagahana.data.windows import PreparedSource, SourceData, prepare_source
from nagahana.datamodel.columnar import STATUS_ORDER, ColumnarUpdates
from nagahana.datamodel.fields import Kind
from nagahana.datamodel.status import CONTRIBUTING
from nagahana.inference.buffer import take_rows
from nagahana.models.config import NagaHanaConfig
from nagahana.models.generator.acceptance import AcceptanceGate
from nagahana.models.generator.codec import FieldCodec
from nagahana.models.generator.config import LEARNED_FAMILIES, GeneratorPolicy
from nagahana.models.generator.diffusion import TabularDiffusion
from nagahana.models.generator.energy import EnergyProducer, EnergySSL, fit_energy_ssl
from nagahana.models.generator.learned import DiffusionProducer, MaskedProducer, fit_diffusion, fit_masked
from nagahana.models.generator.limits import PhysicalSetting
from nagahana.models.generator.masked import build_masked_model
from nagahana.models.generator.model import active_families
from nagahana.models.generator.pipeline import Producer
from nagahana.models.generator.transforms import FingerprintClasses
from nagahana.training.assumptions import use
from nagahana.training.augment import AugmentationPlan, AugmentedData, draw_variants, verify_replay
from nagahana.training.config import TrainingRun
from nagahana.training.data import SplitPlan, ingest_source, plan_splits, source_digest, source_segments, window_has_trigger
from nagahana.training.serialization import atomic_write_text, load_state, save_state, sha256_file
from nagahana.training.variants import producers_for, segment_input

PREP_DIR = "prep"


@dataclass
class PrepResult:
    """What the data preparation hands to the stages."""

    sources: list[PreparedSource]
    segments: list[WindowRecord]
    splits: SplitPlan
    augmented: AugmentedData | None
    analysis: dict[str, Any]
    digests: dict[str, str]
    generator_state: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)


def deduplicate(data: SourceData) -> tuple[SourceData, int]:
    """Drop exact duplicate records (same raw-record SHA-256; all-zero digests are not records) (AS-589)."""
    use("AS-589", by=__name__)
    cu = data.updates
    raw = cu.raw_hash
    nonzero = raw.any(axis=1)
    keys = [bytes(r) if nz else None for r, nz in zip(raw, nonzero, strict=True)]
    seen: set[bytes] = set()
    keep = []
    for i, k in enumerate(keys):
        if k is not None and k in seen:
            continue
        if k is not None:
            seen.add(k)
        keep.append(i)
    dropped = len(cu) - len(keep)
    if dropped == 0:
        return data, 0
    rows = np.asarray(keep, dtype=np.int64)
    new_cu = take_rows(cu, rows)
    lab = data.labels.set_index("seq").loc[cu.updates["seq"].to_numpy()[rows]].reset_index()
    lab["seq"] = np.arange(len(rows), dtype=np.int64)
    return dataclasses.replace(data, updates=new_cu, labels=lab[list(data.labels.columns)]), dropped


def _column_profile(cu: ColumnarUpdates, quantiles: tuple[float, ...]) -> dict[str, Any]:
    """Status shares and value quantiles of every column (contributing cells only)."""
    contrib_codes = [i for i, s in enumerate(STATUS_ORDER) if s in CONTRIBUTING]
    out: dict[str, Any] = {}
    n = max(1, len(cu))
    for j, c in enumerate(cu.columns):
        st = cu.status[:, j]
        shares = {STATUS_ORDER[k].value: float((st == k).sum()) / n for k in range(len(STATUS_ORDER))}
        prof: dict[str, Any] = {"kind": c.kind.value, "status_share": shares}
        m = np.isin(st, contrib_codes)
        v = cu.values[m, j]
        if v.size and c.kind in (Kind.CONTINUOUS, Kind.COUNT, Kind.HISTOGRAM):
            prof["quantiles"] = dict(zip([str(q) for q in quantiles], np.quantile(v, quantiles).tolist(), strict=True))
            prof["mean"] = float(v.mean())
        elif v.size:
            codes, counts = np.unique(np.round(v).astype(np.int64), return_counts=True)
            top = np.argsort(-counts)[:20]
            prof["top_codes"] = {str(int(codes[i])): int(counts[i]) for i in top}
            prof["distinct"] = int(codes.size)
        out[c.name] = prof
    return out


def analyse_source(src: PreparedSource, cfg: NagaHanaConfig, *, quantiles: tuple[float, ...], file_sha256: str,
                   dropped: int) -> dict[str, Any]:
    """The analytics of one prepared source (module docstring, step 3)."""
    cu = src.data.updates
    fam = pd.Series(src.family).astype(str)
    mal = src.malicious
    stream = plan_stream(src, cfg)
    cadence = cfg.forecaster.window_seconds
    return {
        "source_id": cu.source_id, "network": src.data.network, "dataset": src.data.dataset, "adapter": cu.adapter,
        "adapter_version": cu.adapter_version, "file_sha256": file_sha256, "content_sha256": source_digest(src),
        "records": int(len(cu)), "duplicates_dropped": int(dropped),
        "time": {"first": float(src.time[0]) if len(src.time) else None, "last": float(src.time[-1]) if len(src.time) else None,
                 "span_s": float(src.time[-1] - src.time[0]) if len(src.time) else 0.0, "timeless": bool(src.timeless)},
        "entities": {k: int(v) for k, v in cu.entities["kind"].astype(str).value_counts().items()},
        "labels": {"malicious": int(np.nansum(mal == 1.0)), "benign": int(np.nansum(mal == 0.0)),
                   "unknown": int(np.isnan(mal).sum()), "families": {k: int(v) for k, v in fam.value_counts().items()}},
        "reorder_uncertainty_s": {"mean": float(np.mean(src.reorder)) if len(src.reorder) else 0.0,
                                  "max": float(np.max(src.reorder)) if len(src.reorder) else 0.0},
        "windows": len(stream), "windows_with_trigger": int(sum(window_has_trigger(w, cadence) for w in stream)),
        "columns": _column_profile(cu, quantiles),
    }


def _fit_learned(mcfg: NagaHanaConfig, train: list[WindowRecord], sources: list[PreparedSource], cfg: TrainingRun, *,
                 families: tuple[str, ...], setting: PhysicalSetting) -> tuple[dict[str, Producer], dict[str, Any], list[str]]:
    """Fit the active learned families on training segments only (AS-594); returns (producers, state, ids)."""
    use("AS-594", by=__name__)
    g = mcfg.generator
    learned = [f for f in families if f in LEARNED_FAMILIES]
    if not learned:
        return {}, {}, []
    windows = [segment_input(sources[r.source], r, mcfg.forecaster.n_techniques) for r in train if r.origin == "real"]
    ids = [s.id for _, _, s in windows]
    codec = FieldCodec.fit([w[0] for w in windows], value_bins=g.value_bins, cat_vocab=g.cat_vocab, fitted_on=ids)
    producers: dict[str, Producer] = {}
    state: dict[str, Any] = {"codec": codec, "families": tuple(learned)}
    a = cfg.augment
    from nagahana.training.randomness import derive_seed

    if "masked-generative" in learned or "autoregressive" in learned:
        masked = build_masked_model(g, codec)
        fit_masked(masked, codec, windows, g, steps=a.fit_steps, lr=a.fit_lr, seed=derive_seed(cfg.seed, "fit-masked"))
        state["masked"] = masked.state_dict()
        if "masked-generative" in learned:
            producers["masked-generative"] = MaskedProducer(masked, codec, g, setting, order="maskgit")
        if "autoregressive" in learned:
            producers["autoregressive"] = MaskedProducer(masked, codec, g, setting, order="left-to-right")
    if "diffusion" in learned:
        diff = TabularDiffusion(g, codec)
        fit_diffusion(diff, windows, steps=a.fit_steps, lr=a.fit_lr, seed=derive_seed(cfg.seed, "fit-diffusion"))
        state["diffusion"] = diff.state_dict()
        producers["diffusion"] = DiffusionProducer(diff, g, setting)
    if "energy-ssl" in learned:
        ssl = EnergySSL(g, codec, cfg.generator)
        fit_energy_ssl(ssl, windows, steps=a.fit_steps, lr=a.fit_lr, seed=derive_seed(cfg.seed, "fit-energy-ssl"))
        state["energy"] = ssl.state_dict()
        producers["energy-ssl"] = EnergyProducer(ssl, g, setting)
    return producers, state, ids


def restore_learned(mcfg: NagaHanaConfig, state: dict[str, Any], policy: GeneratorPolicy, setting: PhysicalSetting
                    ) -> dict[str, Producer]:
    """Rebuild the fitted learned producers from `prep/generator.state`."""
    g = mcfg.generator
    codec: FieldCodec = state["codec"]
    out: dict[str, Producer] = {}
    fams = tuple(state.get("families", ()))
    if "masked" in state:
        masked = build_masked_model(g, codec)
        masked.load_state_dict(state["masked"])
        if "masked-generative" in fams:
            out["masked-generative"] = MaskedProducer(masked, codec, g, setting, order="maskgit")
        if "autoregressive" in fams:
            out["autoregressive"] = MaskedProducer(masked, codec, g, setting, order="left-to-right")
    if "diffusion" in state:
        diff = TabularDiffusion(g, codec)
        diff.load_state_dict(state["diffusion"])
        out["diffusion"] = DiffusionProducer(diff, g, setting)
    if "energy" in state:
        ssl = EnergySSL(g, codec, policy)
        ssl.load_state_dict(state["energy"])
        out["energy-ssl"] = EnergyProducer(ssl, g, setting)
    return out


def _augment(mcfg: NagaHanaConfig, cfg: TrainingRun, sources: list[PreparedSource], splits: SplitPlan, *,
             learned: dict[str, Producer], fingerprints: FingerprintClasses | None, fitted_ids: list[str]) -> AugmentedData:
    """Draw the variants of the full training split (training/augment.py)."""
    from nagahana.training.randomness import derive_seed

    gate = AcceptanceGate.from_config(mcfg.generator, enabled_proposals=cfg.enabled_proposals)
    producers, weights, _jem = producers_for(mcfg, policy=cfg.generator, learned=learned, fingerprints=fingerprints)
    train = splits.records("full", Role.TRAIN)
    return draw_variants(sources, splits.full, train, gcfg=mcfg.generator, acfg=cfg.augment, producers=producers,
                         weights=weights, gate=gate, seed=derive_seed(cfg.seed, "augment"),
                         n_techniques=mcfg.forecaster.n_techniques, generator_training_ids=fitted_ids)


def _physical_setting(mcfg: NagaHanaConfig) -> PhysicalSetting:
    """The site's physical setting of the Generator's physics gate (the MTU is required, never defaulted)."""
    if mcfg.generator.mtu is None:
        raise InvariantViolation("the Generator's physics gate needs the site MTU (run.mtu); it is never defaulted")
    return PhysicalSetting(mtu=float(mcfg.generator.mtu), link_bps=mcfg.generator.link_bps)


def _digest_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def run_prep(cfg: TrainingRun, mcfg: NagaHanaConfig, *, run_dir: str | Path, is_main: bool = True) -> PrepResult:
    """The data preparation (module docstring). Writes its artefacts under `<run_dir>/prep` on rank 0."""
    if not cfg.data.sources:
        raise InvariantViolation("data preparation needs at least one source (data.sources)")
    out_dir = Path(run_dir) / PREP_DIR
    sources: list[PreparedSource] = []
    analyses: list[dict[str, Any]] = []
    digests: dict[str, str] = {}
    for spec in cfg.data.sources:
        src = ingest_source(spec)
        data, dropped = deduplicate(src.data)
        if dropped:
            src = prepare_source(data)
        file_sha = sha256_file(spec.path)
        a = analyse_source(src, mcfg, quantiles=cfg.prep.quantiles, file_sha256=file_sha, dropped=dropped)
        analyses.append(a)
        digests[spec.path] = a["content_sha256"]
        sources.append(src)
    segments = source_segments(sources, mcfg, cfg.data)
    splits = plan_splits(segments, mcfg, cfg.data)
    roles = {r.value: len(splits.full.ids(r)) for r in Role}
    analysis = {"sources": analyses, "segments": len(segments), "roles_full": roles,
                "roles_pretrain": {r.value: len(splits.pretrain.ids(r)) for r in Role},
                "novelty": {k: v.value for k, v in splits.full.novelty.items()}}
    notes: list[str] = []
    augmented: AugmentedData | None = None
    gen_state: dict[str, Any] | None = None
    if cfg.augment.enabled and cfg.ablation.generator and cfg.augment.target != "none":
        setting = _physical_setting(mcfg)
        with run_mode(RunMode.TRAIN):
            fams, _jem = active_families()
            train = splits.records("full", Role.TRAIN)
            fp = None
            if "tool-fingerprint" in fams:
                fp_windows = [segment_input(sources[r.source], r, mcfg.forecaster.n_techniques)[:2] for r in train]
                fp = FingerprintClasses.build(fp_windows, cfg.generator.fingerprint_fields)
            learned, gen_state, fitted = _fit_learned(mcfg, train, sources, cfg, families=fams, setting=setting)
            augmented = _augment(mcfg, cfg, sources, splits, learned=learned, fingerprints=fp, fitted_ids=fitted)
            if gen_state is not None:
                gen_state["fingerprints"] = fp.to_json() if fp is not None else None
                gen_state["fitted_on"] = list(fitted)
        analysis["augmentation"] = {"target": augmented.plan.target, "quota": augmented.plan.quota,
                                    "accepted": augmented.plan.accepted, "mixture": augmented.plan.mixture(),
                                    "rejections": augmented.plan.rejections, "notes": augmented.plan.notes}
        notes += augmented.plan.notes
    else:
        notes.append("no Generator augmentation (augment.enabled, ablation.generator or target 'none')")
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out_dir / "analysis.json", json.dumps(analysis, indent=1, sort_keys=True, default=_json_default))
        save_state(out_dir / "splits.state", {"full": splits.full, "pretrain": splits.pretrain}, kind="nagahana-splits",
                   meta={"source_digests": digests})
        if augmented is not None:
            save_state(out_dir / "generator.state", gen_state or {}, kind="nagahana-generator")
            save_state(out_dir / "augmentation.state", augmented.plan, kind="nagahana-augmentation-plan")
            if cfg.augment.store_variants:
                save_state(out_dir / "variants.state", [v.updates for v in augmented.variants], kind="nagahana-variants")
    digests["analysis"] = _digest_json(analysis)
    return PrepResult(sources=sources, segments=segments, splits=splits, augmented=augmented, analysis=analysis,
                      digests=digests, generator_state=gen_state, notes=notes)


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def load_prep(cfg: TrainingRun, mcfg: NagaHanaConfig, *, run_dir: str | Path) -> PrepResult:
    """Restore a data preparation in another process: re-ingest, verify digests, regenerate and verify variants."""
    out_dir = Path(run_dir) / PREP_DIR
    if not (out_dir / "splits.state").is_file():
        raise InvariantViolation(f"{out_dir}: no data preparation found (run train --prep first)")
    stored, meta = load_state(out_dir / "splits.state", kind="nagahana-splits")
    digests: dict[str, str] = {}
    sources: list[PreparedSource] = []
    for spec in cfg.data.sources:
        src = ingest_source(spec)
        data, dropped = deduplicate(src.data)
        if dropped:
            src = prepare_source(data)
        d = source_digest(src)
        if meta["source_digests"].get(spec.path) != d:
            raise InvariantViolation(f"{spec.path}: prepared content differs from the one the preparation recorded")
        digests[spec.path] = d
        sources.append(src)
    segments = source_segments(sources, mcfg, cfg.data)
    splits = plan_splits(segments, mcfg, cfg.data)
    if splits.full.role != stored["full"].role or splits.pretrain.role != stored["pretrain"].role:
        raise InvariantViolation("the recomputed split plan differs from the stored one")
    analysis = json.loads((out_dir / "analysis.json").read_text(encoding="utf-8"))
    augmented: AugmentedData | None = None
    gen_state: dict[str, Any] | None = None
    if (out_dir / "augmentation.state").is_file() and cfg.ablation.generator:
        plan, _ = load_state(out_dir / "augmentation.state", kind="nagahana-augmentation-plan")
        gen_state, _ = load_state(out_dir / "generator.state", kind="nagahana-generator")
        setting = _physical_setting(mcfg)
        with run_mode(RunMode.TRAIN):
            learned = restore_learned(mcfg, gen_state, cfg.generator, setting) if gen_state else {}
            fp_json = gen_state.get("fingerprints") if gen_state else None
            fp = FingerprintClasses.from_json(fp_json) if fp_json else None
            augmented = _augment(mcfg, cfg, sources, splits, learned=learned, fingerprints=fp,
                                 fitted_ids=list(gen_state.get("fitted_on", [])) if gen_state else [])
        if not isinstance(plan, AugmentationPlan):
            raise InvariantViolation(f"{out_dir / 'augmentation.state'}: not an augmentation plan")
        verify_replay(plan, augmented)
    return PrepResult(sources=sources, segments=segments, splits=splits, augmented=augmented, analysis=analysis,
                      digests=digests, generator_state=gen_state)


__all__ = ["PREP_DIR", "PrepResult", "analyse_source", "deduplicate", "load_prep", "restore_learned", "run_prep"]
