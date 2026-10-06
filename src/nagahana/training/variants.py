"""Generator variants as training data: segment inputs, label tables, producers, the marginal-energy callable.

Purpose
-------
"Variants are mixed into the real training batches" (architecture section 3.11). For a real training
segment (the unit of splitting and of the carry, data/stream.py) the Generator's `VariantPipeline`
makes label-preserving variants; each accepted variant becomes a source of its own (origin
"generated", `derived_from` = the real segment) with one stream segment, served beside the real ones.
A variant is a stream of its own: its lane is reset at its start, so no carried state mixes real and
generated windows. The balanced planning, leakage guards and provenance of a training run are in
training/augment.py; this module holds the pieces it shares with tools and tests:

- `segment_input`: a real training segment as the Generator's input (rows, per-update labels, the
  manifest row);
- `variant_label_table`: a variant's label table, each row taking the labels of its source row;
- `producers_for`: the producers of the active families (D-14 option in force, AS-27) for a run;
- `make_variant_sources`: a quick draw of `total` variants with the pipeline's own attack-share budget
  (AS-370), for tools and tests; training runs use `augment.draw_variants`;
- `marginal_energy_fn`: E(empty, y) of a `ColumnarUpdates` window for the JEM-style acceptance (AS-421).

Rules (enforced by the Generator and re-checked by augment.py): training only (D-40); sources are real
training segments (AS-367); zero-shot never feeds the Generator and an evaluation loader refuses
generated segments (D-23); the acceptance gate needs the site MTU (`GeneratorConfig.mtu`, never
defaulted).

Assumptions: AS-27, AS-28, AS-361, AS-366, AS-367, AS-370, AS-421.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from nagahana.core.errors import InvariantViolation
from nagahana.data.collate import collate_items
from nagahana.data.labels import LABEL_COLUMNS, technique_slot
from nagahana.data.sampling import WindowRecord
from nagahana.data.windows import PreparedSource, SourceData, build_window, plan_windows, prepare_source
from nagahana.datamodel.columnar import ColumnarUpdates
from nagahana.governance.assumptions import assume
from nagahana.inference.buffer import take_rows, unknown_labels
from nagahana.models.generator.acceptance import AcceptanceGate, EnergyAcceptance
from nagahana.models.generator.config import DETERMINISTIC_FAMILIES, GeneratorPolicy
from nagahana.models.generator.model import GENERATORS, active_families
from nagahana.models.generator.pipeline import Producer, VariantBatch, VariantPipeline
from nagahana.models.generator.variants import UpdateLabels
from nagahana.models.config import NagaHanaConfig
from nagahana.models.nagahana import NagaHana
from nagahana.pipeline.splits import Origin, Sample, Split


@dataclass
class VariantSources:
    """Accepted variants as sources and stream segments, plus the Generator's own report."""

    sources: list[PreparedSource]
    segments: list[WindowRecord]
    batch: VariantBatch


def _model_cfg(model: NagaHana | NagaHanaConfig) -> NagaHanaConfig:
    return model.cfg if isinstance(model, NagaHana) else model


def deterministic_producers(model: NagaHana | NagaHanaConfig, policy: GeneratorPolicy | None = None,
                            fingerprints: Any = None) -> list[Producer]:
    """The deterministic families (observability, signature, topology; tool fingerprints when a table exists)."""
    assume("AS-27", by=__name__)
    g = _model_cfg(model).generator
    pol = policy if policy is not None else GeneratorPolicy()
    out: list[Producer] = []
    out += list(GENERATORS.build("observability-sliding", g, pol))
    out += list(GENERATORS.build("signature-variation", g, pol))
    out += list(GENERATORS.build("topology-variation", g, pol))
    if fingerprints is not None:
        out += list(GENERATORS.build("tool-fingerprint", fingerprints))
    return out


def producers_for(model: NagaHana | NagaHanaConfig, *, policy: GeneratorPolicy, learned: dict[str, Producer],
                  fingerprints: Any = None) -> tuple[list[Producer], list[float], bool]:
    """(producers, sampling weights, JEM acceptance on) of the active families (D-14 option in force).

    learned: family name -> producer of the fitted learned families (training/prep.py fits them).
    """
    fams, jem = active_families()
    producers: list[Producer] = []
    weights: list[float] = []
    for fam in fams:
        if fam in DETERMINISTIC_FAMILIES:
            if fam == "tool-fingerprint":
                if fingerprints is None:
                    continue
                built = list(GENERATORS.build("tool-fingerprint", fingerprints))
            else:
                built = list(GENERATORS.build(fam, _model_cfg(model).generator, policy))
        else:
            if fam not in learned:
                raise InvariantViolation(f"learned family {fam!r} is active (D-14 option in force) but was not fitted")
            built = [learned[fam]]
        w = policy.weight_of(fam)
        # A family's weight is shared by its producers, so every family gets its weight whatever its size.
        producers += built
        weights += [w / len(built)] * len(built)
    if not producers or sum(weights) <= 0:
        raise InvariantViolation("no Generator producer is active with a positive weight")
    return producers, weights, jem


def segment_input(src: PreparedSource, rec: WindowRecord, n_techniques: int) -> tuple[ColumnarUpdates, UpdateLabels, Sample]:
    """A real training segment as the Generator's input: its rows, per-update labels and manifest row."""
    rows = src.order[rec.start:rec.stop]
    cu = take_rows(src.data.updates, rows)
    tech = np.array([technique_slot(str(x), n_techniques) for x in src.technique[rec.start:rec.stop]], dtype=np.int64)
    labels = UpdateLabels(malicious=src.malicious[rec.start:rec.stop].astype(np.float32).copy(),
                          stage=src.stage[rec.start:rec.stop].astype(np.int64).copy(), technique=tech, family=rec.family)
    sample = Sample(id=rec.id, origin=Origin.REAL, split=Split.TRAIN, family=rec.family, network=rec.network)
    return cu, labels, sample


def variant_label_table(src: PreparedSource, rec: WindowRecord, source_rows: np.ndarray, variant: ColumnarUpdates
                        ) -> pd.DataFrame:
    """The label table of a variant: each row takes the labels of its source row (label-preserving)."""
    table = src.data.labels.set_index("seq")
    seqs = src.data.updates.updates["seq"].to_numpy()[src.order[rec.start:rec.stop][source_rows]]
    rows = table.loc[seqs].reset_index(drop=True)
    rows.insert(0, "seq", variant.updates["seq"].to_numpy(dtype=np.int64))
    rows["record"] = variant.updates["record"].to_numpy(dtype=np.int64)
    return rows[list(LABEL_COLUMNS)]


def make_variant_sources(model: NagaHana, sources: Sequence[PreparedSource], segments: Sequence[WindowRecord], *,
                         total: int, seed: int, producers: Sequence[Producer] | None = None,
                         energy: EnergyAcceptance | None = None,
                         enabled_proposals: Collection[str] = ("P-11", "P-23")) -> VariantSources:
    """Variants of real training segments with the pipeline's attack-share budget (module docstring).

    enabled_proposals: the run's proposals; P-11 makes the physics gate reject (AS-28), as in a training run.
    """
    assume("AS-27", by=__name__)
    cfg = model.cfg
    gate = AcceptanceGate.from_config(cfg.generator, energy=energy, enabled_proposals=enabled_proposals)  # ConfigMissing without an MTU
    prods = list(producers) if producers is not None else deterministic_producers(model)
    pipe = VariantPipeline(cfg.generator, prods, gate, seed=seed)
    items = [segment_input(sources[r.source], r, cfg.forecaster.n_techniques) for r in segments]
    batch = pipe.generate_many(items, total=total)
    by_id = {r.id: r for r in segments}
    out_src: list[PreparedSource] = []
    out_seg: list[WindowRecord] = []
    base = len(sources)
    for v in batch.accepted:
        rec = by_id[v.sample.derived_from or ""]
        src = sources[rec.source]
        table = variant_label_table(src, rec, v.source_rows, v.updates)
        prepared = prepare_source(SourceData(v.updates, table, network=rec.network, dataset=src.data.dataset,
                                             origin="generated", derived_from=rec.id))
        idx = base + len(out_src)
        out_src.append(prepared)
        n = len(prepared.order)
        out_seg.append(WindowRecord(id=f"{v.id}:seg", source=idx, start=0, stop=n, network=rec.network,
                                    t_start=float(prepared.time[0]), t_end=float(prepared.time[-1]), family=rec.family,
                                    families=rec.families, origin="generated", derived_from=rec.id))
    return VariantSources(sources=out_src, segments=out_seg, batch=batch)


def marginal_energy_fn(model: NagaHana, *, passes: int, descent_steps: int, max_windows: int) -> Callable[[ColumnarUpdates], float]:
    """E(empty, y) of a `ColumnarUpdates` window for the JEM-style acceptance (AS-421).

    The updates are windowed by the training rule; each of the first `max_windows` windows gets one
    trigger at its last update; perception reads the posterior mean without carry; the value is the mean
    over windows of the mean marginal energy of the active entity tokens.
    """
    assume("AS-421", by=__name__)
    cfg = model.cfg
    dev = next(model.parameters()).device

    def energy(cu: ColumnarUpdates) -> float:
        src = prepare_source(SourceData(cu, unknown_labels(cu), network="generator-acceptance"))
        plan = plan_windows(src, window_updates=cfg.training.window_updates, max_entities=cfg.training.max_entities)
        vals: list[float] = []
        with torch.no_grad():
            for a, b in plan[:max_windows]:
                item = build_window(src, a, b, cfg)
                tau = float(item.update_time[-1])
                latest = np.full((1, len(item.entity_rows)), -1, dtype=np.int64)
                for p in range(len(item.pos_entity)):
                    if item.pos_time[p] <= tau:
                        latest[0, item.pos_entity[p]] = p
                item.trigger_time, item.entity_latest = np.array([tau]), latest
                item.entity_malicious_share = np.full((len(item.entity_rows), 1), np.nan, dtype=np.float32)
                w, _ = collate_items([item], cfg)
                if dev.type != "cpu":
                    from nagahana.training.engine import to_device

                    w = to_device(w, dev)
                _, env = model.perceive(w, sample=False, passes=passes)
                an = model.analyse(env, w, passes=passes, descent_steps=descent_steps)
                e, _ = model.taaft.marginal_energy(an.y, an.token_mask, n_entities=w.entity_mask.shape[1])
                vals.append(float(e.mean()))
        return float(np.mean(vals)) if vals else float("nan")

    return energy


__all__ = ["VariantSources", "deterministic_producers", "make_variant_sources", "marginal_energy_fn", "producers_for",
           "segment_input", "variant_label_table"]
