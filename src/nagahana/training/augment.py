"""Class- and family-balanced augmentation of the training split with the Generator (D-23, D-40, P-23).

Purpose
-------
Training reads real training segments and accepted Generator variants of them (D-23); validation,
test and zero-shot read real data only (AS-575). This module plans and draws the variants:

1. Target mixture (AS-582). Over the real training segments, counted by dominant family (AS-331),
   the target shares are, by default ("balanced"), one half for benign and one half for attacks, the
   attack half split equally over the attack families present: the classes are balanced and so are the
   families within the attack class. "uniform-families" gives every family (benign included) an equal
   share; "explicit" takes the configured shares; "none" draws no variant. Segments of unknown family
   are neither sources nor counted.
2. Budget (AS-582). With N real segments, a budget of T = round(generated_to_real * N) variants and
   M = N + T, the per-family variant counts g_f minimise sum_f (n_f + g_f - pi_f M)^2 subject to
   g_f >= 0 and sum_f g_f = T. The solution is water-filling, g_f = max(0, pi_f M - n_f - theta), with
   theta found by bisection so the counts sum to T (theta = 0 when no family exceeds its target, and the
   target is then met exactly); the counts are made integers by the largest-remainder method. A family's
   count is spread equally over its segments (largest remainder again). When the budget cannot reach the
   target (a family already above its share), the result is the closest mixture and the report says so.
3. Drawing. For every source segment, `VariantPipeline.generate` draws its quota from the active
   families (D-14 option in force, AS-27) through the acceptance gate (hard limits always; Phi_phys <= tau
   when P-11 is enabled, AS-28; no energy screen before TAAFT is trained, AS-583). Rejections are made
   up in further rounds over the family's segments (at most `max_rounds`), each round with its own seed.
   `reserve_fraction` more variants per family are drawn for the stage-3 energy re-screen.
4. Leakage guards (P-23 via AS-367; AS-575, AS-584), asserted on every run:
   - every source segment is a real segment of the training role of the split plan;
   - no source holds a family that is novel (zero-shot) or that appears outside the training role only;
   - every row of every variant derives from a row of its own source segment (`derived_from_seq`), and
     every entity a variant row references is an entity of its source segment's rows;
   - the learned families were fitted on training segments only (`pipeline.splits.validate`);
   - the variants join the manifest in the training role and the whole manifest validates.
5. Provenance (AS-369, AS-596). Every variant carries origin "generated", `derived_from` (its real
   segment), the producer and its parameters, the acceptance report and a SHA-256 digest of its
   content; the plan (`AugmentationPlan`) records them, and a later stage regenerates the variants from
   the plan and the stored Generator state and verifies every digest (deterministic under the seed:
   each segment's draws come from (seed, round, SHA-256 of the segment id)).

Decisions: D-14 (held), D-23, D-40. Proposals: P-11, P-23. Assumptions: AS-27, AS-28, AS-325, AS-331,
AS-367, AS-369, AS-575, AS-582 ... AS-584, AS-592, AS-594, AS-596.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.core.modes import RunMode, require_mode
from nagahana.data.sampling import Role, SplitManifest, WindowRecord
from nagahana.data.windows import PreparedSource, SourceData, prepare_source
from nagahana.datamodel.columnar import ColumnarUpdates
from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.acceptance import AcceptanceGate, EnergyAcceptance
from nagahana.models.generator.pipeline import Producer, Variant, VariantBatch, VariantPipeline
from nagahana.models.generator.variants import entity_columns
from nagahana.pipeline import splits as psplits
from nagahana.training.assumptions import use
from nagahana.training.config import AugmentConfig
from nagahana.training.variants import segment_input, variant_label_table

BENIGN = "benign"
UNKNOWN = "unknown"


def target_mixture(counts: Mapping[str, int], cfg: AugmentConfig) -> dict[str, float]:
    """Target shares over the families present (module docstring, step 1)."""
    use("AS-582", by=__name__)
    fams = sorted(f for f, n in counts.items() if n > 0 and f != UNKNOWN)
    if cfg.target == "none" or not fams:
        return {}
    if cfg.target == "uniform-families":
        return {f: 1.0 / len(fams) for f in fams}
    if cfg.target == "explicit":
        mix = dict(cfg.mixture)
        missing = [f for f in mix if f not in fams and mix[f] > 0]
        if missing:
            raise InvariantViolation(f"explicit mixture names families without a training segment: {missing}")
        return {f: float(mix.get(f, 0.0)) for f in fams}
    attacks = [f for f in fams if f != BENIGN]
    if BENIGN not in fams:
        return {f: 1.0 / len(attacks) for f in attacks}
    if not attacks:
        return {BENIGN: 1.0}
    return {BENIGN: 0.5} | {f: 0.5 / len(attacks) for f in attacks}


def largest_remainder(quotas: Mapping[str, float], total: int) -> dict[str, int]:
    """Integers summing to `total`, each the floor of its quota plus one for the largest remainders (Hamilton)."""
    keys = sorted(quotas)
    base = {k: int(math.floor(quotas[k])) for k in keys}
    rest = total - sum(base.values())
    order = sorted(keys, key=lambda k: (-(quotas[k] - base[k]), k))
    for k in order[:max(0, rest)]:
        base[k] += 1
    return base


def allocate(counts: Mapping[str, int], target: Mapping[str, float], total: int) -> dict[str, int]:
    """Water-filling variant counts per family (module docstring, step 2).

    With deficits d_f = pi_f M - n_f (M = N + T), sum_f d_f = T because the target covers every counted
    family; g_f = max(0, d_f - theta) with theta = 0 when no deficit is negative, else the theta in
    [0, max d] at which the counts sum to T (bisection to machine precision), then integers by the
    largest-remainder method.
    """
    if total < 0:
        raise InvariantViolation("the variant budget must be >= 0")
    fams = sorted(target)
    if not fams or total == 0:
        return dict.fromkeys(fams, 0)
    m = sum(counts.get(f, 0) for f in fams) + total
    deficit = {f: target[f] * m - counts.get(f, 0) for f in fams}

    def spent(theta: float) -> float:
        return sum(max(0.0, deficit[f] - theta) for f in fams)

    theta = 0.0
    if spent(0.0) > total:
        lo, hi = 0.0, max(deficit.values())
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if spent(mid) > total:
                lo = mid
            else:
                hi = mid
        theta = hi
    quotas = {f: max(0.0, deficit[f] - theta) for f in fams}
    s = sum(quotas.values())
    if s <= 0:
        raise InvariantViolation("no family is below its target share: nothing to allocate")
    return largest_remainder({f: q * total / s for f, q in quotas.items()}, total)


def _segment_seed(seed: int, segment_id: str, round_no: int) -> int:
    digest = int.from_bytes(hashlib.sha256(segment_id.encode("utf-8")).digest()[:8], "big")
    return int((seed * 1_000_003 + round_no * 7_919 + digest) % (2**62))


def variant_digest(cu: ColumnarUpdates) -> str:
    """SHA-256 of a variant's content: values, statuses, times, entities, provenance columns."""
    h = hashlib.sha256()
    for arr in (cu.values, cu.status):
        h.update(np.ascontiguousarray(arr).tobytes())
    for col in ("event_time", "derived_from_seq", *entity_columns(cu.updates)):
        if col in cu.updates:
            h.update(np.ascontiguousarray(cu.updates[col].to_numpy()).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class PlannedVariant:
    """One accepted variant in the plan (AS-596)."""

    variant_id: str
    source_segment: str
    family: str
    producer: str
    round_no: int
    rows: int
    digest: str
    phi_phys: float
    reserve: bool


@dataclass
class AugmentationPlan:
    """What was drawn, why, and from what (module docstring, step 5)."""

    seed: int
    target: dict[str, float]
    real_counts: dict[str, int]
    quota: dict[str, int]
    accepted: dict[str, int]
    variants: list[PlannedVariant] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    generator_training_ids: tuple[str, ...] = ()

    def mixture(self, *, include_reserve: bool = False) -> dict[str, float]:
        """Realised shares of real plus accepted variants per family."""
        tot = dict(self.real_counts)
        for v in self.variants:
            if include_reserve or not v.reserve:
                tot[v.family] = tot.get(v.family, 0) + 1
        n = sum(n for f, n in tot.items() if f != UNKNOWN)
        return {f: c / n for f, c in sorted(tot.items()) if f != UNKNOWN and n > 0}


@dataclass
class AugmentedData:
    """Variant sources and segments (indices continue the real sources), with the plan."""

    sources: list[PreparedSource]
    segments: list[WindowRecord]
    plan: AugmentationPlan
    variants: list[Variant]


def guard_sources(manifest: SplitManifest, segments: Sequence[WindowRecord]) -> None:
    """Leakage guards on the Generator's inputs (module docstring, step 4)."""
    use("AS-575", by=__name__)
    train_fams = {f for i, r in manifest.role.items() if r is Role.TRAIN for f in manifest.records[i].families | {manifest.records[i].family}}
    other_fams = {f for i, r in manifest.role.items() if r in (Role.TEST, Role.VAL, Role.ZERO_SHOT)
                  for f in manifest.records[i].families | {manifest.records[i].family}}
    held_only = other_fams - train_fams
    for r in segments:
        if r.origin != "real":
            raise InvariantViolation(f"{r.id}: the Generator reads real segments only (no variants of variants)")
        if manifest.role.get(r.id) is not Role.TRAIN:
            raise InvariantViolation(f"{r.id}: the Generator reads training segments only (P-23, AS-367)")
        fams = r.families | {r.family}
        if fams & set(manifest.novel_families):
            raise InvariantViolation(f"{r.id}: holds a novel (zero-shot) family {sorted(fams & set(manifest.novel_families))}")
        if fams & held_only:
            raise InvariantViolation(f"{r.id}: holds a family seen only outside training {sorted(fams & held_only)}")


def guard_variant(v: Variant, src: PreparedSource, rec: WindowRecord) -> None:
    """Every variant row derives from its source segment's rows; every entity is one of theirs (AS-584)."""
    rows = src.order[rec.start:rec.stop]
    seqs = set(src.data.updates.updates["seq"].to_numpy()[rows].tolist())
    derived = v.updates.updates["derived_from_seq"].to_numpy()
    # derived_from_seq indexes the segment input (renumbered 0..n-1 by take_rows); map back to source seqs.
    if len(derived) and (derived.min() < 0 or derived.max() >= len(rows)):
        raise InvariantViolation(f"{v.id}: a row does not derive from its source segment")
    src_seq = src.data.updates.updates["seq"].to_numpy()[rows][derived]
    if not set(src_seq.tolist()) <= seqs:
        raise InvariantViolation(f"{v.id}: a row derives from outside its source segment")
    ents_v = set(np.unique(v.updates.updates[entity_columns(v.updates.updates)].to_numpy()).tolist()) - {-1}
    ents_s = set(np.unique(src.ents[rec.start:rec.stop]).tolist()) - {-1}
    if not ents_v <= ents_s:
        raise InvariantViolation(f"{v.id}: references entities outside its source segment {sorted(ents_v - ents_s)[:5]}")


def _family_segments(records: Sequence[WindowRecord]) -> dict[str, list[WindowRecord]]:
    out: dict[str, list[WindowRecord]] = {}
    for r in sorted(records, key=lambda r: (r.t_start, r.id)):
        out.setdefault(r.family, []).append(r)
    return out


def draw_variants(sources: Sequence[PreparedSource], manifest: SplitManifest, train: Sequence[WindowRecord], *,
                  gcfg: GeneratorConfig, acfg: AugmentConfig, producers: Sequence[Producer],
                  weights: Sequence[float] | None, gate: AcceptanceGate, seed: int, n_techniques: int,
                  generator_training_ids: Sequence[str] = (), energy: EnergyAcceptance | None = None) -> AugmentedData:
    """Plan and draw the variants of the training split (module docstring). Training mode only."""
    require_mode(RunMode.TRAIN, component="Generator")
    use("AS-592", by=__name__)
    use("AS-596", by=__name__)
    real = [r for r in train if r.origin == "real" and r.family != UNKNOWN]
    guard_sources(manifest, real)
    counts = {f: len(rs) for f, rs in _family_segments(real).items()}
    target = target_mixture(counts, acfg)
    total = int(round(acfg.generated_to_real * len(real))) if acfg.enabled else 0
    quota = allocate(counts, target, total) if target else {}
    plan = AugmentationPlan(seed=seed, target=target, real_counts=counts, quota=quota, accepted=dict.fromkeys(quota, 0),
                            generator_training_ids=tuple(generator_training_ids))
    if any(quota.get(f, 0) > 0 and target[f] * (len(real) + total) < counts.get(f, 0) for f in quota):
        plan.notes.append("the budget cannot reach the target mixture exactly: a family is above its share")
    by_family = _family_segments(real)
    accepted: list[tuple[Variant, WindowRecord, int, bool]] = []
    rejections: dict[str, int] = {}
    for fam in sorted(quota):
        want = quota[fam]
        reserve = int(math.ceil(acfg.reserve_fraction * want)) if want > 0 else 0
        need = want + reserve
        segs = by_family.get(fam, [])
        got = 0
        for round_no in range(acfg.max_rounds):
            if got >= need or not segs:
                break
            share = largest_remainder({r.id: (need - got) / len(segs) for r in segs}, need - got)
            for r in segs:
                k = share[r.id]
                if k <= 0:
                    continue
                cu, labels, sample = segment_input(sources[r.source], r, n_techniques)
                rpipe = VariantPipeline(gcfg, list(producers), gate, seed=_segment_seed(seed, r.id, round_no), weights=weights)
                batch: VariantBatch = rpipe.generate(cu, labels, sample, n=k)
                for rej in batch.rejected:
                    for reason in rej.reasons:
                        key = ":".join(reason.split(":")[:2])
                        rejections[key] = rejections.get(key, 0) + 1
                for v in batch.accepted:
                    v.id = f"{v.id}~r{round_no}"
                    v.updates.source_id = f"{v.updates.source_id}~r{round_no}"
                    v.sample = psplits.Sample(id=v.id, origin=psplits.Origin.GENERATED, split=psplits.Split.TRAIN,
                                              family=v.sample.family, network=v.sample.network, derived_from=r.id)
                    accepted.append((v, r, round_no, got >= want))
                    got += 1
                    if got >= need:
                        break
                if got >= need:
                    break
        plan.accepted[fam] = min(got, want)
        if got < want:
            plan.notes.append(f"family {fam!r}: {got} of {want} variants accepted after {acfg.max_rounds} rounds")
    plan.rejections = rejections
    out_src: list[PreparedSource] = []
    out_seg: list[WindowRecord] = []
    variants: list[Variant] = []
    base = len(sources)
    for v, rec, round_no, is_reserve in accepted:
        src = sources[rec.source]
        guard_variant(v, src, rec)
        table = variant_label_table(src, rec, v.source_rows, v.updates)
        prepared = prepare_source(SourceData(v.updates, table, network=rec.network, dataset=src.data.dataset,
                                             origin="generated", derived_from=rec.id))
        idx = base + len(out_src)
        out_src.append(prepared)
        n = len(prepared.order)
        out_seg.append(WindowRecord(id=f"{v.id}:seg", source=idx, start=0, stop=n, network=rec.network,
                                    t_start=float(prepared.time[0]), t_end=float(prepared.time[-1]), family=rec.family,
                                    families=rec.families, origin="generated", derived_from=rec.id))
        variants.append(v)
        plan.variants.append(PlannedVariant(variant_id=v.id, source_segment=rec.id, family=rec.family, producer=v.producer,
                                            round_no=round_no, rows=len(v.updates), digest=variant_digest(v.updates),
                                            phi_phys=float(v.report.phi_phys), reserve=is_reserve))
    # The variants join the manifest in the training role; the whole manifest must validate (P-23, AS-575).
    joined = SplitManifest(records={**manifest.records, **{s.id: s for s in out_seg}},
                           role={**manifest.role, **dict.fromkeys((s.id for s in out_seg), Role.TRAIN)},
                           novelty=dict(manifest.novelty), mode=manifest.mode, novel_families=manifest.novel_families)
    joined.validate()
    if generator_training_ids:
        rows = [psplits.Sample(id=i, origin=psplits.Origin.REAL, split=psplits.Split(manifest.role[i].value),
                               family=manifest.records[i].family, network=manifest.records[i].network)
                for i in generator_training_ids]
        psplits.validate(rows, generator_training_ids=generator_training_ids)
    return AugmentedData(sources=out_src, segments=out_seg, plan=plan, variants=variants)


def select_pool(data: AugmentedData, *, keep: Callable[[Variant], bool] | None = None) -> AugmentedData:
    """The variant pool of a stage: per family the first `quota` variants (plan order) that `keep` accepts.

    Stages 1 and 2 keep every variant up to the quota (no energy screen yet); stage 3 passes the
    JEM-style energy acceptance (AS-583), so reserve variants replace screened-out ones.
    """
    quota = dict(data.plan.quota)
    taken: dict[str, int] = dict.fromkeys(quota, 0)
    sources, segments, variants = [], [], []
    for src, seg, v, pv in zip(data.sources, data.segments, data.variants, data.plan.variants, strict=True):
        if taken.get(pv.family, 0) >= quota.get(pv.family, 0):
            continue
        if keep is not None and not keep(v):
            continue
        taken[pv.family] = taken.get(pv.family, 0) + 1
        sources.append(src)
        segments.append(seg)
        variants.append(v)
    return AugmentedData(sources=sources, segments=segments, plan=data.plan, variants=variants)


def reindex(data: AugmentedData, base: int) -> AugmentedData:
    """Variant segments re-pointed at sources numbered from `base` (after the real sources)."""
    import dataclasses

    segs = [dataclasses.replace(s, source=base + i) for i, s in enumerate(data.segments)]
    return AugmentedData(sources=list(data.sources), segments=segs, plan=data.plan, variants=list(data.variants))


def verify_replay(plan: AugmentationPlan, data: AugmentedData) -> None:
    """A regenerated pool must equal the plan variant for variant (ids, sources and digests; AS-596)."""
    if len(data.plan.variants) != len(plan.variants):
        raise InvariantViolation(f"replay drew {len(data.plan.variants)} variants, the plan holds {len(plan.variants)}")
    for a, b in zip(plan.variants, data.plan.variants, strict=True):
        if (a.variant_id, a.source_segment, a.digest) != (b.variant_id, b.source_segment, b.digest):
            raise InvariantViolation(f"replay of {a.variant_id} differs from the plan (digest {a.digest[:12]} vs {b.digest[:12]})")


def plan_frame(plan: AugmentationPlan) -> pd.DataFrame:
    """The plan's variants as a table (audit)."""
    return pd.DataFrame([v.__dict__ for v in plan.variants])


__all__ = ["AugmentationPlan", "AugmentedData", "PlannedVariant", "allocate", "draw_variants", "guard_sources",
           "guard_variant", "largest_remainder", "plan_frame", "reindex", "select_pool", "target_mixture", "variant_digest",
           "verify_replay"]
