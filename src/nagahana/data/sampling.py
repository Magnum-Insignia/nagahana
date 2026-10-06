"""Window index, class-balanced sampling and split manifests (build-spec §4b.6–4b.7; D-23, AS-35).

Purpose
-------
1. `index_windows`: plan the windows of every source and describe each one cheaply (time span,
   dominant family, every family it contains, network, origin) without building its tensors.
2. `class_balanced_weights` / `ClassBalancedSampler`: draw windows so that every attack family is
   equally likely (§4b.6 "class-balanced window sampling").
3. `assign_splits`: the split manifest, with the owner's ratios and rules:
   - full training (§4b.7): 60 % train (real + generated), 20 % test (real), 20 % validation;
     zero-shot = real windows of novel families and of held-out networks (AS-35);
   - pretraining (§4b.7): 70 % / 30 % (real);
   - generated windows only in train, derived from real *training* windows (§4b.7; P-23 check);
   - novel and known families evaluated separately (D-23): zero-shot rows are marked known/novel;
   - leave-one-network-out (AS-35, D-16 held): `leave_one_network_out`.
   `SplitManifest.validate()` runs `pipeline.splits.validate` (the decided rules) plus the checks the
   pipeline module cannot express yet (see "Test split" below).

Owner sources, decisions, assumptions: D-23 (splits), D-16 held → AS-35 (zero-shot includes unseen
networks), P-23 (generator no-leakage; its check is run because §4b.7 puts generated data in train
only), new AS-325 … AS-331 (`docs/assumptions/data.md`).

Maths and rules
---------------
Class balance (AS-325). With n_f windows of family f among the candidates and F families,
    w_i = 1 / (F · n_{f(i)})        so  P(family f) = 1/F   and  Σ_i w_i = 1.
(Inverse frequency; the "effective number" re-weighting of Cui et al., CVPR 2019, arXiv:1901.05555,
is an extension point: `power` < 1 interpolates towards uniform, w ∝ n_f^(−power).)

Chronological stratified split (AS-326). Within each stratum (network, family) of the windows that
are neither zero-shot nor generated, windows are ordered by start time and cut by cumulative count:
the first share → train, the next → test, the rest → validation (full mode; pretrain: train, val).
Earlier traffic trains, later traffic evaluates, as a deployed model would meet it; random window
shuffles would let a test window sit between two training windows of the same session (temporal
snooping, Arp et al., USENIX Security 2022, "Dos and Don'ts of Machine Learning in Computer
Security"). `purge` windows at each cut are dropped (AS-327), after the purging idea of
López de Prado, "Advances in Financial Machine Learning" (Wiley 2018, ch. 7).

Zero-shot (AS-35, AS-328). A window is zero-shot if its network is held out or if *any* family in it
(not only the dominant one) is novel; so no update of a novel family ever reaches train/test/val.
Its novelty is "novel" if any of its families is novel or not seen in train/test/val, else "known".

Test split (AS-329). `pipeline.splits.Split` has TRAIN, VAL, ZERO_SHOT and no TEST. Until a TEST
member is added (requested change), test rows are validated as VAL rows: the decided rules apply to
them identically (real data; families count as seen). The extra checks here: test rows are real;
generated rows are train only; zero-shot rows never appear in pretraining.

Extension points: other strata keys (`family_key="subfamily"`), `power`, `purge`.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

from nagahana.core.errors import InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.models.config import NagaHanaConfig
from nagahana.pipeline import splits as psplits

from .windows import PreparedSource, plan_windows


class Role(enum.Enum):
    """The split a window belongs to (the pipeline's Split plus TEST, AS-329)."""

    TRAIN = "train"
    TEST = "test"
    VAL = "val"
    ZERO_SHOT = "zero_shot"
    EXCLUDED = "excluded"          # purged at a cut (AS-327)


@dataclass(frozen=True)
class WindowRecord:
    """One planned window, described without building it."""

    id: str
    source: int                    # index into the source list
    start: int
    stop: int
    network: str
    t_start: float
    t_end: float
    family: str                    # dominant family (windows.py rule)
    families: frozenset[str]       # every labelled family present
    origin: str = "real"           # "real" | "generated"
    derived_from: str | None = None  # generated: the real window id it varies


def _family_of(src: PreparedSource, a: int, b: int, key: str) -> tuple[str, frozenset[str]]:
    """Dominant malicious family and the family set of sorted rows [a, b) (windows.py rule)."""
    mal = src.malicious[a:b]
    fam = src.family[a:b] if key == "family" else _subfamily(src)[a:b]
    known = ~np.isnan(mal)
    counts = pd.Series(fam[mal == 1.0]).value_counts()
    dom = str(counts.index[0]) if len(counts) else ("benign" if (mal == 0.0).any() else "unknown")
    return dom, frozenset(str(f) for f in fam[known])


def _subfamily(src: PreparedSource) -> np.ndarray:
    lab = src.data.labels.set_index("seq").sort_index()
    return lab["subfamily"].to_numpy(dtype=object)[src.order]


def index_windows(sources: Sequence[PreparedSource], cfg: NagaHanaConfig, *, family_key: str = "family") -> list[WindowRecord]:
    """Plan every source's windows and describe them (see `WindowRecord`)."""
    out: list[WindowRecord] = []
    for s_i, src in enumerate(sources):
        plan = plan_windows(src, window_updates=cfg.training.window_updates, max_entities=cfg.training.max_entities)
        for a, b in plan:
            dom, fams = _family_of(src, a, b, family_key)
            out.append(WindowRecord(
                id=f"{src.data.source_id}:{a}", source=s_i, start=a, stop=b, network=src.data.network,
                t_start=float(src.time[a]), t_end=float(src.time[b - 1]), family=dom, families=fams,
                origin=src.data.origin, derived_from=src.data.derived_from and f"{src.data.derived_from}:{a}",
            ))
    return out


# ====================================================================================== class balance
def class_balanced_weights(records: Sequence[WindowRecord], *, power: float = 1.0) -> np.ndarray:
    """w_i ∝ n_{f(i)}^(−power), normalised to sum 1 (power 1: every family equally likely; AS-325)."""
    assume("AS-34", by=__name__)
    if not records:
        return np.zeros(0)
    fam = pd.Series([r.family for r in records])
    counts = fam.map(fam.value_counts()).to_numpy(dtype=np.float64)
    w = counts ** (-power)
    return w / w.sum()


class ClassBalancedSampler(Sampler[int]):
    """Draws `num_samples` window indices with `class_balanced_weights` (with replacement), seeded."""

    def __init__(self, records: Sequence[WindowRecord], *, num_samples: int, seed: int, power: float = 1.0) -> None:
        self.weights = torch.from_numpy(class_balanced_weights(records, power=power))
        self.num_samples = num_samples
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Change the draw per epoch, deterministically."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed + 1_000_003 * self.epoch)
        yield from torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g).tolist()

    def __len__(self) -> int:
        return self.num_samples


# ====================================================================================== splits
@dataclass(frozen=True)
class SplitPolicy:
    """How to split. mode "full" uses `ratios_full` (train, test, val); "pretrain" uses `ratios_pretrain`."""

    mode: str = "full"
    ratios_full: tuple[float, float, float] = (0.6, 0.2, 0.2)
    ratios_pretrain: tuple[float, float] = (0.7, 0.3)
    novel_families: frozenset[str] = frozenset()
    held_out_networks: frozenset[str] = frozenset()
    purge: int = 1

    @classmethod
    def from_config(cls, cfg: NagaHanaConfig, **kw: object) -> SplitPolicy:
        """Ratios from `TrainingConfig` (§4b.7)."""
        return cls(ratios_full=cfg.training.splits_full, ratios_pretrain=cfg.training.splits_pretrain, **kw)  # type: ignore[arg-type]


@dataclass
class SplitManifest:
    """Role (and zero-shot novelty) of every window id."""

    records: dict[str, WindowRecord]
    role: dict[str, Role]
    novelty: dict[str, psplits.Novelty] = field(default_factory=dict)
    mode: str = "full"
    #: the families declared novel by the policy: a row of such a family is always marked novel, so a
    #: novel family that leaked into train/test/val is caught by `pipeline.splits.validate`
    novel_families: frozenset[str] = frozenset()

    def ids(self, role: Role) -> list[str]:
        return [i for i, r in self.role.items() if r is role]

    def samples(self) -> list[psplits.Sample]:
        """Rows for `pipeline.splits.validate` (TEST validated as VAL, AS-329; EXCLUDED left out)."""
        to_split = {Role.TRAIN: psplits.Split.TRAIN, Role.TEST: psplits.Split.TEST, Role.VAL: psplits.Split.VAL,
                    Role.ZERO_SHOT: psplits.Split.ZERO_SHOT}
        out: list[psplits.Sample] = []
        for i, role in self.role.items():
            if role is Role.EXCLUDED:
                continue
            r = self.records[i]
            # a window carries several families; the manifest row uses each one so the rules see all of them
            for k, fam in enumerate(sorted(r.families | {r.family})):
                out.append(psplits.Sample(
                    id=f"{i}#{k}", origin=psplits.Origin(r.origin), split=to_split[role], family=fam, network=r.network,
                    novelty=self._row_novelty(i, fam) if role is Role.ZERO_SHOT else None,
                    derived_from=(f"{r.derived_from}#0" if r.derived_from else None),
                ))
        return out

    def _row_novelty(self, window_id: str, family: str) -> psplits.Novelty:
        if family in self.novel_families or family not in self.seen_families():
            return psplits.Novelty.NOVEL
        return psplits.Novelty.KNOWN

    def seen_families(self) -> set[str]:
        return {f for i, role in self.role.items() if role in (Role.TRAIN, Role.TEST, Role.VAL)
                for f in self.records[i].families | {self.records[i].family}}

    def validate(self) -> None:
        """Decided split rules (pipeline.splits.validate, with the P-23 check) + AS-329 checks."""
        for i, role in self.role.items():
            r = self.records[i]
            if r.origin == "generated" and role not in (Role.TRAIN, Role.EXCLUDED):
                raise InvariantViolation(f"{i}: generated windows belong to train only (§4b.7)")
            if role in (Role.TEST, Role.ZERO_SHOT) and r.origin != "real":
                raise InvariantViolation(f"{i}: {role.value} windows must be real (§4b.7, D-23)")
            if self.mode == "pretrain" and role is Role.TEST:
                raise InvariantViolation(f"{i}: pretraining has no test split (70/30)")
        rows = self.samples()
        # generated rows derive from a real train window: make their derived_from point at a row id
        psplits.validate(rows, enabled_proposals=("generator-no-leakage",))


def _cut(n: int, ratios: Sequence[float]) -> list[int]:
    """Boundaries of contiguous blocks of n items with the given ratios (largest-remainder rounding)."""
    r = np.asarray(ratios, dtype=np.float64)
    if (r < 0).any() or not np.isclose(r.sum(), 1.0):
        raise InvariantViolation(f"split ratios must be ≥ 0 and sum to 1, got {tuple(ratios)}")
    raw = r * n
    counts = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
        counts[k] += 1
    return np.cumsum(counts).tolist()


def assign_splits(records: Sequence[WindowRecord], policy: SplitPolicy) -> SplitManifest:
    """The split manifest for `policy` (module docstring)."""
    for a in ("AS-35",):
        assume(a, by=__name__)
    by_id = {r.id: r for r in records}
    if len(by_id) != len(records):
        raise InvariantViolation("duplicate window ids")
    role: dict[str, Role] = {}
    # ---- zero-shot: held-out networks or any novel family (real only; generated never zero-shot)
    for r in records:
        if r.origin != "real":
            continue
        if r.network in policy.held_out_networks or (r.families | {r.family}) & policy.novel_families:
            role[r.id] = Role.ZERO_SHOT
    # ---- generated windows: train only, and only when their source window is in train (checked later)
    generated = [r for r in records if r.origin == "generated"]
    # ---- chronological stratified split of the rest
    rest = [r for r in records if r.origin == "real" and r.id not in role]
    strata: dict[tuple[str, str], list[WindowRecord]] = {}
    for r in rest:
        strata.setdefault((r.network, r.family), []).append(r)
    if policy.mode == "full":
        ratios: tuple[float, ...] = tuple(policy.ratios_full)
        names: tuple[Role, ...] = (Role.TRAIN, Role.TEST, Role.VAL)
    elif policy.mode == "pretrain":
        ratios = tuple(policy.ratios_pretrain)
        names = (Role.TRAIN, Role.VAL)
    else:
        raise InvariantViolation(f"unknown split mode {policy.mode!r}")
    for group in strata.values():
        group.sort(key=lambda r: (r.t_start, r.id))
        bounds = _cut(len(group), ratios)
        lo = 0
        for k, hi in enumerate(bounds):
            for j in range(lo, hi):
                role[group[j].id] = names[k]
            # purge the first `purge` windows after each internal cut (AS-327)
            if k > 0:
                for j in range(lo, min(lo + policy.purge, hi)):
                    role[group[j].id] = Role.EXCLUDED
            lo = hi
    for g in generated:
        # a variant joins train only if the real window it varies is in train and it holds no novel family
        src = role.get(g.derived_from or "")
        clean = not ((g.families | {g.family}) & policy.novel_families)
        role[g.id] = Role.TRAIN if src is Role.TRAIN and clean else Role.EXCLUDED
    manifest = SplitManifest(records=dict(by_id), role=role, mode=policy.mode, novel_families=policy.novel_families)
    seen = manifest.seen_families()
    for i, ro in role.items():
        if ro is Role.ZERO_SHOT:
            r = by_id[i]
            fams = r.families | {r.family}
            novel = bool(fams & policy.novel_families) or not fams <= seen
            manifest.novelty[i] = psplits.Novelty.NOVEL if novel else psplits.Novelty.KNOWN
    return manifest


def label_limits(manifest: SplitManifest) -> dict[str, float]:
    """AS-334: per record, the epoch time its label look-ahead must not pass.

    For a real record: the start of the next record of the same source (in time) whose split is
    another one (purged records do not count). +inf if there is none. A generated record takes the
    limit of the real record it varies (its labels mirror that source's later labels). Works for
    window records and for segment records (`data.stream`).
    """
    by_src: dict[int, list[WindowRecord]] = {}
    for r in manifest.records.values():
        if r.origin == "real":
            by_src.setdefault(r.source, []).append(r)
    out: dict[str, float] = {}
    for recs in by_src.values():
        recs.sort(key=lambda r: (r.t_start, r.id))
        nxt: dict[Role, float] = {}                       # earliest later start per split, swept backwards
        for r in reversed(recs):
            own = manifest.role.get(r.id, Role.EXCLUDED)
            out[r.id] = min((t for ro, t in nxt.items() if ro not in (own, Role.EXCLUDED)), default=float("inf"))
            nxt[own] = r.t_start
    for r in manifest.records.values():
        if r.origin != "real":
            out[r.id] = out.get(r.derived_from or "", float("-inf"))   # unknown source window: no look-ahead
    return out


def leave_one_network_out(records: Sequence[WindowRecord], policy: SplitPolicy) -> Iterator[tuple[str, SplitManifest]]:
    """One manifest per network, with that network held out as zero-shot (AS-35; D-16 held)."""
    for net in sorted({r.network for r in records}):
        yield net, assign_splits(records, replace(policy, held_out_networks=frozenset({net})))


def records_for(manifest: SplitManifest, roles: Collection[Role]) -> list[WindowRecord]:
    """The records of the given roles, in id order."""
    return [manifest.records[i] for i in sorted(manifest.role) if manifest.role[i] in roles]


__all__ = [
    "ClassBalancedSampler", "Role", "SplitManifest", "SplitPolicy", "WindowRecord", "assign_splits",
    "class_balanced_weights", "index_windows", "label_limits", "leave_one_network_out", "records_for",
]
