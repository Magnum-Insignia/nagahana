"""Evaluation protocols of the reproduced papers, and the runner that applies them.

A protocol turns one frame into a sequence of (train, validation, test) row-index splits:

    RandomHoldout         random test fraction, optionally stratified, repeated R times
                          (Sarhan et al. 2022: 70/30 over five splits; Cantone et al. 2024 and Neto et
                          al. 2023: 80/20)
    StratifiedKFold       K folds, stratified, repeated R times (Leevy et al. 2021: 10 x 5-fold)
    LastRows              the last fraction of rows in time order is the evaluation set
                          (FlowTransformer's 90/10 split, read as a temporal hold-out, AS-544)
    GroupHoldout          train on some groups, test on held-out ones; leave-one-group-out when no test
                          group is named (Ongun et al. 2019: train on two CTU-13 scenarios, test on the
                          third; NagaHana protocol P3)
    CrossDataset          train on one dataset, test on another (Cantone et al. 2024; protocol P2)
    TimeCutoff            train before a time (or before the first positive, EULER), test after
    DisjointGroups        random disjoint groups for train / validation / test (Tiresias: disjoint
                          machines, 80/10/10)
    GivenSplit            the frame's own `split` column (NagaHana's train / val / test / zero_shot)

`run_protocol` fits a fresh baseline per split with a seed derived from the run seed and the split
index, predicts the test rows, scores them with `metrics.outputs_metrics` and summarises every metric by
mean, standard deviation, minimum and maximum. Where a paper reports "the best repeat"
(FlowTransformer), `ProtocolRun.best` selects it, and the mean is reported beside it so the optimism of
the selection stays visible (AS-535).
"""

from __future__ import annotations

import abc
import dataclasses
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from nagahana.baselines.published.base import BaselineConfig, PublishedBaseline, derive_seed
from nagahana.baselines.published.frames import epoch_seconds, take_rows
from nagahana.baselines.published.metrics import mean_and_spread, outputs_metrics
from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.predictions import ModelOutputs


@dataclass(frozen=True)
class Split:
    """Integer row positions of one split; `validation` may be empty."""

    name: str
    train: np.ndarray
    test: np.ndarray
    validation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        a, b, c = (np.asarray(x, dtype=np.int64) for x in (self.train, self.test, self.validation))
        if np.intersect1d(a, b).size or np.intersect1d(a, c).size or np.intersect1d(b, c).size:
            raise InvariantViolation(f"split {self.name}: train, validation and test overlap")
        if self.train.size == 0 or self.test.size == 0:
            raise InvariantViolation(f"split {self.name}: train and test must be non-empty")


class SplitProtocol(abc.ABC):
    """A rule that yields splits of a frame."""

    @abc.abstractmethod
    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        """Yield the splits of `frame`; random choices depend only on `seed`."""

    @abc.abstractmethod
    def describe(self) -> str:
        """One line, as a paper would state it."""


def _strata(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    # Integer stratum per row (0 everywhere without a stratification column).
    if column is None:
        return np.zeros(len(frame), dtype=np.int64)
    if column not in frame.columns:
        raise InvariantViolation(f"stratification column {column!r} is absent")
    codes, _ = pd.factorize(frame[column].astype(str), sort=True)
    return codes.astype(np.int64)


def stratified_holdout(strata: np.ndarray, fraction: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split positions into (rest, held) with round(fraction * n_s) held rows per stratum s.

    Every stratum with at least two rows keeps at least one row on each side, so a rare class is never
    left out of training or testing entirely.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    held: list[np.ndarray] = []
    rest: list[np.ndarray] = []
    for s in np.unique(strata):
        idx = np.nonzero(strata == s)[0]
        idx = idx[rng.permutation(idx.size)]
        k = int(round(fraction * idx.size))
        if idx.size >= 2:
            k = min(max(k, 1), idx.size - 1)
        held.append(idx[:k])
        rest.append(idx[k:])
    return np.sort(np.concatenate(rest)), np.sort(np.concatenate(held))


@dataclass(frozen=True)
class RandomHoldout(SplitProtocol):
    """Random hold-out of `test_fraction`, repeated `repeats` times with independent draws."""

    test_fraction: float
    repeats: int = 1
    stratify: str | None = None
    validation_fraction: float = 0.0

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        strata = _strata(frame, self.stratify)
        for r in range(self.repeats):
            rng = np.random.default_rng(derive_seed(seed, "holdout", r))
            train, test = stratified_holdout(strata, self.test_fraction, rng)
            val = np.zeros(0, dtype=np.int64)
            if self.validation_fraction > 0.0:
                keep, v = stratified_holdout(strata[train], self.validation_fraction, rng)
                train, val = train[keep], train[v]
            yield Split(f"repeat-{r}", train, test, val)

    def describe(self) -> str:
        strat = f", stratified on {self.stratify}" if self.stratify else ""
        return f"random {1 - self.test_fraction:.0%}/{self.test_fraction:.0%} hold-out{strat}, {self.repeats} repeat(s)"


@dataclass(frozen=True)
class StratifiedKFold(SplitProtocol):
    """K-fold cross-validation, stratified, repeated with reshuffling (Leevy et al.: 10 x 5-fold)."""

    n_splits: int = 5
    repeats: int = 1
    stratify: str | None = None

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        strata = _strata(frame, self.stratify)
        n = len(frame)
        for r in range(self.repeats):
            rng = np.random.default_rng(derive_seed(seed, "kfold", r))
            fold = np.empty(n, dtype=np.int64)
            offset = 0
            for s in np.unique(strata):
                idx = np.nonzero(strata == s)[0]
                idx = idx[rng.permutation(idx.size)]
                # Deal the stratum's rows round-robin, continuing where the previous stratum stopped,
                # so fold sizes differ by at most one overall.
                fold[idx] = (np.arange(idx.size) + offset) % self.n_splits
                offset = (offset + idx.size) % self.n_splits
            for k in range(self.n_splits):
                test = np.nonzero(fold == k)[0]
                train = np.nonzero(fold != k)[0]
                yield Split(f"repeat-{r}-fold-{k}", train, test)

    def describe(self) -> str:
        strat = "stratified " if self.stratify else ""
        return f"{self.repeats} x {strat}{self.n_splits}-fold cross-validation"


@dataclass(frozen=True)
class LastRows(SplitProtocol):
    """The last `eval_fraction` of rows in time order is the evaluation set; validation precedes it."""

    eval_fraction: float
    time_column: str | None = None
    validation_fraction: float = 0.0

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        n = len(frame)
        # Without the time column, file order is the time order (flow exports are written as flows end).
        timed = self.time_column is not None and self.time_column in frame.columns
        order = np.argsort(epoch_seconds(frame[self.time_column]), kind="stable") if timed else np.arange(n)
        n_eval = max(1, int(round(self.eval_fraction * n)))
        n_val = int(round(self.validation_fraction * (n - n_eval)))
        train = np.sort(order[: n - n_eval - n_val])
        val = np.sort(order[n - n_eval - n_val: n - n_eval])
        test = np.sort(order[n - n_eval:])
        yield Split("last-rows", train, test, val)

    def describe(self) -> str:
        return f"first {1 - self.eval_fraction:.0%} of rows in time order train, last {self.eval_fraction:.0%} evaluate"


@dataclass(frozen=True)
class GroupHoldout(SplitProtocol):
    """Hold out whole groups (scenarios, captures, networks); each group in turn when none is named."""

    group_column: str
    test_groups: tuple[str, ...] = ()
    train_groups: tuple[str, ...] = ()

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        if self.group_column not in frame.columns:
            raise InvariantViolation(f"group column {self.group_column!r} is absent")
        groups = frame[self.group_column].astype(str).to_numpy()
        names = sorted(set(groups.tolist()))
        tests = [(g,) for g in names] if not self.test_groups else [tuple(self.test_groups)]
        for held in tests:
            is_test = np.isin(groups, held)
            is_train = ~is_test if not self.train_groups else np.isin(groups, self.train_groups)
            yield Split(f"held-out-{'+'.join(held)}", np.nonzero(is_train & ~is_test)[0], np.nonzero(is_test)[0])

    def describe(self) -> str:
        if self.test_groups:
            return f"train on {self.group_column} {self.train_groups or 'all others'}, test on {self.test_groups}"
        return f"leave one {self.group_column} out"


@dataclass(frozen=True)
class CrossDataset(SplitProtocol):
    """Train on the whole of one dataset, test on the whole of another (column `dataset`)."""

    train_dataset: str
    test_dataset: str
    dataset_column: str = "dataset"

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        if self.dataset_column not in frame.columns:
            raise InvariantViolation(f"dataset column {self.dataset_column!r} is absent")
        d = frame[self.dataset_column].astype(str).to_numpy()
        yield Split(f"{self.train_dataset}->{self.test_dataset}", np.nonzero(d == self.train_dataset)[0],
                    np.nonzero(d == self.test_dataset)[0])

    def describe(self) -> str:
        return f"train on {self.train_dataset}, test on {self.test_dataset}"


@dataclass(frozen=True)
class TimeCutoff(SplitProtocol):
    """Train on rows before a cutoff time, test on the rest.

    With `before_first_positive`, the cutoff is the time of the first row whose label column is 1 (EULER
    trains on all snapshots before the first anomalous edge). With `align_seconds` > 0 the cutoff is moved
    back to the start of its snapshot, t0 + floor((cut - t0) / align) * align with t0 the frame's first
    time, so that no snapshot is split between training and test. `validation_fraction` of the training
    rows (the latest ones) form the validation set.
    """

    time_column: str
    cutoff: float | None = None
    before_first_positive: str | None = None
    validation_fraction: float = 0.0
    align_seconds: float = 0.0

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        t = epoch_seconds(frame[self.time_column])
        cut = self.cutoff
        if cut is None:
            if self.before_first_positive is None:
                raise ValueError("TimeCutoff needs a cutoff or before_first_positive")
            pos = frame[self.before_first_positive].to_numpy() == 1
            if not pos.any():
                raise InvariantViolation("no positive row to place the cutoff before")
            cut = float(t[pos].min())
        if self.align_seconds > 0:
            t0 = float(t.min())
            cut = t0 + np.floor((cut - t0) / self.align_seconds) * self.align_seconds
        train = np.nonzero(t < cut)[0]
        test = np.nonzero(t >= cut)[0]
        val = np.zeros(0, dtype=np.int64)
        if self.validation_fraction > 0.0 and train.size:
            order = train[np.argsort(t[train], kind="stable")]
            n_val = max(1, int(round(self.validation_fraction * order.size)))
            train, val = np.sort(order[:-n_val]), np.sort(order[-n_val:])
        yield Split("time-cutoff", train, test, val)

    def describe(self) -> str:
        what = f"the first positive {self.before_first_positive}" if self.before_first_positive else f"t = {self.cutoff}"
        return f"train before {what}, test from then on"


@dataclass(frozen=True)
class DisjointGroups(SplitProtocol):
    """Random disjoint groups (for example machines) for train / validation / test fractions."""

    group_column: str
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        if abs(sum(self.fractions) - 1.0) > 1e-9 or min(self.fractions) < 0:
            raise ValueError("fractions must be non-negative and sum to 1")
        groups = frame[self.group_column].astype(str).to_numpy()
        names = np.asarray(sorted(set(groups.tolist())))
        rng = np.random.default_rng(derive_seed(seed, "groups"))
        names = names[rng.permutation(names.size)]
        n_train = int(round(self.fractions[0] * names.size))
        n_val = int(round(self.fractions[1] * names.size))
        g_train, g_val, g_test = names[:n_train], names[n_train:n_train + n_val], names[n_train + n_val:]
        yield Split("disjoint-groups", np.nonzero(np.isin(groups, g_train))[0], np.nonzero(np.isin(groups, g_test))[0],
                    np.nonzero(np.isin(groups, g_val))[0])

    def describe(self) -> str:
        f = self.fractions
        return f"disjoint {self.group_column} groups, {f[0]:.0%}/{f[1]:.0%}/{f[2]:.0%}"


@dataclass(frozen=True)
class GivenSplit(SplitProtocol):
    """The frame's own split column (NagaHana's protocols assign it; D-23)."""

    split_column: str = "split"
    train: tuple[str, ...] = ("train",)
    validation: tuple[str, ...] = ("val",)
    test: tuple[str, ...] = ("test", "zero_shot")

    def splits(self, frame: pd.DataFrame, seed: int) -> Iterator[Split]:
        s = frame[self.split_column].astype(str).to_numpy()
        yield Split("given", np.nonzero(np.isin(s, self.train))[0], np.nonzero(np.isin(s, self.test))[0],
                    np.nonzero(np.isin(s, self.validation))[0])

    def describe(self) -> str:
        return f"given split column {self.split_column!r}"


def stratified_choice(strata: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Exactly k sorted positions drawn without replacement, allocated to strata in proportion to their sizes.

    Quotas are k * n_s / n, floored, with the remaining rows given to the largest fractional parts (ties
    to the earlier stratum), so the total is exactly k and no stratum exceeds its size.
    """
    strata = np.asarray(strata)
    n = strata.shape[0]
    if not 0 <= k <= n:
        raise ValueError(f"cannot draw {k} of {n} rows")
    values, counts = np.unique(strata, return_counts=True)
    quota = counts / counts.sum() * k
    base = np.floor(quota).astype(np.int64)
    order = np.argsort(-(quota - base), kind="stable")
    base[order[: k - int(base.sum())]] += 1
    picks = [rng.choice(np.nonzero(strata == v)[0], size=int(q), replace=False) for v, q in zip(values, base, strict=True) if q > 0]
    return np.sort(np.concatenate(picks)) if picks else np.zeros(0, dtype=np.int64)


def stratified_sample(frame: pd.DataFrame, n: int, *, stratify: str | None, seed: int) -> pd.DataFrame:
    """A stratified random sample of exactly n rows (Layeghy and Portmann: 1,000,000-flow stratified samples)."""
    if n >= len(frame):
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(derive_seed(seed, "sample"))
    return take_rows(frame, stratified_choice(_strata(frame, stratify), n, rng))


@dataclass
class ProtocolRun:
    """Outputs and metrics of one protocol run, per split and summarised."""

    protocol: str
    split_names: list[str]
    outputs: list[ModelOutputs]
    metrics: list[dict[str, float]]
    summary: dict[str, dict[str, float]]
    fit_reports: list[dict[str, Any]]

    def best(self, metric: str = "f1") -> tuple[int, dict[str, float]]:
        """Index and metrics of the split with the highest `metric` (the "best repeat" of a paper)."""
        values = [m.get(metric, math.nan) for m in self.metrics]
        finite = [i for i, v in enumerate(values) if not math.isnan(v)]
        if not finite:
            raise InvariantViolation(f"no split has a finite {metric}")
        i = max(finite, key=lambda j: values[j])
        return i, self.metrics[i]


def run_protocol(
    baseline: type[PublishedBaseline],
    config: BaselineConfig,
    frame: pd.DataFrame,
    protocol: SplitProtocol,
    *,
    protocol_name: str = "paper",
    threshold: float | None = None,
    prepare: Callable[[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> ProtocolRun:
    """Fit and score `baseline` on every split of `protocol`.

    `prepare(train, test)` may transform the two frames of a split before fitting (for example removing
    attack rows from training for a label-free detector); it must not look at test labels to change
    training data.
    """
    outputs: list[ModelOutputs] = []
    metrics: list[dict[str, float]] = []
    names: list[str] = []
    reports: list[dict[str, Any]] = []
    for i, split in enumerate(protocol.splits(frame, config.seed)):
        cfg = dataclasses.replace(config, seed=derive_seed(config.seed, "split", i))
        model = baseline(cfg)
        train, test = take_rows(frame, split.train), take_rows(frame, split.test)
        if prepare is not None:
            train, test = prepare(train, test)
        validation = take_rows(frame, split.validation) if split.validation.size else None
        model.fit(train, validation=validation)
        out = model.predict(test, protocol=f"{protocol_name}:{split.name}")
        outputs.append(out)
        metrics.append(outputs_metrics(out, threshold=threshold))
        names.append(split.name)
        reports.append(dict(model.fit_report))
    if not outputs:
        raise InvariantViolation(f"protocol {protocol.describe()} produced no split")
    return ProtocolRun(protocol.describe(), names, outputs, metrics, mean_and_spread(metrics), reports)
