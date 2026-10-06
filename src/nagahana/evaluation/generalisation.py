"""Generalisation groups: known and novel families apart, per family, cross-dataset, leave-one-network-out.

Known and novel zero-shot units are never pooled (D-23; thesis protocol P1): pooling lets strong results
on familiar attacks hide weak results on unfamiliar ones. `novelty_masks` returns them as separate
groups and `check_not_pooled` refuses any group that mixes them. Units of the zero-shot split without
a novelty mark form a group of their own ("unmarked") and are reported separately as well.

Whether "novel" covers unseen networks as well as unseen families is held (D-16); the working reading is
AS-35 (both count as zero-shot and are reported separately), set by `protocols.p1_novelty` in the
evaluation configuration. With "families_and_networks", every zero-shot unit keeps its family-based
group, and the units on networks absent from every training split (meta split "train") are reported
again in the groups "known_unseen_network" and "novel_unseen_network": the unseen-network view keeps
known and novel families apart as well.

Variants of a model run are read from `ModelOutputs.config`:

    train_datasets    datasets a P2 model was trained on (list of dataset ids)
    held_out          the network a P3 model held out of training
    site_calibration  true when a P3 model was calibrated to the held-out site (AS-26 budget)
    regime            observability regime of a P4 run
    corruption        telemetry corruption of a P5 run
    ablation          ablation id of an ablated NagaHana (config.ablations)
    base              for P-ABL runs: the protocol (P1 ... P7) whose units the run is scored on, together with
                      that protocol's own keys
P-CW runs are distinguished by the environment of their ArenaRun record.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from nagahana.core.errors import InvariantViolation
from nagahana.evaluation.config import EvaluationConfig
from nagahana.evaluation.predictions import ModelOutputs

NOVELTY_ORDER: tuple[str, ...] = ("known", "novel", "known_unseen_network", "novel_unseen_network", "unmarked", "")


def novelty_masks(meta: pd.DataFrame, *, split: str, novelty_rule: str, train_networks: set[str] | None = None
                  ) -> dict[str, np.ndarray]:
    """Boolean masks of the evaluated units of `split`, one per novelty group (never pooled).

    For splits other than the zero-shot split the units form one group keyed "".
    """
    sp = meta["split"].astype(str).to_numpy()
    nov = meta["novelty"].astype(str).to_numpy()
    in_split = sp == split
    if split != "zero_shot":
        return {"": in_split} if in_split.any() else {}
    out: dict[str, np.ndarray] = {}
    for name in ("known", "novel"):
        m = in_split & (nov == name)
        if m.any():
            out[name] = m
    unmarked = in_split & (nov == "")
    if unmarked.any():
        out["unmarked"] = unmarked
    if novelty_rule == "families_and_networks" and train_networks is not None:
        net = meta["network"].astype(str).to_numpy()
        unseen = in_split & ~np.isin(net, sorted(train_networks))
        for name in ("known", "novel"):
            m = unseen & (nov == name)
            if m.any():
                out[f"{name}_unseen_network"] = m
    return out


def check_not_pooled(meta: pd.DataFrame, mask: np.ndarray) -> None:
    """Refuse a group that mixes known and novel zero-shot units."""
    nov = set(meta.loc[mask, "novelty"].astype(str))
    if {"known", "novel"} <= nov:
        raise InvariantViolation("known and novel zero-shot units must never be pooled (D-23)")


def report_groups(meta: pd.DataFrame, cfg: EvaluationConfig) -> np.ndarray:
    """Report group of each unit (dataset id -> group id, for example the OT testbeds -> "ot")."""
    ds = meta["dataset"].astype(str).to_numpy()
    lookup = {name: cfg.dataset(name).group for name in np.unique(ds)}
    return np.array([lookup[d] for d in ds], dtype=object).astype(str)


def family_masks(meta: pd.DataFrame, base: np.ndarray) -> dict[str, np.ndarray]:
    """Per attack family: that family's units plus the benign units of the base group."""
    fam = meta["family"].astype(str).to_numpy()
    benign = base & (fam == "benign")
    return {f: base & ((fam == f) | benign) for f in sorted(set(fam[base]) - {"benign"})}


def training_networks(bundles: list[ModelOutputs]) -> set[str]:
    """Networks that appear in the training split of any record of the bundles."""
    nets: set[str] = set()
    for b in bundles:
        for task in ("detection", "forecast", "stage", "state_forecast", "time_to_event", "paths"):
            rec = getattr(b, task)
            if rec is None:
                continue
            m = rec.meta
            nets |= set(m.loc[m["split"].astype(str) == "train", "network"].astype(str))
    return nets


def variant_of(outputs: ModelOutputs, protocol: str) -> str:
    """The protocol variant a bundle belongs to ("" when the protocol has none)."""
    c: dict[str, Any] = outputs.config
    if protocol == "P-ABL":
        base = str(c.get("base", "") or "").upper()
        if base not in ("P1", "P2", "P3", "P4", "P5", "P6", "P7"):
            raise InvariantViolation(f"P-ABL outputs of {outputs.model!r} must name config['base'] (P1 ... P7)")
        inner = variant_of(outputs, base)
        return f"base={base}" + (f";{inner}" if inner else "")
    if protocol == "P-CW":
        if outputs.arena is None:
            raise InvariantViolation(f"P-CW outputs of {outputs.model!r} must carry an ArenaRun record")
        return f"environment={outputs.arena.environment}"
    if protocol == "P2":
        tr = c.get("train_datasets")
        if not tr:
            raise InvariantViolation(f"P2 outputs of {outputs.model!r} must name config['train_datasets']")
        return "train=" + ",".join(sorted(str(t) for t in (tr if isinstance(tr, list | tuple) else [tr])))
    if protocol == "P3":
        held = c.get("held_out")
        if not held:
            raise InvariantViolation(f"P3 outputs of {outputs.model!r} must name config['held_out']")
        return f"held_out={held}" + (";site" if bool(c.get("site_calibration")) else "")
    if protocol == "P4":
        reg = c.get("regime")
        if not reg:
            raise InvariantViolation(f"P4 outputs of {outputs.model!r} must name config['regime']")
        return f"regime={reg}"
    if protocol == "P5":
        cor = c.get("corruption")
        if not cor:
            raise InvariantViolation(f"P5 outputs of {outputs.model!r} must name config['corruption'] ('none' for clean)")
        return f"corruption={cor}"
    return ""


def variant_value(variant: str, key: str) -> str:
    """The value of `key` in a variant string (for example 'train' in 'train=cic-ids2017')."""
    for part in variant.split(";"):
        k, _, v = part.partition("=")
        if k == key:
            return v
    return ""
