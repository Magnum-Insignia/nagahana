"""Collation: `WindowItem`s → `WindowBatch` + `LabelBatch`, and the contract checker `validate_window_batch`.

Purpose
-------
Pads and stacks windows built by `data/windows.py` into the tensor contracts of `models/batch.py`,
using engineer A's `graph.window.collate_structures` for the structural fields, and checks that any
`WindowBatch` (ours, the synthetic one of `testing/synthetic.py`, or a future producer's) honours the
contract before a model reads it.

Owner sources, decisions, assumptions: models/batch.py (the contract), D-41 (NaN only where a cell
does not contribute), D-49 (float64 relative times, no index encodings), AS-41, AS-324 (batch shape).

Shapes (AS-324)
---------------
    U = TrainingConfig.window_updates (fixed), P = 2·U (AS-41), V = largest entity count in the batch,
    M = largest trigger count in the batch (at least 1; a window with no trigger has mask all False),
    C = len(COLUMN_SLOTS).
Padding (models/batch.py conventions): masks False; index tensors −1; times 0 for padded updates and
positions (masked); value NaN with status NOT_SUPPLIED for padded cells; contacts +inf; labels NaN /
−1 / +inf.

Checker (`validate_window_batch`)
---------------------------------
dtypes and shapes of every field; status codes in range; NaN never in a contributing real cell;
update times sorted within each window; positions sorted by (time, update, role), each real position
pointing at its update's entity in its role; `next_index` points forward to the same entity with
`next_dt` = the time difference; `entity_latest` is exactly the latest position with time ≤ trigger
time (so no position after a trigger is visible); contact matrices symmetric with zero diagonal on
real entities and C² ≤ C¹; column slots unique; graph indices in range. It raises
`InvariantViolation` naming the first broken rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import CODE_NOT_SUPPLIED, STATUS_ORDER
from nagahana.datamodel.status import CONTRIBUTING
from nagahana.governance.assumptions import assume
from nagahana.models.batch import FieldBatch, GraphBatch, LabelBatch, PositionBatch, TriggerBatch, WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.models.vocab import COLUMN_KINDS, N_STAGES, N_STATUS_CODES, NODE_KINDS, ROLES

from .windows import CANONICAL_KIND_CODES, COLUMN_SLOTS, WindowItem, field_slots

_CONTRIB_CODES = torch.tensor([i for i, s in enumerate(STATUS_ORDER) if s in CONTRIBUTING], dtype=torch.long)
INF = float("inf")


def collate_items(
    items: Sequence[WindowItem],
    cfg: NagaHanaConfig,
    *,
    novelty: Sequence[str] | None = None,
    collate_fn: Any = None,
) -> tuple[WindowBatch, LabelBatch]:
    """Pad and stack windows (shapes in the module docstring). `novelty` per window ("known" | "novel" | "")."""
    assume("AS-41", by=__name__)
    if not items:
        raise InvariantViolation("collate_items needs at least one window")
    b_n = len(items)
    u_w = cfg.training.window_updates
    p_w = 2 * u_w
    v_w = max(len(it.entity_rows) for it in items)
    m_w = max(1, max(len(it.trigger_time) for it in items))
    c_n = len(COLUMN_SLOTS)
    for it in items:
        if len(it.rows) > u_w:
            raise InvariantViolation(f"window {it.source_id}:{it.start} has {len(it.rows)} updates > {u_w}")

    # ---- inputs, padded
    values = torch.full((b_n, u_w, c_n), float("nan"), dtype=torch.float32)
    status = torch.full((b_n, u_w, c_n), CODE_NOT_SUPPLIED, dtype=torch.long)
    origin = torch.zeros(b_n, dtype=torch.float64)
    utime = torch.zeros(b_n, u_w, dtype=torch.float64)
    uent = torch.full((b_n, u_w, 3), -1, dtype=torch.long)
    umask = torch.zeros(b_n, u_w, dtype=torch.bool)
    reorder = torch.zeros(b_n, u_w, dtype=torch.float32)
    ekind = torch.zeros(b_n, v_w, dtype=torch.long)
    eint = torch.zeros(b_n, v_w, dtype=torch.bool)
    emask = torch.zeros(b_n, v_w, dtype=torch.bool)
    p_ent = torch.full((b_n, p_w), -1, dtype=torch.long)
    p_time = torch.zeros(b_n, p_w, dtype=torch.float64)
    p_upd = torch.full((b_n, p_w), -1, dtype=torch.long)
    p_role = torch.full((b_n, p_w), -1, dtype=torch.long)
    p_mask = torch.zeros(b_n, p_w, dtype=torch.bool)
    p_next = torch.full((b_n, p_w), -1, dtype=torch.long)
    p_ndt = torch.zeros(b_n, p_w, dtype=torch.float64)
    t_time = torch.zeros(b_n, m_w, dtype=torch.float64)
    t_mask = torch.zeros(b_n, m_w, dtype=torch.bool)
    t_latest = torch.full((b_n, m_w, v_w), -1, dtype=torch.long)
    # ---- labels, padded
    l_mal = torch.full((b_n, u_w), float("nan"), dtype=torch.float32)
    l_stage = torch.full((b_n, u_w), -1, dtype=torch.long)
    l_tech = torch.full((b_n, u_w), -1, dtype=torch.long)
    l_infil = torch.full((b_n, v_w), INF, dtype=torch.float64)
    l_share = torch.full((b_n, v_w, m_w), float("nan"), dtype=torch.float32)
    l_horizon = torch.full((b_n,), INF, dtype=torch.float64)

    for b, it in enumerate(items):
        u, p, v, m = len(it.rows), len(it.pos_entity), len(it.entity_rows), len(it.trigger_time)
        values[b, :u] = torch.from_numpy(it.values)
        status[b, :u] = torch.from_numpy(it.status)
        origin[b] = it.origin
        utime[b, :u] = torch.from_numpy(it.update_time)
        uent[b, :u] = torch.from_numpy(it.update_entities)
        umask[b, :u] = True
        reorder[b, :u] = torch.from_numpy(it.reorder)
        ekind[b, :v] = torch.from_numpy(it.entity_kind)
        eint[b, :v] = torch.from_numpy(it.entity_internal)
        emask[b, :v] = True
        p_ent[b, :p] = torch.from_numpy(it.pos_entity)
        p_time[b, :p] = torch.from_numpy(it.pos_time)
        p_upd[b, :p] = torch.from_numpy(it.pos_update)
        p_role[b, :p] = torch.from_numpy(it.pos_role)
        p_mask[b, :p] = True
        p_next[b, :p] = torch.from_numpy(it.next_index)
        p_ndt[b, :p] = torch.from_numpy(it.next_dt)
        t_time[b, :m] = torch.from_numpy(it.trigger_time)
        t_mask[b, :m] = True
        t_latest[b, :m, :v] = torch.from_numpy(it.entity_latest)
        l_mal[b, :u] = torch.from_numpy(it.update_malicious)
        l_stage[b, :u] = torch.from_numpy(it.update_stage)
        l_tech[b, :u] = torch.from_numpy(it.update_technique)
        l_infil[b, :v] = torch.from_numpy(it.entity_infiltrated_at)
        l_share[b, :v, :m] = torch.from_numpy(it.entity_malicious_share)
        l_horizon[b] = it.label_horizon
    # padded positions carry role "none" so a role embedding never reads −1
    p_role[~p_mask] = ROLES.index("none")

    # ---- structure (engineer A's collation)
    if collate_fn is None:
        from nagahana.graph.window import collate_structures as collate_fn
    st = collate_fn([it.structure for it in items], updates_per_window=u_w, positions_per_window=p_w,
                    entities_per_window=v_w)

    fields = FieldBatch(
        values=values, status=status, column_kind=torch.tensor(CANONICAL_KIND_CODES, dtype=torch.long),
        column_slot=torch.from_numpy(field_slots(cfg.inputs.n_slots)), column_names=COLUMN_SLOTS,
    )
    window = WindowBatch(
        fields=fields, origin=origin, update_time=utime, update_entities=uent,
        update_relation=st.update_relation, update_planes=st.update_planes, update_mask=umask,
        reorder_uncertainty=reorder, entity_kind=ekind, entity_internal=eint, entity_mask=emask,
        contact1=st.contact1, contact2=st.contact2, contact_planes=st.contact_planes,
        positions=PositionBatch(entity=p_ent, time=p_time, update=p_upd, role=p_role, mask=p_mask,
                                next_index=p_next, next_dt=p_ndt),
        graph=st.graph,
        triggers=TriggerBatch(time=t_time, mask=t_mask, entity_latest=t_latest),
    )
    labels = LabelBatch(
        update_malicious=l_mal, update_stage=l_stage, update_technique=l_tech, entity_infiltrated_at=l_infil,
        entity_malicious_share=l_share, family=tuple(it.family for it in items),
        novelty=tuple(novelty) if novelty is not None else tuple("" for _ in items),
        label_horizon=l_horizon,
    )
    return window, labels


# ====================================================================================== checker
def _need(cond: bool | torch.Tensor, what: str) -> None:
    if not bool(cond):
        raise InvariantViolation(f"WindowBatch contract: {what}")


def _check(t: torch.Tensor, name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    _need(t.dtype == dtype, f"{name} must be {dtype}, got {t.dtype}")
    _need(tuple(t.shape) == shape, f"{name} must have shape {shape}, got {tuple(t.shape)}")


def validate_window_batch(w: WindowBatch, *, n_planes: int | None = None) -> None:
    """Check `w` against the `models/batch.py` contract (module docstring). Raises on the first failure."""
    f = w.fields
    _need(f.values.dim() == 3, "fields.values must be [B, U, C]")
    b_n, u_n, c_n = f.values.shape
    v_n = int(w.entity_kind.shape[1]) if w.entity_kind.dim() == 2 else -1
    p_n = int(w.positions.entity.shape[1]) if w.positions.entity.dim() == 2 else -1
    m_n = int(w.triggers.time.shape[1]) if w.triggers.time.dim() == 2 else -1
    n_pl = n_planes if n_planes is not None else int(w.update_planes.shape[-1])
    # ---- dtypes and shapes
    _check(f.values, "fields.values", torch.float32, (b_n, u_n, c_n))
    _check(f.status, "fields.status", torch.long, (b_n, u_n, c_n))
    _check(f.column_kind, "fields.column_kind", torch.long, (c_n,))
    _check(f.column_slot, "fields.column_slot", torch.long, (c_n,))
    _need(len(f.column_names) == c_n, "column_names must name every column")
    _check(w.origin, "origin", torch.float64, (b_n,))
    _check(w.update_time, "update_time", torch.float64, (b_n, u_n))
    _check(w.update_entities, "update_entities", torch.long, (b_n, u_n, 3))
    _check(w.update_relation, "update_relation", torch.long, (b_n, u_n))
    _check(w.update_planes, "update_planes", torch.bool, (b_n, u_n, n_pl))
    _check(w.update_mask, "update_mask", torch.bool, (b_n, u_n))
    _check(w.reorder_uncertainty, "reorder_uncertainty", torch.float32, (b_n, u_n))
    _check(w.entity_kind, "entity_kind", torch.long, (b_n, v_n))
    _check(w.entity_internal, "entity_internal", torch.bool, (b_n, v_n))
    _check(w.entity_mask, "entity_mask", torch.bool, (b_n, v_n))
    _check(w.contact1, "contact1", torch.float64, (b_n, v_n, v_n))
    _check(w.contact2, "contact2", torch.float64, (b_n, v_n, v_n))
    _check(w.contact_planes, "contact_planes", torch.float64, (b_n, v_n, v_n, n_pl))
    pos = w.positions
    for name, t, dt in (("entity", pos.entity, torch.long), ("time", pos.time, torch.float64), ("update", pos.update, torch.long),
                        ("role", pos.role, torch.long), ("mask", pos.mask, torch.bool), ("next_index", pos.next_index, torch.long),
                        ("next_dt", pos.next_dt, torch.float64)):
        _check(t, f"positions.{name}", dt, (b_n, p_n))
    tr = w.triggers
    _check(tr.time, "triggers.time", torch.float64, (b_n, m_n))
    _check(tr.mask, "triggers.mask", torch.bool, (b_n, m_n))
    _check(tr.entity_latest, "triggers.entity_latest", torch.long, (b_n, m_n, v_n))

    # ---- fields: codes, slots, NaN discipline (D-41)
    _need(((f.status >= 0) & (f.status < N_STATUS_CODES)).all(), "status codes out of range")
    _need(((f.column_kind >= 0) & (f.column_kind < len(COLUMN_KINDS))).all(), "column_kind out of range")
    _need(f.column_slot.unique().numel() == c_n and bool((f.column_slot >= 0).all()), "column_slot must be unique and ≥ 0")
    contributing = torch.isin(f.status, _CONTRIB_CODES) & w.update_mask[..., None]
    _need(torch.isfinite(f.values[contributing]).all(), "a contributing cell holds NaN/inf (D-41)")

    # ---- per window: times, entities, positions, triggers, contacts
    for b in range(b_n):
        um = w.update_mask[b]
        nu = int(um.sum())
        _need(bool(um[:nu].all()) and not bool(um[nu:].any()), f"window {b}: real updates must come first")
        ut = w.update_time[b, :nu]
        _need(torch.isfinite(ut).all(), f"window {b}: update times must be finite")
        _need(bool((ut[1:] >= ut[:-1]).all()), f"window {b}: updates must be sorted by time")
        em = w.entity_mask[b]
        ue = w.update_entities[b, :nu]
        _need(bool((ue >= -1).all()) and bool((ue < v_n).all()), f"window {b}: update entity index out of range")
        real_e = ue[ue >= 0]
        _need(bool(em[real_e].all()), f"window {b}: an update touches a padded entity")
        _need(not bool((w.update_entities[b, nu:] >= 0).any()), f"window {b}: padded updates must touch no entity")
        _need(((w.entity_kind[b] >= 0) & (w.entity_kind[b] < len(NODE_KINDS))).all(), f"window {b}: entity_kind out of range")
        # positions
        pm = pos.mask[b]
        npos = int(pm.sum())
        _need(bool(pm[:npos].all()) and not bool(pm[npos:].any()), f"window {b}: real positions must come first")
        pe, pt, pu, pr = pos.entity[b, :npos], pos.time[b, :npos], pos.update[b, :npos], pos.role[b, :npos]
        _need(bool(((pu >= 0) & (pu < nu)).all()), f"window {b}: position update index out of range")
        _need(bool(((pr == 0) | (pr == 1)).all()), f"window {b}: position roles must be initiator/responder (AS-41)")
        _need(bool((w.update_entities[b, pu, pr] == pe).all()), f"window {b}: position entity ≠ its update's entity in that role")
        _need(bool((pt == w.update_time[b, pu]).all()), f"window {b}: position time ≠ its update's time")
        key_ok = (pt[1:] > pt[:-1]) | ((pt[1:] == pt[:-1]) & ((pu[1:] > pu[:-1]) | ((pu[1:] == pu[:-1]) & (pr[1:] > pr[:-1]))))
        _need(bool(key_ok.all()), f"window {b}: positions must be sorted by (time, update, role)")
        nx = pos.next_index[b, :npos]
        # next_index: the same entity's next position, or −1 if none
        exp = torch.full((npos,), -1, dtype=torch.long)
        last: dict[int, int] = {}
        for p in range(npos - 1, -1, -1):
            e = int(pe[p])
            if e in last:
                exp[p] = last[e]
            last[e] = p
        _need(bool((nx == exp).all()), f"window {b}: next_index is not the same entity's next position")
        has = nx >= 0
        dt_exp = torch.where(has, pt[nx.clamp(min=0)] - pt, torch.zeros_like(pt))
        _need(torch.allclose(pos.next_dt[b, :npos], dt_exp), f"window {b}: next_dt ≠ time to the next position")
        # triggers: entity_latest = the latest position with time ≤ τ (no future position visible)
        tm = tr.mask[b]
        for m in torch.nonzero(tm).flatten().tolist():
            tau = float(tr.time[b, m])
            exp_l = torch.full((v_n,), -1, dtype=torch.long)
            vis = torch.nonzero(pt <= tau).flatten()
            for p in vis.tolist():
                exp_l[int(pe[p])] = p
            _need(bool((tr.entity_latest[b, m] == exp_l).all()), f"window {b}, trigger {m}: entity_latest is not the latest position ≤ trigger time")
        # contacts on real entities
        nv = int(em.sum())
        c1, c2 = w.contact1[b, :nv, :nv], w.contact2[b, :nv, :nv]
        _need(bool((c1 == c1.T).all()) and bool((torch.diagonal(c1) == 0).all()), f"window {b}: C¹ must be symmetric with zero diagonal")
        _need(bool((c2 <= c1).all()), f"window {b}: C² must not exceed C¹")

    # ---- graph indices
    g = w.graph
    _validate_graph(g, b_n=b_n, u_n=u_n, p_n=p_n)


def _validate_graph(g: GraphBatch, *, b_n: int, u_n: int, p_n: int) -> None:
    n = g.num_nodes
    for name in ("node_update", "node_role", "node_kind", "node_hop", "node_owner"):
        t = getattr(g, name)
        _need(t.dtype == torch.long and tuple(t.shape) == (n,), f"graph.{name} must be long [N]")
    _need(g.node_age.shape == (n,) and g.node_age.dtype == torch.float32, "graph.node_age must be float32 [N]")
    _need(bool(((g.node_update >= -1) & (g.node_update < b_n * u_n)).all()), "graph.node_update out of range")
    _need(bool(((g.node_owner >= 0) & (g.node_owner < b_n * p_n)).all()), "graph.node_owner out of range")
    _need(bool((g.node_hop >= 0).all()), "graph.node_hop must be ≥ 0")
    _need(tuple(g.center.shape) == (b_n * p_n,), "graph.center must be [B·P]")
    _need(bool(((g.center >= -1) & (g.center < max(n, 1))).all()), "graph.center out of range")
    for pl, inc in g.incidence.items():
        _need(inc.dim() == 2 and inc.shape[0] == 2, f"graph.incidence[{pl}] must be [2, nnz]")
        e_n = int(g.hyperedge_kind[pl].shape[0])
        if inc.numel():
            _need(bool(((inc[0] >= 0) & (inc[0] < n)).all()), f"graph.incidence[{pl}] node index out of range")
            _need(bool(((inc[1] >= 0) & (inc[1] < e_n)).all()), f"graph.incidence[{pl}] hyperedge index out of range")
    _need(g.rwse.dim() == 3 and g.rwse.shape[0] == n, "graph.rwse must be [N, n_planes, steps]")


def validate_label_batch(labels: LabelBatch, w: WindowBatch) -> None:
    """Shapes and codes of a `LabelBatch` against its `WindowBatch`."""
    b_n, u_n = w.update_mask.shape
    v_n = int(w.entity_mask.shape[1])
    m_n = int(w.triggers.time.shape[1])
    _check(labels.update_malicious, "update_malicious", torch.float32, (b_n, u_n))
    _check(labels.update_stage, "update_stage", torch.long, (b_n, u_n))
    _check(labels.update_technique, "update_technique", torch.long, (b_n, u_n))
    _check(labels.entity_infiltrated_at, "entity_infiltrated_at", torch.float64, (b_n, v_n))
    _check(labels.entity_malicious_share, "entity_malicious_share", torch.float32, (b_n, v_n, m_n))
    mal = labels.update_malicious
    _need(bool((torch.isnan(mal) | (mal == 0) | (mal == 1)).all()), "update_malicious must be 0, 1 or NaN")
    _need(bool(((labels.update_stage >= -1) & (labels.update_stage < N_STAGES)).all()), "update_stage out of range")
    share = labels.entity_malicious_share
    ok = torch.isnan(share) | ((share >= 0) & (share <= 1))
    _need(bool(ok.all()), "entity_malicious_share must be in [0, 1] or NaN")
    _need(len(labels.family) == b_n, "one family per window")
    if labels.label_horizon is not None:
        _check(labels.label_horizon, "label_horizon", torch.float64, (b_n,))
        infil = labels.entity_infiltrated_at
        seen = torch.isfinite(infil)
        _need(bool((infil <= labels.label_horizon[:, None])[seen].all()), "an infiltration time lies beyond the label horizon")


__all__ = ["collate_items", "validate_label_batch", "validate_window_batch"]

