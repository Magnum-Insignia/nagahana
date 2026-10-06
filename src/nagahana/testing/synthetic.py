"""Synthetic `WindowBatch` + `LabelBatch` for unit tests of every component.

What it guarantees (so component tests can rely on it)
------------------------------------------------------
- Times are float64 relative to a float64 epoch origin; updates are sorted by time.
- Every update has a distinct initiator and responder; positions are 2 per update (AS-41), sorted by
  (time, update, role); `next_index`/`next_dt` point to the same entity's next position.
- Contact matrices are as-of correct: C¹[u, v] is the time of the first update joining u and v,
  C² = min_w max(C¹[u, w], C¹[w, v]) (diagonals 0), per-plane contacts likewise.
- Each position's local graph is built **as of its own time**: the centre entity, its neighbours with
  C¹ ≤ t (most recent first, capped), node features from their latest update with time ≤ t, and one
  hyperedge per (centre, neighbour) contact on each plane where that contact exists by t.
- Triggers at a fixed cadence; `entity_latest` uses only positions with time ≤ trigger time.
- Field statuses mix OBSERVED with NOT_SUPPLIED / NOT_OBSERVABLE; excluded cells hold NaN (D-41).

It is *not* a traffic simulator: values carry no network meaning. The ground-truth world simulator
is proposal P-14.
"""

from __future__ import annotations

import numpy as np
import torch

from nagahana.datamodel.columnar import CODE_NOT_OBSERVABLE, CODE_NOT_SUPPLIED, CODE_OBSERVED
from nagahana.datamodel.fields import Kind
from nagahana.models.batch import FieldBatch, GraphBatch, LabelBatch, PositionBatch, TriggerBatch, WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.models.vocab import COLUMN_KIND_CODE, N_STAGES, NODE_KINDS
from nagahana.nn.positional import random_walk_se

#: Column layout of synthetic fields: (name, kind). Mirrors a few catalogue fields.
SYNTH_COLUMNS: tuple[tuple[str, Kind], ...] = (
    ("flow.duration", Kind.CONTINUOUS),
    ("flow.iat_mean", Kind.CONTINUOUS),
    ("flow.iat_max", Kind.CONTINUOUS),
    ("flow.bytes_fwd", Kind.COUNT),
    ("flow.bytes_bwd", Kind.COUNT),
    ("flow.packets_fwd", Kind.COUNT),
    ("flow.packets_bwd", Kind.COUNT),
    ("flow.flag_count.syn", Kind.COUNT),
    ("flow.dst_port", Kind.CATEGORICAL),
    ("flow.protocol", Kind.CATEGORICAL),
    ("flow.tcp_flags", Kind.BITMASK),
)
INF = float("inf")


def _min_max_product(c1: np.ndarray) -> np.ndarray:
    # C²[u, v] = min_w max(C¹[u, w], C¹[w, v]) (tropical min-max product).
    return np.min(np.maximum(c1[:, :, None], c1[None, :, :]), axis=1)


def make_window_batch(
    cfg: NagaHanaConfig,
    *,
    batch: int = 2,
    updates: int = 24,
    entities: int = 8,
    seed: int = 0,
    malicious_entity: int | None = 1,
) -> tuple[WindowBatch, LabelBatch]:
    """Build B synthetic windows. See the module docstring for the guarantees."""
    rng = np.random.default_rng(seed)
    n_planes = len(cfg.graph.planes)
    b_, u_, v_, p_ = batch, updates, entities, 2 * updates
    c_ = len(SYNTH_COLUMNS)

    values = np.full((b_, u_, c_), np.nan, dtype=np.float32)
    status = np.full((b_, u_, c_), CODE_NOT_SUPPLIED, dtype=np.int64)
    utime = np.zeros((b_, u_), dtype=np.float64)
    uent = np.full((b_, u_, 3), -1, dtype=np.int64)
    urel = np.zeros((b_, u_), dtype=np.int64)
    uplanes = np.zeros((b_, u_, n_planes), dtype=bool)
    ekind = np.zeros((b_, v_), dtype=np.int64)
    einternal = np.ones((b_, v_), dtype=bool)
    c1 = np.full((b_, v_, v_), INF)
    cpl = np.full((b_, v_, v_, n_planes), INF)
    malicious = np.zeros((b_, u_), dtype=np.float32)
    stage = np.zeros((b_, u_), dtype=np.int64)
    infil = np.full((b_, v_), INF)

    for b in range(b_):
        ekind[b] = rng.integers(0, len(NODE_KINDS), v_)
        einternal[b, -1] = False                                    # one external entity
        utime[b] = np.cumsum(rng.exponential(5.0, u_))
        pair_id: dict[tuple[int, int], int] = {}
        for u in range(u_):
            i, r = (int(x) for x in rng.choice(v_, 2, replace=False))
            uent[b, u, 0], uent[b, u, 1] = i, r
            key = (min(i, r), max(i, r))
            urel[b, u] = pair_id.setdefault(key, len(pair_id))
            uplanes[b, u, 0] = True                                 # connectivity: every flow
            uplanes[b, u, 1 + rng.integers(0, n_planes - 1)] = rng.random() < 0.5
            t = utime[b, u]
            c1[b, i, r] = c1[b, r, i] = min(c1[b, i, r], t)
            for pl_i in np.nonzero(uplanes[b, u])[0].tolist():
                cpl[b, i, r, pl_i] = cpl[b, r, i, pl_i] = min(cpl[b, i, r, pl_i], t)
            # Fields: mostly observed, some not supplied / not observable; NaN where excluded.
            for c, (_, kind) in enumerate(SYNTH_COLUMNS):
                roll = rng.random()
                if roll < 0.8:
                    status[b, u, c] = CODE_OBSERVED
                    if kind is Kind.CATEGORICAL:
                        values[b, u, c] = float(rng.choice([22, 53, 80, 443, 445, 3389, 31337]))
                    elif kind is Kind.BITMASK:
                        values[b, u, c] = float(rng.integers(0, 64))
                    else:
                        values[b, u, c] = float(rng.lognormal(3.0, 2.0))
                else:
                    status[b, u, c] = CODE_NOT_SUPPLIED if roll < 0.9 else CODE_NOT_OBSERVABLE
            if malicious_entity is not None and malicious_entity in (i, r) and u >= u_ // 2:
                malicious[b, u] = 1.0
                stage[b, u] = 1 + int(rng.integers(0, N_STAGES - 1))
        np.fill_diagonal(c1[b], 0.0)
        for pl_i in range(n_planes):
            np.fill_diagonal(cpl[b, :, :, pl_i], 0.0)
        if malicious_entity is not None:
            mal_times = utime[b][(malicious[b] > 0)]
            if mal_times.size:
                infil[b, malicious_entity] = mal_times[0]
    c2 = np.stack([_min_max_product(c1[b]) for b in range(b_)])

    # ------------------------------------------------------------------ positions (2 per update)
    pent = uent[:, :, :2].reshape(b_, p_)
    ptime = np.repeat(utime, 2, axis=1)
    pupd = np.repeat(np.arange(u_)[None, :], b_, axis=0).repeat(2, axis=1)
    prole = np.tile(np.array([0, 1]), (b_, u_))
    pnext = np.full((b_, p_), -1, dtype=np.int64)
    pnext_dt = np.zeros((b_, p_), dtype=np.float64)
    for b in range(b_):
        last: dict[int, int] = {}
        for p in range(p_ - 1, -1, -1):  # noqa: B007
            e = int(pent[b, p])
            if e in last:
                pnext[b, p] = last[e]
                pnext_dt[b, p] = ptime[b, last[e]] - ptime[b, p]
            last[e] = p

    # ------------------------------------------------------------------ local graphs as of t
    node_update: list[int] = []
    node_role: list[int] = []
    node_kind: list[int] = []
    node_age: list[float] = []
    node_hop: list[int] = []
    node_owner: list[int] = []
    center = np.full(b_ * p_, -1, dtype=np.int64)
    inc: dict[str, list[tuple[int, int]]] = {pl: [] for pl in cfg.graph.planes}
    hkind: dict[str, list[int]] = {pl: [] for pl in cfg.graph.planes}
    hent: dict[str, list[tuple[int, int]]] = {pl: [] for pl in cfg.graph.planes}
    rwse_rows: list[np.ndarray] = []
    for b in range(b_):
        for p in range(p_):
            e, t = int(pent[b, p]), float(ptime[b, p])
            nbrs = [w for w in range(v_) if w != e and c1[b, e, w] <= t]
            nbrs.sort(key=lambda w: -c1[b, e, w])
            members = [e, *nbrs[: cfg.graph.max_nodes - 1]]
            base = len(node_kind)
            center[b * p_ + p] = base
            for hop, w in enumerate(members):
                past = [u for u in range(u_) if utime[b, u] <= t and w in uent[b, u, :2]]
                lu = past[-1] if past else -1
                node_update.append(b * u_ + lu if lu >= 0 else -1)
                node_role.append(int(np.nonzero(uent[b, lu, :2] == w)[0][0]) if lu >= 0 else 3)
                node_kind.append(int(ekind[b, w]))
                node_age.append(float(t - utime[b, lu]) if lu >= 0 else 0.0)
                node_hop.append(0 if hop == 0 else 1)
                node_owner.append(b * p_ + p)
            adj = np.zeros((n_planes, len(members), len(members)))
            for j, w in enumerate(members[1:], start=1):
                for pi, pl in enumerate(cfg.graph.planes):
                    if cpl[b, e, w, pi] <= t:
                        hid = len(hkind[pl])
                        inc[pl].extend([(base, hid), (base + j, hid)])
                        hkind[pl].append(0)
                        hent[pl].append((e, w))
                        adj[pi, 0, j] = adj[pi, j, 0] = 1.0
            rwse_rows.append(np.stack([random_walk_se(torch.from_numpy(adj[pi]), cfg.graph.rwse_steps).numpy()
                                       for pi in range(n_planes)], axis=1))
    graph = GraphBatch(
        node_update=torch.tensor(node_update), node_role=torch.tensor(node_role), node_kind=torch.tensor(node_kind),
        node_age=torch.tensor(node_age, dtype=torch.float32), node_hop=torch.tensor(node_hop),
        node_owner=torch.tensor(node_owner), center=torch.from_numpy(center),
        incidence={pl: (torch.tensor(inc[pl], dtype=torch.long).t().contiguous() if inc[pl]
                        else torch.zeros(2, 0, dtype=torch.long)) for pl in cfg.graph.planes},
        hyperedge_kind={pl: torch.tensor(hkind[pl], dtype=torch.long) for pl in cfg.graph.planes},
        hyperedge_entities={pl: (torch.tensor(hent[pl], dtype=torch.long) if hent[pl]
                                 else torch.zeros(0, 2, dtype=torch.long)) for pl in cfg.graph.planes},
        rwse=torch.from_numpy(np.concatenate(rwse_rows, axis=0)).float(),
    )

    # ------------------------------------------------------------------ triggers (fixed cadence)
    cadence = cfg.forecaster.window_seconds / 4.0          # several triggers per short window
    m_ = max(1, int(utime.max() // cadence))
    ttime = np.array([[cadence * (m + 1) for m in range(m_)] for _ in range(b_)], dtype=np.float64)
    tlatest = np.full((b_, m_, v_), -1, dtype=np.int64)
    share = np.full((b_, v_, m_), np.nan, dtype=np.float32)
    for b in range(b_):
        for m in range(m_):
            for p in range(p_):
                if ptime[b, p] <= ttime[b, m]:
                    tlatest[b, m, pent[b, p]] = p
            for v in range(v_):
                rows = [u for u in range(u_) if utime[b, u] <= ttime[b, m] and v in uent[b, u, :2]]
                if rows:
                    share[b, v, m] = float(malicious[b, rows].mean())

    kinds = torch.tensor([COLUMN_KIND_CODE[k] for _, k in SYNTH_COLUMNS])
    fields = FieldBatch(values=torch.from_numpy(values), status=torch.from_numpy(status), column_kind=kinds,
                        column_slot=torch.arange(c_), column_names=tuple(n for n, _ in SYNTH_COLUMNS))
    window = WindowBatch(
        fields=fields,
        origin=torch.full((b_,), 1_519_826_740.0, dtype=torch.float64),
        update_time=torch.from_numpy(utime), update_entities=torch.from_numpy(uent),
        update_relation=torch.from_numpy(urel), update_planes=torch.from_numpy(uplanes),
        update_mask=torch.ones(b_, u_, dtype=torch.bool), reorder_uncertainty=torch.zeros(b_, u_),
        entity_kind=torch.from_numpy(ekind), entity_internal=torch.from_numpy(einternal),
        entity_mask=torch.ones(b_, v_, dtype=torch.bool),
        contact1=torch.from_numpy(c1), contact2=torch.from_numpy(c2), contact_planes=torch.from_numpy(cpl),
        positions=PositionBatch(entity=torch.from_numpy(pent), time=torch.from_numpy(ptime),
                                update=torch.from_numpy(pupd), role=torch.from_numpy(prole),
                                mask=torch.ones(b_, p_, dtype=torch.bool), next_index=torch.from_numpy(pnext),
                                next_dt=torch.from_numpy(pnext_dt)),
        graph=graph,
        triggers=TriggerBatch(time=torch.from_numpy(ttime), mask=torch.ones(b_, m_, dtype=torch.bool),
                              entity_latest=torch.from_numpy(tlatest)),
    )
    labels = LabelBatch(
        update_malicious=torch.from_numpy(malicious), update_stage=torch.from_numpy(stage),
        update_technique=torch.full((b_, u_), -1, dtype=torch.long), entity_infiltrated_at=torch.from_numpy(infil),
        entity_malicious_share=torch.from_numpy(share), family=tuple("synthetic" for _ in range(b_)),
        novelty=tuple("" for _ in range(b_)),
    )
    return window, labels
