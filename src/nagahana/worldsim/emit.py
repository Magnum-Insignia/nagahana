"""Assemble a world's observed records and ground truth into NagaHana data-model artifacts (P-14).

The observation layer produces per-session records and their aligned labels. This module turns them
into the objects the rest of the pipeline consumes and that the evaluation scores against known
truth:

    source      a `data.windows.SourceData`: the columnar event log (`datamodel.columnar`) plus the
                mapped label table (`data.labels` columns), ready to window and feed the model.
    episodes    a `evaluation.predictions.EpisodeTable`: one row per compromised internal entity, with
                the start of malicious activity and the time it reached an infiltration stage (AS-18).
    entity_truth per-entity hidden state at the end of the run (compromise, privilege, persistence,
                impact), the ground truth belief and forecast are judged against.
    event_truth  every ground-truth event with its causal parent (the attack DAG, per-event cause
                links), so causal-structure constraints can be checked against the known causes.
    alerts      the intrusion-detection alerts (true and false), a ground-truth table rather than a
                data-model field.
    meta        scenario, campaign, seed, world index, saturation and counts.

Labels never enter the state updates (they are ground truth, kept beside the records and keyed by the
same sequence number). Records validate against the data model by construction, because they are
built through `datamodel.columnar.to_columnar`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nagahana.datamodel.columnar import ColumnarUpdates, columns_for, to_columnar
from nagahana.worldsim import vocab
from nagahana.worldsim.config import ScenarioConfig
from nagahana.worldsim.observe import ObservationResult
from nagahana.worldsim.state import EVENT_FIELDS, KIND_ATTACK
from nagahana.worldsim.topology import Topology

ADAPTER = "worldsim"
ADAPTER_VERSION = "1.0.0"

#: Labels of the world simulator, aligned with `data.labels.LABEL_COLUMNS`.
LABEL_COLUMNS: tuple[str, ...] = (
    "seq", "record", "label_raw", "malicious", "stage", "technique", "family", "subfamily",
    "actor_role", "mapped",
)
_PAYLOAD_LABELS: tuple[str, ...] = ("0", "1-63", "64-127", "128-255", "256-511", "512-1023", "1024-1459", ">=1460")


@dataclass
class WorldOutput:
    """Everything one simulated world produces."""

    updates: ColumnarUpdates
    labels: pd.DataFrame
    episodes: pd.DataFrame
    entity_truth: pd.DataFrame
    event_truth: pd.DataFrame
    alerts: pd.DataFrame
    meta: dict[str, Any]

    def source_data(self) -> Any:
        """Wrap the columnar log and labels as a `data.windows.SourceData` (imported lazily)."""
        from nagahana.data.windows import SourceData

        return SourceData(updates=self.updates, labels=self.labels, network=self.meta["network"],
                          dataset=self.meta["scenario"], origin="generated")

    def episode_table(self) -> Any:
        """Wrap the episodes as an `evaluation.predictions.EpisodeTable` (imported lazily)."""
        from nagahana.evaluation.predictions import EpisodeTable

        return EpisodeTable(frame=self.episodes)


def _columns() -> Any:
    from nagahana.worldsim.observe import WORLDSIM_FIELDS

    return columns_for(WORLDSIM_FIELDS, histogram_bins={"pkt.payload_size_hist": _PAYLOAD_LABELS})


def build_world_output(scenario: ScenarioConfig, topo: Topology, result: ObservationResult,
                       events: np.ndarray, saturated: bool, seed: int, world_index: int,
                       final_state: Any) -> WorldOutput:
    """Assemble the data-model artifacts and ground-truth tables of one world."""
    records = result.records
    updates = [r.update for r in records]
    columns = _columns()
    if updates:
        cu = to_columnar(updates, columns)
        cu.source_id = f"{scenario.name}:w{world_index}"
    else:
        cu = _empty_columnar(scenario, world_index, columns)

    labels = _label_table(records)
    episodes = _episode_table(scenario, topo, records)
    entity_truth = _entity_truth(topo, final_state)
    event_truth = _event_truth(topo, events)
    alerts = _alert_table(topo, result)
    meta = {
        "scenario": scenario.name, "network": scenario.network, "campaign": scenario.attack.campaign,
        "family": vocab.campaign(scenario.attack.campaign).family, "novelty": scenario.novelty,
        "seed": int(seed), "world_index": int(world_index), "saturated": bool(saturated),
        "n_entities": int(topo.n_entities), "n_records": int(len(records)),
        "n_sessions": int(len(result.sessions)), "n_alerts": int(len(result.alerts)),
        "n_attack_events": int((events[:, 0] == KIND_ATTACK).sum()),
        "horizon_s": float(scenario.simulation.horizon_s),
    }
    return WorldOutput(updates=cu, labels=labels, episodes=episodes, entity_truth=entity_truth,
                       event_truth=event_truth, alerts=alerts, meta=meta)


def _empty_columnar(scenario: ScenarioConfig, world_index: int, columns: Any) -> ColumnarUpdates:
    # A valid, empty columnar log (a world that produced no observable record).
    frame = pd.DataFrame({c: pd.Series([], dtype="float64") for c in
                          ("seq", "record", "event_time", "ingest_time", "watermark", "reorder_uncertainty_s")})
    for c in ("entity_0", "entity_1", "relation", "direction", "raw_offset", "raw_len"):
        frame[c] = pd.Series([], dtype="int64")
    cu = ColumnarUpdates(
        source_id=f"{scenario.name}:w{world_index}", adapter=ADAPTER, adapter_version=ADAPTER_VERSION,
        columns=columns, updates=frame, entities=pd.DataFrame({"kind": [], "key": []}),
        relations=pd.DataFrame(), values=np.zeros((0, len(columns)), dtype=np.float64),
        status=np.zeros((0, len(columns)), dtype=np.uint8), raw_hash=np.zeros((0, 32), dtype=np.uint8),
        clock_quality="ntp-synced",
    )
    return cu


def _label_table(records: list[Any]) -> pd.DataFrame:
    n = len(records)
    seq = np.arange(n, dtype=np.int64)
    raw = np.array([("attack" if r.malicious == 1.0 else "benign") for r in records], dtype=object)
    return pd.DataFrame({
        "seq": seq, "record": seq,
        "label_raw": raw,
        "malicious": np.array([r.malicious for r in records], dtype=np.float32),
        "stage": np.array([r.stage for r in records], dtype=np.int64),
        "technique": np.array([r.technique for r in records], dtype=object),
        "family": np.array([r.family for r in records], dtype=object),
        "subfamily": np.array([r.family for r in records], dtype=object),
        "actor_role": np.array([r.actor_role for r in records], dtype=np.int8),
        "mapped": np.ones(n, dtype=bool),
    })


def _episode_table(scenario: ScenarioConfig, topo: Topology, records: list[Any]) -> pd.DataFrame:
    from nagahana.models.vocab import INFILTRATION_STAGES

    # The compromised internal entity of a malicious record is its internal actor.
    start: dict[int, float] = {}
    completion: dict[int, float] = {}
    for r in records:
        if r.malicious != 1.0:
            continue
        upd = r.update
        actor_entity = r.responder_entity if r.actor_role == 1 else _initiator_entity(topo, upd)
        if actor_entity < 0 or not bool(topo.internal[actor_entity]):
            continue
        t = float(upd.ordering.event_time)
        start[actor_entity] = min(start.get(actor_entity, t), t)
        if r.stage in INFILTRATION_STAGES:
            completion[actor_entity] = min(completion.get(actor_entity, t), t)
    rows = []
    fam = vocab.campaign(scenario.attack.campaign).family
    for ent, s in sorted(start.items()):
        comp = completion.get(ent, s)
        rows.append({
            "episode": f"{scenario.name}:w?:e{ent}", "dataset": scenario.name, "network": scenario.network,
            "family": fam, "novelty": scenario.novelty, "entity": int(ent), "start": s,
            "completion": max(comp, s),
        })
    if not rows:
        return pd.DataFrame({c: pd.Series([], dtype="float64" if c in ("start", "completion") else "object")
                             for c in ("episode", "dataset", "network", "family", "novelty", "entity", "start", "completion")})
    return pd.DataFrame(rows)


def _initiator_entity(topo: Topology, upd: Any) -> int:
    # Recover the initiator entity row from its address (entities[0] is the initiator by convention).
    addr = upd.entities[0].id
    hits = np.nonzero(topo.address == addr)[0]
    return int(hits[0]) if hits.size else -1


def _entity_truth(topo: Topology, final: Any) -> pd.DataFrame:
    v = topo.n_entities
    foothold = np.asarray(final.foothold)
    return pd.DataFrame({
        "entity": np.arange(v, dtype=np.int64),
        "address": topo.address.astype(str),
        "kind": topo.kind.astype(str),
        "archetype": np.array([vocab.ARCHETYPES[a].name for a in topo.archetype], dtype=object),
        "internal": topo.internal,
        "control_level": foothold,
        "compromised": foothold >= 1,
        "privileged": foothold >= 2,
        "persistent": np.asarray(final.persist),
        "collected": np.asarray(final.collected),
        "exfiltrated": np.asarray(final.exfiltrated),
        "manipulated": np.asarray(final.manipulated),
        "ransomed": np.asarray(final.ransomed),
    })


def _event_truth(topo: Topology, events: np.ndarray) -> pd.DataFrame:
    used_slots = np.nonzero(events[:, 0] > 0)[0]
    used = events[used_slots]
    cols = {name: used[:, i] for i, name in enumerate(EVENT_FIELDS)}
    df = pd.DataFrame(cols)
    df["event"] = np.arange(len(df), dtype=np.int64)
    # Remap the cause pointer from the original slot index to the row of this compacted table, so the
    # attack DAG is expressed over the emitted events (parent row, or -1 for a root).
    slot_to_row = {int(s): i for i, s in enumerate(used_slots)}
    df["cause"] = [slot_to_row.get(int(c), -1) if c >= 0 else -1 for c in used[:, EVENT_FIELDS.index("cause")]]
    df["technique_id"] = [vocab.TECHNIQUES[t].attack_id if t >= 0 else "" for t in df["technique"]]
    df["tactic_id"] = [vocab.TECHNIQUES[t].tactic_id if t >= 0 else "" for t in df["technique"]]
    return df


def _alert_table(topo: Topology, result: ObservationResult) -> pd.DataFrame:
    a = result.alerts
    return pd.DataFrame({
        "t_us": np.array([x.t_us for x in a], dtype=np.int64),
        "sensor": np.array([x.sensor for x in a], dtype=object),
        "entity": np.array([x.entity for x in a], dtype=np.int64),
        "address": np.array([str(topo.address[x.entity]) for x in a], dtype=object),
        "true_alert": np.array([x.true_alert for x in a], dtype=bool),
        "technique_id": np.array([x.technique for x in a], dtype=object),
    })


__all__ = ["ADAPTER", "ADAPTER_VERSION", "LABEL_COLUMNS", "WorldOutput", "build_world_output"]
