"""Data-model layering (P-04): what each layer holds, and the topic it travels on.

Requirement (datamodel.md items 1, 3, 9, 10): a Kafka-compatible, human-auditable superset data model
that accommodates every passive source (pcap, NetFlow, sFlow, IPFIX, Zeek, Suricata, Snort, routers,
switches, TAPs, custom hardware), compatible with OCSF and CSTS without letting either limit the depth
of what is recorded.

Layers and what lives in each:

    L0  RAW          the immutable source record: its bytes, their SHA-256, and where they sit in the
                     source (file, byte offset, line, Kafka partition and offset). Chain of custody.
    L1  EVENT        the canonical event: every field of the source record, mapped to a catalogue
                     field (shared across sources where the meaning is the same, source-native
                     otherwise) or retained as an uncatalogued attribute. OCSF-compatible
                     (`ingest/ocsf.py`).
    L2  STATE        the entity-relational state update: typed entities with roles, the relation they
                     form, and the fields that describe the transition. CSTS-compatible
                     (`ingest/csts.py`).
    L3  EVIDENCE     observation status, reliability and age of every value, clock quality, sampling
                     and sensor-loss evidence (P-03, D-41).
    L4  OT_CII       OT and CII extensions: industrial protocol semantics (Modbus, DNP3, IEC 60870-5-104).
    L5  MACROSTATE   values derived across records (port-access evidence, query-name statistics).

Every catalogue field names its layer (`fields.FieldSpec.layer`). A sensor change touches only the L0
and L1 adapters; the model's input contract (L2 to L5) stays stable, which is the "narrow waist" of
the design. Each layer has its own Kafka topic (`topic_name`).
"""

from __future__ import annotations

import enum


class Layer(enum.Enum):
    """Data-model layers (P-04). Values double as Kafka topic suffixes."""

    RAW = "l0-raw"
    EVENT = "l1-event"
    STATE = "l2-state"
    EVIDENCE = "l3-evidence"
    OT_CII = "l4-ot-cii"
    MACROSTATE = "l5-macrostate"

    @property
    def level(self) -> int:
        """Layer number 0 to 5."""
        return _LEVEL[self]

    @property
    def label(self) -> str:
        """Short label "L0" to "L5"."""
        return f"L{_LEVEL[self]}"


_LEVEL: dict[Layer, int] = {layer: i for i, layer in enumerate(Layer)}

#: What each layer holds, in one line (used by the documentation renderer).
LAYER_SUMMARY: dict[Layer, str] = {
    Layer.RAW: "immutable source record, its SHA-256 and its location",
    Layer.EVENT: "canonical event: every source field mapped or retained (OCSF-compatible)",
    Layer.STATE: "entity-relational state update: entities, roles, relation, transition fields (CSTS-compatible)",
    Layer.EVIDENCE: "observation status, reliability, age, clock quality, sampling and sensor-loss evidence",
    Layer.OT_CII: "industrial protocol semantics (Modbus, DNP3, IEC 60870-5-104)",
    Layer.MACROSTATE: "values derived across records",
}


def topic_name(site: str, layer: Layer) -> str:
    """Kafka topic of a site's layer: `nagahana.<site>.<layer>` (site: no spaces, dots or slashes)."""
    if not site or any(c in site for c in " ./"):
        raise ValueError("site must be a non-empty token without spaces, dots or slashes")
    return f"nagahana.{site}.{layer.value}"


def parse_layer(text: str) -> Layer:
    """The layer named by `text`: its value ("l2-state"), its label ("L2") or its name ("STATE")."""
    for layer in Layer:
        if text in (layer.value, layer.label, layer.name):
            return layer
    raise ValueError(f"Unknown data-model layer {text!r}")
