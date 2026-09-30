"""Data-model layering (proposal P-04, awaiting approval).

The requirement (datamodel.md; [Q-15], [Q-27], [A-23])
------------------------------------------------------
A Kafka-compatible, human-auditable **superset** data model that accommodates every passive source
(pcap, NetFlow/sFlow/IPFIX, Zeek, Suricata, Snort, routers/switches/TAPs, custom hardware). It
follows OCSF and CSTS "but nothing should limit the data depth" (datamodel.md item 9). "We must also
create the data model as of now and keep updating it overtime" [A-23].

The proposed layers
-------------------
    L0  RAW          immutable source record, hashed (chain of custody)
    L1  EVENT        OCSF-compatible canonical event (D-06: "OSTF" read as OCSF, to confirm)
    L2  STATE        CSTS-compatible entity-relational state (D-06: CSTS = Rahman 2026, to confirm)
    L3  EVIDENCE     our additions: observation status, reliability, clock quality (P-03)
    L4  OT_CII       OT/CII extensions: asset role, Purdue level, zone/conduit, protocol semantics
    L5  MACROSTATE   derived macrostates (ARCH §3.2: model the macrostate, not the microstate)

Each layer gets its own Kafka topic and versioned schema. That way a sensor change touches only
L0/L1 adapters, and the model's input contract (L2–L5) stays stable (ARCH §9.2 "narrow waist").

Why this is only an enum for now
--------------------------------
The layer *contents* depend on D-06 (which standards) and on stage-1 analysis of the real datasets
[Q-02]. Encoding the layer names now lets adapters and topics refer to them. Anything beyond that
would be deciding for the owner.
"""

from __future__ import annotations

import enum


class Layer(enum.Enum):
    """Proposed data-model layers (P-04). Values double as Kafka topic suffixes."""

    RAW = "l0-raw"
    EVENT = "l1-event"
    STATE = "l2-state"
    EVIDENCE = "l3-evidence"
    OT_CII = "l4-ot-cii"
    MACROSTATE = "l5-macrostate"


def topic_name(site: str, layer: Layer) -> str:
    """Kafka topic naming convention proposed with P-04: `nagahana.<site>.<layer>`."""
    if not site or any(c in site for c in " ./"):
        raise ValueError("site must be a non-empty token without spaces, dots or slashes")
    return f"nagahana.{site}.{layer.value}"
