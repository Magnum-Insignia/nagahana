"""Router and switch telemetry mapping table: gNMI notifications carrying OpenConfig interface state.

Records are gNMI `Notification` messages (or `SubscribeResponse` messages wrapping one) in the
protobuf JSON mapping, one per line: gNMI specification (openconfig/reference, gnmi-specification.md,
sections 2.1 and 2.2; gnmi.proto messages Notification, Update, Path, PathElem, TypedValue) and the
proto3 JSON mapping (64-bit integers as strings, bytes as base64). Leaf paths are those of the
openconfig-interfaces model (interfaces/interface[name]/state, state/counters) and
openconfig-if-ethernet (ethernet/state). One state update is built per interface named in a
notification; the row sources below are leaf paths relative to interfaces/interface[name=...].
"""

from __future__ import annotations

from nagahana.datamodel.native import R, RecordMap
from nagahana.datamodel.spec import Level
from nagahana.datamodel.status import STATE_FACT_STATUSES

REF = "gNMI specification and gnmi.proto; OpenConfig openconfig-interfaces and openconfig-if-ethernet models"

_COUNTERS = (
    "in-octets", "in-pkts", "in-unicast-pkts", "in-broadcast-pkts", "in-multicast-pkts", "in-discards", "in-errors",
    "in-unknown-protos", "in-fcs-errors", "out-octets", "out-pkts", "out-unicast-pkts", "out-broadcast-pkts",
    "out-multicast-pkts", "out-discards", "out-errors", "carrier-transitions",
)

INTERFACE = RecordMap(
    "gnmi", "interface", "gNMI OpenConfig interface telemetry", Level.DEVICE, (
        R("timestamp", "i", "@time", "epoch_ns", note="Notification timestamp, nanoseconds since the epoch."),
        R("target", "s", "event.hostname", "string", note="prefix.target: the device the data describes."),
        R("origin", "s"),
        R("name", "s", "dev.if_name", "string", note="Key of interfaces/interface."),
        R("state/name", "s"),
        R("state/ifindex", "i", "dev.if_index", "count"),
        R("state/type", "s", kind="fp", note="IANA interface type identity."),
        R("state/mtu", "i", conv="count", unit="bytes"),
        R("state/description", "s"),
        R("state/enabled", "b", conv="bool"),
        R("state/admin-status", "s", note="UP, DOWN or TESTING."),
        R("state/oper-status", "s", note="UP, DOWN, TESTING, UNKNOWN, DORMANT, NOT_PRESENT, LOWER_LAYER_DOWN."),
        R("state/last-change", "i", conv="count", unit="ns"),
        R("state/logical", "b", conv="bool"),
        R("state/management", "b", conv="bool"),
        R("state/cpu", "b", conv="bool"),
        *(R(f"state/counters/{c}", "i", conv="count", unit="bytes" if "octets" in c else "packets",
            note="Cumulative counter.") for c in _COUNTERS),
        R("state/counters/last-clear", "i", conv="count", unit="ns"),
        R("ethernet/state/port-speed", "s", "dev.if_speed", "oc_port_speed", also="native"),
        R("ethernet/state/negotiated-port-speed", "s", kind="fp"),
        R("ethernet/state/duplex-mode", "s"),
        R("ethernet/state/negotiated-duplex-mode", "s"),
        R("ethernet/state/mac-address", "m", conv="mac", kind="id"),
        R("ethernet/state/hw-mac-address", "m", conv="mac", kind="id"),
        R("ethernet/state/auto-negotiate", "b", conv="bool"),
        *(R(f"ethernet/state/counters/{c}", "i", conv="count", unit="frames") for c in (
            "in-mac-control-frames", "in-mac-pause-frames", "in-oversize-frames", "in-undersize-frames", "in-jabber-frames",
            "in-fragment-frames", "in-8021q-frames", "in-crc-errors", "in-block-errors", "in-carrier-errors",
            "in-interrupted-tx", "in-late-collision", "in-mac-errors-rx", "in-single-collision", "in-symbol-error",
            "in-maxsize-exceeded", "out-mac-control-frames", "out-mac-pause-frames", "out-8021q-frames",
            "out-mac-errors-tx")),
    ), REF, native_statuses=STATE_FACT_STATUSES, ocsf_class=0,
    notes=(
        "Counter deltas between consecutive notifications of the same target and interface (64-bit counters; a "
        "decrease is a counter reset and gives NOT_SUPPLIED): dev.if_in_octets, dev.if_out_octets, dev.if_in_packets "
        "and dev.if_out_packets (in-pkts / out-pkts, else unicast + multicast + broadcast), dev.if_errors and "
        "dev.if_discards (in + out), the per-direction components, dev.interval from the timestamps.",
        "dev.if_status: bit 0 from admin-status UP, bit 1 from oper-status UP (OBSERVED only when both are reported "
        "or carried from the interface's last report, STALE with its age in that case, AS-702).",
        "Entities: the target as subject. Paths outside interfaces/interface are retained as attributes 'gnmi.<path>'.",
    ),
)

DEVTELEMETRY_MAPS: tuple[RecordMap, ...] = (INTERFACE,)
