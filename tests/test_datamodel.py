"""Data model: absence ≠ zero, status invariants, required field coverage, stable slots."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.datamodel import fields
from nagahana.datamodel.records import EntityRef, FieldValue, OrderingInfo, Provenance, StateUpdate
from nagahana.datamodel.status import (
    CONTRIBUTING,
    EXCLUDED,
    ObservationStatus,
    contributing_mask,
    evidence_weight,
    validate_weights,
)
from nagahana.datamodel.versioning import FieldIndex

S = ObservationStatus


@given(st.sampled_from(list(S)), st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)))
def test_field_value_status_invariants(status, value):
    kwargs = {}
    if status is S.LOW_RELIABILITY:
        kwargs["reliability"] = 0.5
    if status is S.STALE:
        kwargs["age_s"] = 3.0
    ok = (status in EXCLUDED and value is None) or (status in CONTRIBUTING and value is not None)
    if ok:
        FieldValue("flow.duration", value, status, "test", **kwargs)
    else:
        with pytest.raises(InvariantViolation):
            FieldValue("flow.duration", value, status, "test", **kwargs)


def test_excluded_fields_have_no_weight_only_removal():
    with pytest.raises(InvariantViolation):
        evidence_weight(S.NOT_SUPPLIED, {})
    assert evidence_weight(S.OBSERVED, {}) == 1.0
    with pytest.raises(ConfigMissing):
        evidence_weight(S.STALE, {})
    w = {S.STALE: 0.6, S.LOW_RELIABILITY: 0.3}
    assert evidence_weight(S.STALE, w) == 0.6
    with pytest.raises(InvariantViolation):
        validate_weights({S.STALE: 1.0, S.LOW_RELIABILITY: 0.3})


def test_contributing_mask():
    m = contributing_mask([S.OBSERVED, S.NOT_OBSERVABLE, S.STALE])
    assert m.tolist() == [True, False, True]


def _update(**flds):
    return StateUpdate(
        update_id="u1",
        ordering=OrderingInfo(event_time=1.0, ingest_time=1.1),
        entities=(EntityRef("host", "h1"),),
        fields=flds,
        provenance=Provenance("s", "a", "0"),
    )


def test_state_update_absent_field_is_not_supplied_and_strict_catalogue():
    u = _update(**{"flow.duration": FieldValue("flow.duration", 2.0, S.OBSERVED, "netflow")})
    assert u.status_of("pkt.ttl_mean") is S.NOT_SUPPLIED
    assert list(u.contributing()) == ["flow.duration"]
    with pytest.raises(InvariantViolation):
        _update(**{"made.up": FieldValue("made.up", 1.0, S.OBSERVED, "x")})
    with pytest.raises(TypeError):
        u.fields["x"] = 1  # immutable


def test_required_fields_cover_problem_statement():
    req = set(fields.required_ids())
    for needed in ("flow.tcp_flags", "flow.iat_mean", "flow.iat_var", "flow.iat_max", "flow.bidir_ratio",
                   "pkt.ttl_mean", "pkt.ttl_var", "pkt.retransmissions", "pkt.payload_size_hist",
                   "derived.portscan_sequential", "derived.portscan_random"):
        assert needed in req
    assert fields.by_level(fields.Level.OT), "OT fields are first-class (D-34)"


def test_field_index_reserved_slots():
    idx = FieldIndex(["a", "b"], reserved=1)
    assert idx.capacity == 3 and idx.slot("b") == 1
    assert idx.add("c") == 2
    with pytest.raises(InvariantViolation):
        idx.add("d")
    idx.retire("a")
    with pytest.raises(InvariantViolation):
        idx.slot("a")
