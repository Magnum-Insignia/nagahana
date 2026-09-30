"""Governance: held decisions block, decided ones pass, proposals are opt-in, the report finds usages."""

import pytest

from nagahana.core.errors import DecisionHeld, ProposalNotEnabled
from nagahana.core.registry import Registry
from nagahana.governance import decisions, report


def test_ids_and_slugs_unique_and_resolvable():
    for d in decisions.all_entries():
        assert decisions.get(d.id) is d
        assert decisions.get(d.slug) is d


def test_held_decision_raises_with_instructions():
    with pytest.raises(DecisionHeld, match="D-12"):
        decisions.require("taaft-policy-coupling")


def test_decided_decision_passes():
    d = decisions.require("advisor-reads-environment")
    assert d.status is decisions.Status.DECIDED and d.value


def test_proposal_is_not_a_decision_and_is_opt_in():
    with pytest.raises(DecisionHeld):
        decisions.require("P-18")
    with pytest.raises(ProposalNotEnabled):
        decisions.require_proposal("P-18", enabled=())
    assert decisions.require_proposal("P-18", enabled=("P-18",)).id == "P-18"
    assert decisions.require_proposal("kvcache-as-view", enabled=("kvcache-as-view",)).id == "P-18"


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        decisions.get("D-999")


def test_every_decided_entry_has_a_value_and_every_entry_a_source():
    for d in decisions.all_entries():
        if d.status is decisions.Status.DECIDED:
            assert d.value, d.id
        assert d.sources, d.id


def test_registry_gates_on_held_decisions_and_proposals():
    reg: Registry = Registry("test")

    @reg.register("needs-held", requires=("D-12",))
    def _a():
        return "built"

    @reg.register("needs-proposal", proposal="P-18")
    def _b():
        return "built"

    @reg.register("free")
    def _c():
        return "built"

    with pytest.raises(DecisionHeld):
        reg.build("needs-held")
    with pytest.raises(ProposalNotEnabled):
        reg.build("needs-proposal")
    assert reg.build("needs-proposal", enabled_proposals=("P-18",)) == "built"
    assert reg.build("free") == "built"


def test_registry_rejects_unknown_ids_at_registration_and_duplicates():
    reg: Registry = Registry("test")
    with pytest.raises(KeyError):
        reg.register("x", requires=("D-999",))
    reg.register("dup")(lambda: None)
    with pytest.raises(ValueError):
        reg.register("dup")(lambda: None)


def test_report_finds_code_usages():
    use = report.usages()
    assert any("memory/retention.py" in p for p in use.get("D-02", []))
    assert "D-12" in use
    md = report.markdown()
    assert "## Held" in md and "D-12" in md
