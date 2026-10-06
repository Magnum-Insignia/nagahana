"""Governance: held decisions resolve to recorded options, decided ones pass, proposals are opt-in, the
registry builds under the option in force, and the report finds usages."""

from __future__ import annotations

import pytest

from nagahana.core.errors import DecisionHeld, InvalidOption, ProposalNotEnabled
from nagahana.core.registry import Registry
from nagahana.governance import assumptions, decisions, report


@pytest.fixture()
def strict() -> object:
    assumptions.strict_mode(True)
    try:
        yield None
    finally:
        assumptions.strict_mode(False)


def test_ids_and_slugs_unique_and_resolvable() -> None:
    for d in decisions.all_entries():
        assert decisions.get(d.id) is d
        assert decisions.get(d.slug) is d


def test_held_decision_resolves_to_its_working_option_and_records_the_assumption() -> None:
    d = decisions.require("taaft-policy-coupling", by="tests.governance")
    assert d.status is decisions.Status.HELD and d.id == "D-12"
    assert d.value == decisions.get("D-12").working == "staged"
    assert "tests.governance" in assumptions.uses()["AS-22"]
    assert ("staged", "working option (AS-22)") in decisions.resolutions()["D-12"]


def test_strict_mode_blocks_working_options_but_not_configured_ones(strict: object) -> None:
    with pytest.raises(DecisionHeld, match="AS-22"):
        decisions.require("D-12")
    with decisions.configure({"D-12": "joint"}):
        assert decisions.require("D-12").value == "joint"


def test_held_decision_without_working_option_needs_a_configured_option() -> None:
    assert decisions.get("D-28").working is None
    with pytest.raises(DecisionHeld, match="Streamlit"):
        decisions.require("demo-interface")
    with decisions.configure({"demo-interface": "Streamlit"}) as merged:
        assert merged == {"D-28": "Streamlit"}
        assert decisions.require("D-28").value == "Streamlit"
    assert ("Streamlit", "configured") in decisions.resolutions()["D-28"]
    with pytest.raises(DecisionHeld):
        decisions.require("D-28")


def test_configure_validates_every_option() -> None:
    with pytest.raises(InvalidOption, match="not admissible"), decisions.configure({"D-12": "sometimes"}):
        pass
    with pytest.raises(InvalidOption, match="decided"), decisions.configure({"advisor-reads-environment": "yes"}):
        pass
    with pytest.raises(InvalidOption, match="proposed"), decisions.configure({"P-18": "on"}):
        pass
    with pytest.raises(KeyError), decisions.configure({"D-999": "x"}):
        pass
    with pytest.raises(InvalidOption, match="not admissible"):
        decisions.require("D-12", "sometimes")


def test_configuration_is_scoped_nests_and_the_caller_option_wins() -> None:
    assert decisions.configured_options() == {}
    with decisions.configure({"D-12": "joint"}):
        with decisions.configure({"D-24": "Advisor"}) as inner:
            assert inner == {"D-12": "joint", "D-24": "Advisor"}
            assert decisions.require("D-12", "separate").value == "separate"
            assert decisions.option_in_force("D-24") == "Advisor"
        assert decisions.configured_options() == {"D-12": "joint"}
    assert decisions.configured_options() == {}
    assert decisions.option_in_force("D-12") == "staged"


def test_decided_decision_passes_and_takes_no_option() -> None:
    d = decisions.require("advisor-reads-environment")
    assert d.status is decisions.Status.DECIDED and d.value
    with pytest.raises(InvalidOption):
        decisions.require("advisor-reads-environment", "anything")
    with pytest.raises(InvalidOption):
        decisions.option_in_force("advisor-reads-environment")


def test_proposal_is_not_a_decision_and_is_opt_in() -> None:
    with pytest.raises(DecisionHeld, match="PROPOSAL"):
        decisions.require("P-18")
    with pytest.raises(ProposalNotEnabled):
        decisions.require_proposal("P-18", enabled=())
    assert decisions.require_proposal("P-18", enabled=("P-18",)).id == "P-18"
    assert decisions.require_proposal("kvcache-as-view", enabled=("kvcache-as-view",)).id == "P-18"
    assert decisions.require_proposal("D-12", enabled=()).id == "D-12" or decisions.get("D-12").status is not decisions.Status.DECIDED


def test_unknown_key_raises() -> None:
    with pytest.raises(KeyError):
        decisions.get("D-999")


def test_registry_entries_are_consistent() -> None:
    for d in decisions.all_entries():
        assert d.sources, d.id
        if d.status is decisions.Status.DECIDED:
            assert d.value, d.id
            assert d.working is None and d.assumption is None, d.id
        if d.working is not None:
            assert d.status is decisions.Status.HELD, d.id
            assert d.working in d.admissible and d.working in d.options + (d.working,), d.id
            assert assumptions.get(str(d.assumption)).id == d.assumption, d.id
    held_without = {d.id for d in decisions.by_status(decisions.Status.HELD) if d.working is None}
    assert held_without == {"D-06", "D-08", "D-10", "D-27", "D-28"}
    assert set(decisions.get("D-27").options) == {"confluent-kafka", "aiokafka", "kafka-python"}


def test_recorded_entries_are_present() -> None:
    for key in ("D-55", "D-56", "P-14"):
        decisions.get(key)
    decided = {f"D-{n}" for n in range(57, 69)}
    assert all(decisions.get(k).status is decisions.Status.DECIDED for k in decided)
    assert decisions.get("D-61").slug == "optimizer-hybrid-muon" and decisions.get("D-61").affects == ("training",)
    slugs = {"D-62": "stage-numbering", "D-63": "site-sized-memory", "D-64": "parameter-count-follows-datamodel",
             "D-65": "verifier-rlhf-rlvr", "D-66": "attention-outputs", "D-67": "pyshark-adapter",
             "D-68": "catboost-baseline"}
    assert {k: decisions.get(k).slug for k in slugs} == slugs
    assert decisions.get("D-28").status is decisions.Status.HELD
    assert "Please confirm" not in (decisions.get("D-09").note or "") + (decisions.get("D-09").value or "")


def test_settings_snapshot_lists_every_held_entry() -> None:
    snap = decisions.settings_snapshot()
    assert set(snap) == {d.id for d in decisions.by_status(decisions.Status.HELD)}
    assert snap["D-12"] == "staged" and snap["D-28"] == "unresolved"
    with decisions.configure({"D-28": "CLI"}):
        assert decisions.settings_snapshot()["D-28"] == "CLI"
    assert decisions.working_options()["D-24"] == "TAAFT"


def test_registry_builds_under_the_option_in_force() -> None:
    reg: Registry[object] = Registry("test")

    @reg.register("needs-held", requires=("D-12",))
    def _a() -> str:
        return "built"

    @reg.register("needs-unresolved", requires=("D-28",))
    def _b() -> str:
        return "built"

    @reg.register("advisor-only", requires={"D-24": ("Advisor",)})
    def _c() -> str:
        return "built"

    @reg.register("needs-proposal", proposal="P-18")
    def _d() -> str:
        return "built"

    @reg.register("free")
    def _e() -> str:
        return "built"

    assert reg.build("needs-held") == "built"
    with pytest.raises(DecisionHeld):
        reg.build("needs-unresolved")
    with decisions.configure({"D-28": "CLI"}):
        assert reg.build("needs-unresolved") == "built"
    with pytest.raises(InvalidOption, match="TAAFT"):
        reg.build("advisor-only")
    with decisions.configure({"D-24": "Advisor"}):
        assert reg.build("advisor-only") == "built"
    with pytest.raises(ProposalNotEnabled):
        reg.build("needs-proposal")
    assert reg.build("needs-proposal", enabled_proposals=("P-18",)) == "built"
    assert reg.build("free") == "built"
    assert reg.entry("advisor-only").options == {"D-24": frozenset({"Advisor"})}


def test_registry_rejects_unknown_ids_options_and_duplicates() -> None:
    reg: Registry[object] = Registry("test")
    with pytest.raises(KeyError):
        reg.register("x", requires=("D-999",))
    with pytest.raises(InvalidOption, match="not admissible"):
        reg.register("y", requires={"D-24": ("Nowhere",)})
    with pytest.raises(InvalidOption, match="held decisions only"):
        reg.register("z", requires={"advisor-reads-environment": ("yes",)})
    reg.register("dup")(lambda: None)
    with pytest.raises(ValueError):
        reg.register("dup")(lambda: None)


def test_report_finds_code_usages() -> None:
    use = report.usages()
    assert any("memory/retention.py" in p for p in use.get("D-02", []))
    assert "D-12" in use
    md = report.markdown()
    assert "## Held" in md and "D-12" in md
