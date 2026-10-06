"""Pipeline plan, split rules, freezing proof, metrics, calibration, forecasting skill, baseline, lab gate."""

import math

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from nagahana.core.errors import InvariantViolation, ProposalNotEnabled
from nagahana.evaluation.baseline import fit_logistic
from nagahana.evaluation.calibration import ece
from nagahana.evaluation.catalogue import CATALOGUE
from nagahana.evaluation.forecasting import concordance_index, lead_time, skill_score
from nagahana.evaluation.metrics import Confusion, f1, fnr, fpr, precision, recall, report
from nagahana.lab.info_audit import mutual_information
from nagahana.pipeline import stages
from nagahana.pipeline.freezing import assert_unchanged, fingerprint, freeze
from nagahana.pipeline.splits import Novelty, Origin, Sample, Split, validate


def test_six_stages_and_frozen_perception_in_stage_four():
    assert [s.id for s in stages.STAGES] == [1, 2, 3, 4, 5, 6]
    s4 = stages.get("pretrain-taaft")
    assert set(s4.frozen) == {"cvgae", "decoder", "tstct"} and s4.trains == ("taaft",)
    assert stages.get(6).splits == ("zero_shot_known", "zero_shot_novel")


def _ok_manifest():
    return [
        Sample("r1", Origin.REAL, Split.TRAIN, "dos", "net-a"),
        Sample("g1", Origin.GENERATED, Split.TRAIN, "dos", "net-a", derived_from="r1"),
        Sample("r2", Origin.REAL, Split.VAL, "benign", "net-a"),
        Sample("z1", Origin.REAL, Split.ZERO_SHOT, "dos", "net-b", novelty=Novelty.KNOWN),
        Sample("z2", Origin.REAL, Split.ZERO_SHOT, "exfil", "net-b", novelty=Novelty.NOVEL),
    ]


def test_split_rules():
    validate(_ok_manifest(), enabled_proposals=("P-23",), generator_training_ids=("r1",))
    bad = _ok_manifest() + [Sample("z3", Origin.GENERATED, Split.ZERO_SHOT, "dos", "net-b", novelty=Novelty.KNOWN, derived_from="r1")]
    with pytest.raises(InvariantViolation, match="real"):
        validate(bad)
    bad = _ok_manifest() + [Sample("z4", Origin.REAL, Split.ZERO_SHOT, "dos", "net-b", novelty=Novelty.NOVEL)]
    with pytest.raises(InvariantViolation, match="novel"):
        validate(bad)
    with pytest.raises(InvariantViolation, match="P-23"):
        validate(_ok_manifest(), enabled_proposals=("P-23",), generator_training_ids=("z1",))
    validate(_ok_manifest(), generator_training_ids=("z1",))  # P-23 not enabled: not checked


def test_freeze_fingerprint_detects_changes():
    m = torch.nn.Linear(3, 2)
    before = fingerprint(m)
    freeze([m])
    assert not any(p.requires_grad for p in m.parameters())
    assert_unchanged(m, before, what="perception")
    with torch.no_grad():
        m.weight.add_(1.0)
    with pytest.raises(InvariantViolation):
        assert_unchanged(m, before, what="perception")


def test_metrics_known_values_and_undefined_as_nan():
    c = Confusion(tp=8, fp=2, tn=85, fn=5)
    assert precision(c) == 0.8 and recall(c) == pytest.approx(8 / 13)
    assert fpr(c) == pytest.approx(2 / 87) and fnr(c) == pytest.approx(5 / 13)
    assert f1(c) == pytest.approx(2 * 0.8 * (8 / 13) / (0.8 + 8 / 13))
    assert math.isnan(precision(Confusion(0, 0, 10, 3)))
    r = report(c)
    assert r["detection_error"] == pytest.approx(7 / 100) and r["base_rate"] == pytest.approx(13 / 100)


@given(st.lists(st.booleans(), min_size=2, max_size=60))
def test_confusion_counts_add_up(bits):
    y = torch.tensor([int(b) for b in bits])
    pred = torch.tensor([int(not b) if i % 3 == 0 else int(b) for i, b in enumerate(bits)])
    c = Confusion.from_predictions(y, pred)
    assert c.n == len(bits)


def test_calibration_and_forecast_skill():
    p = torch.tensor([0.0, 0.0, 1.0, 1.0])
    y = torch.tensor([0, 0, 1, 1])
    assert ece(p, y, bins=10) == pytest.approx(0.0)
    assert ece(torch.full((4,), 0.9), torch.tensor([0, 0, 0, 1]), bins=10) == pytest.approx(0.65)
    assert skill_score(0.05, 0.10) == pytest.approx(0.5)
    t = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert lead_time(t, torch.tensor([0.1, 0.6, 0.8, 0.9]), threshold=0.5, completion_time=3.0) == 2.0
    assert math.isnan(lead_time(t, torch.tensor([0.1, 0.1, 0.1, 0.9]), threshold=0.5, completion_time=3.0))


def test_concordance_index():
    time = torch.tensor([1.0, 2.0, 3.0, 4.0])
    event = torch.tensor([1, 1, 1, 0])
    assert concordance_index(torch.tensor([4.0, 3.0, 2.0, 1.0]), time, event) == 1.0
    assert concordance_index(torch.tensor([1.0, 2.0, 3.0, 4.0]), time, event) == 0.0


def test_logistic_baseline_learns_a_separable_problem():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(400, 3, generator=g)
    y = (x[:, 0] + 0.5 * x[:, 1] > 0).long()
    m = fit_logistic(x, y, l2=1e-3, max_iter=200)
    acc = (m.predict(x, threshold=0.5) == y).float().mean()
    assert acc > 0.97


def test_catalogue_is_well_formed():
    assert len(CATALOGUE) >= 20
    for m in CATALOGUE:
        assert m.status.startswith(("implemented", "template"))


def test_information_audit_is_gated_and_correct():
    with pytest.raises(ProposalNotEnabled):
        mutual_information([0, 1], [0, 1], enabled_proposals=(), miller_madow=False)
    same = mutual_information([0, 1] * 50, [0, 1] * 50, enabled_proposals=("P-15",), miller_madow=False)
    assert same == pytest.approx(math.log(2))
    indep = mutual_information([0, 0, 1, 1] * 25, [0, 1, 0, 1] * 25, enabled_proposals=("P-15",), miller_madow=False)
    assert indep == pytest.approx(0.0, abs=1e-12)


def test_sizing_counts_the_built_model():
    from nagahana.lab.sizing import LABELS, built_counts, table
    from nagahana.models.nagahana import COMPONENTS

    counts = built_counts()
    assert set(LABELS) == set(COMPONENTS)
    assert counts["total"] == sum(counts[c] for c in COMPONENTS)
    assert all(counts[c] > 0 for c in COMPONENTS)
    assert f"{counts['total']:,}" in table()
