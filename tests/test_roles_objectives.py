"""Output contracts, human-gated Verifier, P_inf estimator, loss template, scores, shaping."""

import math

import pytest
import torch

from nagahana.core.errors import ConfigMissing, HumanCommandRequired, InvariantViolation
from nagahana.core.roles import Role
from nagahana.objectives.rewards import potential_shaping
from nagahana.objectives.scoring import brier, crps_ensemble, log_score
from nagahana.objectives.template import USAGE, ComponentLoss
from nagahana.roles.contracts import (
    AdvisoryBundle,
    AttackStage,
    BeliefReadout,
    ComputeRecord,
    FeedbackKind,
    ForecastBundle,
    HumanCommand,
    HumanFeedback,
    OutcomeForecastPair,
)
from nagahana.roles.forecaster import p_inf_from_first_hits
from nagahana.roles.verifier import Verifier

STAGES = tuple(AttackStage)


def _bundle(p_inf):
    k = len(p_inf)
    row = tuple([1.0] + [0.0] * (len(STAGES) - 1))
    return ForecastBundle(tuple(p_inf), tuple(row for _ in range(k)), STAGES, (), (), ComputeRecord(8, k, 0, 0.1))


def test_forecast_bundle_invariants():
    _bundle([0.1, 0.2, 0.2])
    with pytest.raises(InvariantViolation):
        _bundle([0.3, 0.2])       # P_inf must not decrease
    with pytest.raises(InvariantViolation):
        _bundle([1.2])


def test_advisory_only_and_suspicion_floor():
    with pytest.raises(InvariantViolation):
        AdvisoryBundle((), advisory_only=False)
    with pytest.raises(InvariantViolation):
        BeliefReadout({}, 0.0, {}, {}, {})


def test_p_inf_estimator():
    tau = torch.tensor([1, 3, 3, 99])            # 99 = never within K
    p = p_inf_from_first_hits(tau, 4)
    assert p.tolist() == [0.25, 0.25, 0.75, 0.75]
    assert (p[1:] >= p[:-1]).all()


def test_verifier_changes_nothing_without_a_human_command():
    v = Verifier()
    with pytest.raises(HumanCommandRequired):
        v.apply_update(lambda: "changed", command=None)
    cmd = HumanCommand("analyst-1", "apply-calibration", "weekly review", 0.0)
    assert v.apply_update(lambda: "changed", command=cmd) == "changed"
    assert v.command_log() == (cmd,)
    with pytest.raises(InvariantViolation):
        HumanCommand("", "x", "y", 0.0)


def test_responded_to_outcomes_are_not_penalised():
    v = Verifier()
    v.record_outcome(OutcomeForecastPair("f1", 0.9, occurred=False, responded_to=True))
    v.record_outcome(OutcomeForecastPair("f2", 0.8, occurred=True))
    s = v.calibration_summary()
    assert s.n_scored == 1 and s.n_responded_to == 1 and math.isclose(s.brier, 0.04)
    v.record_feedback(HumanFeedback(FeedbackKind.STEP_LABEL, "f2/step3", "implausible", "analyst-1", 0.0))
    assert len(v.feedback(FeedbackKind.STEP_LABEL)) == 1


def test_loss_template_follows_the_matrix():
    assert USAGE[Role.DECODER] == frozenset()
    with pytest.raises(InvariantViolation):
        ComponentLoss(Role.DECODER, {"physics": 1.0})
    with pytest.raises(ConfigMissing):
        ComponentLoss(Role.FORECASTER, {"physics": 1.0})
    loss = ComponentLoss(Role.SIMULATOR, {"physics": 0.5})
    total, parts = loss(torch.tensor(2.0), {"physics": torch.tensor(4.0)})
    assert float(total) == 4.0 and float(parts["physics"]) == 2.0


def test_proper_scores():
    assert float(brier(torch.tensor(1.0), torch.tensor(1))) == 0.0
    assert float(log_score(torch.tensor(0.5), torch.tensor(1), eps=1e-6)) == pytest.approx(math.log(2))
    # CRPS of a point-mass ensemble equals absolute error
    assert float(crps_ensemble(torch.full((5,), 3.0), torch.tensor(1.0))) == pytest.approx(2.0)
    # a spread ensemble centred on the truth scores better than a biased one
    good = crps_ensemble(torch.tensor([0.9, 1.0, 1.1]), torch.tensor(1.0))
    bad = crps_ensemble(torch.tensor([1.9, 2.0, 2.1]), torch.tensor(1.0))
    assert good < bad


def test_potential_shaping_terminal_choice_is_explicit():
    phi_s, phi_n = torch.tensor([1.0, 1.0]), torch.tensor([3.0, 3.0])
    term = torch.tensor([False, True])
    kept = potential_shaping(phi_s, phi_n, gamma=0.5, next_is_terminal=term, zero_terminal_potential=False)
    zeroed = potential_shaping(phi_s, phi_n, gamma=0.5, next_is_terminal=term, zero_terminal_potential=True)
    assert kept.tolist() == [0.5, 0.5] and zeroed.tolist() == [0.5, -1.0]
