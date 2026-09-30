"""Physics boundary: zero inside, positive outside, never fires on absent data; hypergraph validity."""

import math

import pytest
import torch

from nagahana.core.errors import ConfigMissing, DecisionHeld, InvariantViolation, NotBuiltYet
from nagahana.graph.hypergraph import HypergraphSnapshot
from nagahana.physics.constraints import ordered_pair
from nagahana.physics.residuals import RESIDUALS, FlagCountBound, IATMaxBound, MTUBound
from nagahana.physics.term import PhysicsTerm, Target


def _vals(**kw):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in kw.items()}


def test_residuals_zero_inside_positive_outside():
    x = _vals(**{"flow.flag_count.syn": [3.0, 9.0], "flow.packets_fwd": [2.0, 2.0], "flow.packets_bwd": [2.0, 2.0]})
    r = FlagCountBound("syn")(x)
    assert r.tolist() == [0.0, 5.0]
    x = _vals(**{"flow.bytes_fwd": [1500.0, 4000.0], "flow.packets_fwd": [1.0, 2.0]})
    assert MTUBound("fwd", mtu=1500)(x).tolist() == [0.0, 1000.0]
    x = _vals(**{"flow.iat_max": [1.0, 5.0], "flow.duration": [2.0, 3.0]})
    assert IATMaxBound()(x).tolist() == [0.0, 2.0]


def test_term_masks_absent_data_even_if_nan_and_gradients_stay_finite():
    iat = torch.tensor([5.0, float("nan")], requires_grad=True)
    dur = torch.tensor([3.0, 1.0])
    term = PhysicsTerm([IATMaxBound()], {"iat_max_bound": 2.0}, target=Target.MODEL_OUTPUT)
    contributing = {"flow.iat_max": torch.tensor([True, False]), "flow.duration": torch.tensor([True, True])}
    total, parts = term({"flow.iat_max": iat, "flow.duration": dur}, contributing)
    assert math.isclose(float(total.detach()), 2.0 * 4.0)  # w · (5 − 3)², second row masked
    total.backward()
    assert torch.isfinite(iat.grad).all()


def test_term_requires_weights_and_holds_observation_use():
    with pytest.raises(ConfigMissing):
        PhysicsTerm([IATMaxBound()], {}, target=Target.MODEL_OUTPUT)
    with pytest.raises(DecisionHeld):
        PhysicsTerm([IATMaxBound()], {"iat_max_bound": 1.0}, target=Target.OBSERVATION)


def test_template_residuals_raise():
    with pytest.raises(NotBuiltYet):
        RESIDUALS.build("flow_conservation")


def test_hard_limit_ordered_pair():
    lo, hi = ordered_pair(torch.tensor([-3.0, 2.0]), torch.tensor([-5.0, 0.0]))
    assert (hi >= lo).all() and (lo >= 0).all()


def test_hypergraph_validation():
    inc = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    g = HypergraphSnapshot(0.0, ("host", "host", "service"), ("p",), {"p": inc}, {"p": ("session", "session")})
    assert g.num_hyperedges("p") == 2
    assert g.dense_incidence("p").sum().item() == 4
    with pytest.raises(InvariantViolation):  # hyperedge 1 has only one distinct member
        HypergraphSnapshot(0.0, ("host", "host"), ("p",), {"p": torch.tensor([[0, 1, 1], [0, 0, 1]])})
    with pytest.raises(InvariantViolation):  # node index out of range
        HypergraphSnapshot(0.0, ("host",), ("p",), {"p": torch.tensor([[0, 3], [0, 0]])})
