"""Models: variational maths, latent contract, reference CVG-AE equivariance, TSTCT masks, energy descent, coupling."""

import pytest
import torch

from nagahana.core.errors import DecisionHeld, InvariantViolation, ModeViolation, NotBuiltYet
from nagahana.core.modes import RunMode, run_mode
from nagahana.graph.hypergraph import HypergraphSnapshot
from nagahana.models.cvgae.model import ENCODERS
from nagahana.models.generator.model import GENERATORS, PhysicsBoundedSampler
from nagahana.models.heads.policy_value import Coupling, decided_coupling, head_input
from nagahana.models.latent import LatentState
from nagahana.models.taaft.energy import energy_descent
from nagahana.models.taaft.lenses import LENSES
from nagahana.models.tstct.masks import spatial_allowed, temporal_allowed, to_additive
from nagahana.models.variational import kl_categorical, kl_diag_gaussian, reparameterize, straight_through_categorical


def test_kl_and_reparameterisation():
    mean, logvar = torch.zeros(4, 3), torch.zeros(4, 3)
    assert torch.allclose(kl_diag_gaussian(mean, logvar), torch.zeros(4))
    g = torch.Generator().manual_seed(0)
    z = reparameterize(torch.full((20000, 1), 2.0), torch.full((20000, 1), 0.0), generator=g)
    assert abs(float(z.mean()) - 2.0) < 0.05 and abs(float(z.std()) - 1.0) < 0.05
    logits = torch.randn(5, 4)
    assert torch.allclose(kl_categorical(logits, logits), torch.zeros(5), atol=1e-6)


def test_straight_through_is_one_hot_forward_with_gradient():
    logits = torch.randn(6, 2, 5, requires_grad=True)
    s = straight_through_categorical(logits, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(s.detach().sum(-1), torch.ones(6, 2))
    (s * torch.arange(5.0)).sum().backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_latent_contract():
    LatentState(torch.zeros(3, 4), torch.zeros(3, 2, 5), "z-v0")
    with pytest.raises(InvariantViolation):
        LatentState(torch.zeros(3, 4), torch.zeros(2, 2, 5), "z-v0")
    with pytest.raises(InvariantViolation):
        LatentState(torch.zeros(3, 4), torch.zeros(3, 2, 5), "")


def _snapshot(perm=None):
    kinds = ("host", "host", "service", "account")
    inc_p = torch.tensor([[0, 1, 2, 1, 3], [0, 0, 0, 1, 1]])
    inc_q = torch.tensor([[0, 3], [0, 0]])
    if perm is not None:
        inv = torch.argsort(perm)
        kinds = tuple(kinds[i] for i in perm.tolist())
        inc_p = torch.stack([inv[inc_p[0]], inc_p[1]])
        inc_q = torch.stack([inv[inc_q[0]], inc_q[1]])
    return HypergraphSnapshot(0.0, kinds, ("p", "q"), {"p": inc_p, "q": inc_q},
                              {"p": ("session", "scan"), "q": ("auth",)})


def test_reference_cvgae_is_permutation_equivariant():
    torch.manual_seed(0)
    enc = ENCODERS.build("hgnn-mean-reference", planes=("p", "q"), in_dim=3, dim=8, layers=2,
                         node_kinds=("host", "service", "account"),
                         edge_kinds={"p": ("session", "scan"), "q": ("auth",)},
                         cont_dim=4, disc_groups=2, disc_classes=3, latent_space="z-test")
    with torch.no_grad():
        enc.omega.fill_(0.3)
    x = torch.randn(4, 3)
    perm = torch.tensor([2, 0, 3, 1])
    z = enc(_snapshot(), x)
    zp = enc(_snapshot(perm), x[perm])
    assert z.mean.shape == (4, 4) and z.logits.shape == (4, 2, 3) and z.space == "z-test"
    assert torch.allclose(zp.mean, z.mean[perm], atol=1e-5)
    assert torch.allclose(zp.logits, z.logits[perm], atol=1e-5)
    with pytest.raises(DecisionHeld):
        enc.loss_terms()


def test_temporal_mask_never_sees_the_future():
    ent = torch.tensor([0, 1, 0, 0, 1])
    t = torch.tensor([1.0, 2.0, 3.0, 5.0, 4.0])
    a = temporal_allowed(ent, t)
    future = t[None, :] > t[:, None]
    assert not (a & future).any()
    assert a.diagonal().all()
    m = to_additive(a)
    assert torch.isinf(m[0, 2]) and m[2, 0] == 0
    s = spatial_allowed(torch.zeros(3, 3, dtype=torch.bool))
    assert s.equal(torch.eye(3, dtype=torch.bool))
    with pytest.raises(InvariantViolation):
        to_additive(torch.zeros(2, 2, dtype=torch.bool))


def test_energy_descent_finds_the_minimum():
    target = torch.tensor([1.0, -2.0])
    y, trace = energy_descent(lambda y: 0.5 * ((y - target) ** 2).sum(-1), torch.zeros(2),
                              steps=200, step_size=0.1, noise=lambda i: 0.0)
    assert torch.allclose(y, target, atol=1e-4)
    assert trace[0] > trace[-1]


def test_coupling_semantics_and_held_choice():
    f = torch.ones(2, requires_grad=True)
    assert head_input(f, Coupling.JOINT).requires_grad
    assert not head_input(f, Coupling.SEPARATE).requires_grad
    assert not head_input(f, Coupling.STAGED).requires_grad
    assert head_input(f, Coupling.STAGED, joint_phase=True).requires_grad
    with pytest.raises(DecisionHeld):
        decided_coupling()


def test_generator_and_lens_gates():
    with pytest.raises(DecisionHeld):
        GENERATORS.build("diffusion")
    with pytest.raises(DecisionHeld):
        LENSES.build("noise")
    with pytest.raises(NotBuiltYet):
        LENSES.build("belief-trust").analyse(None, None)
    sampler = PhysicsBoundedSampler(family=None)
    with run_mode(RunMode.INFER_LIVE), pytest.raises(ModeViolation):
        sampler.sample(None, None)
    assert sampler.hard_gate_enabled() is False
    assert PhysicsBoundedSampler(family=None, enabled_proposals=("P-11",)).hard_gate_enabled() is True
