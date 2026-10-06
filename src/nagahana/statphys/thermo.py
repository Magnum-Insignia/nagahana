"""Thermodynamic readouts of the energy view: Gibbs ensembles and their potentials (D-56, D-42, D-54).

Purpose
-------
TAAFT's energy E_total = sum_l E_l + lambda * Phi_phys (D-42) is measured in nats: exp(-E_total) is the
unnormalised product of the lens experts (lenses.py). Over any finite ensemble of imagined states or
routes with energies E_i this module forms the canonical (Gibbs) distribution at a temperature T and
reads out its thermodynamic potentials. Architecture section 6 lists "Energy per entity and per
time", "Energy and entropy growth as an early warning" and "Novelty (high energy means unfamiliar,
not proof of attack)"; the free energy below is the novelty reading of an ensemble, and the growth
rates at the end of this module are the "growth" of that list.

Mathematics
-----------
Members i = 1 ... n of an ensemble (a boolean mask selects them), energies E_i, a base measure g_i > 0
(degeneracy or multiplicity of the member; 1 by default) and a temperature T > 0:

    a_i       = -E_i / T + log g_i
    log Z     = logsumexp_i a_i = m + log sum_i exp(a_i - m),  m = max_i a_i       (no overflow)
    p_i       = exp(a_i - log Z)                                 Boltzmann occupation
    F         = -T log Z                                         free energy
    U         = sum_i p_i E_i                                    mean (internal) energy
    S         = (U - F) / T = log Z + U / T                      Gibbs entropy
    C         = Var_p(E) / T^2                                   heat capacity
    chi_O     = Var_p(O) / T                                     susceptibility of an observable O
    chi_OB    = Cov_p(O, B) / T                                  cross-susceptibility
    d<O>/dT   = Cov_p(O, E) / T^2                                thermal response of <O>

Why these identities hold (each is checked against finite differences in the tests):
- S: with p_i = g_i exp(-E_i / T) / Z, log(p_i / g_i) = -E_i / T - log Z, hence
  -sum_i p_i log(p_i / g_i) = U / T + log Z. With g = 1 this is the Shannon entropy of p, so
  0 <= S <= log n; with multiplicities g it counts the degenerate microstates (Boltzmann).
- C = dU/dT: differentiating U = sum_i E_i g_i exp(-E_i / T) / Z gives (<E^2> - <E>^2) / T^2, and
  dF/dT = -S (the Gibbs-Helmholtz relation).
- chi_O: for the perturbed energy E_i - h O_i, d<O>/dh at h = 0 equals (<O^2> - <O>^2) / T, the
  static fluctuation-dissipation relation (Kubo, "The fluctuation-dissipation theorem", Reports on
  Progress in Physics 29(1):255, 1966); d<O>/dT = Cov(O, E) / T^2 by the same differentiation.
Textbook source for the canonical ensemble: Callen, "Thermodynamics and an Introduction to
Thermostatistics", 2nd ed., Wiley 1985, chapters 16 and 19. The free energy of an energy-based model
as a novelty (out-of-distribution) score: Liu, Wang, Owens and Li, "Energy-based Out-of-distribution
Detection", NeurIPS 2020 (arXiv:2010.03759); Grathwohl et al., "Your Classifier is Secretly an Energy
Based Model and You Should Treat it Like One", ICLR 2020 (arXiv:1912.03263). Energy-based models in
general: LeCun, Chopra, Hadsell, Ranzato and Huang, "A Tutorial on Energy-Based Learning", in
Predicting Structured Data, MIT Press 2006.

Numerics
--------
- log Z by the max-shifted log-sum-exp; p by softmax of the same shifted arguments.
- Variances are two-pass (sum_i p_i (E_i - U)^2), never <E^2> - <E>^2, which cancels catastrophically
  when |U| is large against the spread.
- Members outside the mask never reach arithmetic: their energies are replaced by 0 before any
  operation, so a NaN or an infinity in a masked slot cannot leak into values or gradients.
- An ensemble with no member has no Gibbs distribution: `valid` is False and every scalar of it is
  `fill` (NaN by default; TAAFT's readouts pass 0.0 so that training graphs stay finite, AS-762).
- The outputs are differentiable in the energies and the observables (torch autograd).

Precision (D-54): inputs are cast to float64 first; every output is float64.

Growth rates (AS-763)
---------------------
The growth rate of a trajectory x(t) at a valid point m is the least-squares slope of x on t over the
last w valid points up to and including m (w = `GibbsConfig.growth_window`; w = 2 is the backward
difference (x_m - x_m') / (t_m - t_m')):

    slope = sum_j (t_j - tbar)(x_j - xbar) / sum_j (t_j - tbar)^2

computed centred (no cancellation for large t). `trailing_slope` evaluates it for a batch of
trajectories, `SlopeTracker` online with the same value.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import torch

TemperatureLike = float | torch.Tensor


def _temperature(temperature: TemperatureLike, like: torch.Tensor) -> torch.Tensor:
    # Validate T (finite, > 0) and return it as a float64 tensor on the energies' device.
    t = torch.as_tensor(temperature, dtype=torch.float64, device=like.device)
    if not bool(torch.isfinite(t).all()) or bool((t <= 0).any()):
        raise ValueError("temperature must be finite and > 0")
    return t


@dataclass(frozen=True)
class GibbsState:
    """Canonical-ensemble readout over the last axis of an energy tensor (module docstring).

    Ensemble shape [...]: `temperature`, `log_partition`, `free_energy`, `mean_energy`, `entropy`,
    `heat_capacity` (float64, `fill` where the ensemble is empty), `size` (int64 members) and `valid`
    (bool, size > 0). Member shape [..., n]: `occupation` p_i (float64, 0 outside the ensemble).
    """

    temperature: torch.Tensor
    log_partition: torch.Tensor
    free_energy: torch.Tensor
    mean_energy: torch.Tensor
    entropy: torch.Tensor
    heat_capacity: torch.Tensor
    occupation: torch.Tensor
    size: torch.Tensor
    valid: torch.Tensor

    def scalars(self) -> dict[str, float]:
        """The potentials of a single ensemble (0-dimensional state) as Python floats (IEEE double)."""
        if self.log_partition.dim() != 0:
            raise ValueError("scalars() needs a single ensemble (0-dimensional potentials)")
        return {
            "temperature": float(self.temperature), "size": float(self.size),
            "log_partition": float(self.log_partition), "free_energy": float(self.free_energy),
            "mean_energy": float(self.mean_energy), "entropy": float(self.entropy),
            "heat_capacity": float(self.heat_capacity),
        }


def _members(energy: torch.Tensor, mask: torch.Tensor | None, log_measure: torch.Tensor | None,
             check_finite: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Common input handling: float64 energies, the member mask (zero measure = not a member) and log g.
    e = energy.to(torch.float64)
    m = torch.ones_like(e, dtype=torch.bool) if mask is None else mask.to(torch.bool).expand_as(e)
    if log_measure is None:
        lw = torch.zeros_like(e)
    else:
        lw = log_measure.to(torch.float64).expand_as(e)
        if check_finite and bool((m & (torch.isnan(lw) | (lw == math.inf))).any()):
            raise ValueError("log_measure must be finite (or -inf for a member of zero measure)")
        m = m & (lw > -math.inf)
    if check_finite and bool((m & ~torch.isfinite(e)).any()):
        raise ValueError("every member of a Gibbs ensemble needs a finite energy")
    return e, m, lw


def gibbs(
    energy: torch.Tensor,
    *,
    temperature: TemperatureLike,
    mask: torch.Tensor | None = None,
    log_measure: torch.Tensor | None = None,
    fill: float = math.nan,
    check_finite: bool = True,
) -> GibbsState:
    """Gibbs readout of the ensembles laid out on the last axis of `energy` (module docstring).

    energy [..., n]; temperature: a float or a float64 tensor broadcastable to [...]; mask bool
    [..., n] (members; default all); log_measure [..., n] log g_i (default 0; -inf = not a member);
    fill: value of the scalars of an empty ensemble; check_finite: raise on a non-finite member energy.
    """
    e, m, lw = _members(energy, mask, log_measure, check_finite)
    t = _temperature(temperature, e)
    tt = t.unsqueeze(-1)                                                       # broadcast over the member axis
    e0 = torch.where(m, e, torch.zeros_like(e))                                # sanitised energies   [..., n]
    lw0 = torch.where(m, lw, torch.zeros_like(lw))
    size = m.sum(-1)
    valid = size > 0
    a = torch.where(m, -e0 / tt + lw0, torch.full_like(e0, -math.inf))
    a = torch.where(valid.unsqueeze(-1), a, torch.zeros_like(a))               # empty ensembles: a finite dummy row
    log_z = torch.logsumexp(a, dim=-1)                                          # [...]
    p = torch.softmax(a, dim=-1) * m.to(e0.dtype)                              # exact zeros outside the ensemble
    u = (p * e0).sum(-1)
    var = (p * (e0 - u.unsqueeze(-1)) ** 2).sum(-1)                             # two-pass variance
    f = -t * log_z
    s = log_z + u / t
    c = var / t**2

    def keep(x: torch.Tensor) -> torch.Tensor:
        return torch.where(valid, x, torch.full_like(x, fill))

    return GibbsState(temperature=t.expand_as(log_z), log_partition=keep(log_z), free_energy=keep(f),
                      mean_energy=keep(u), entropy=keep(s), heat_capacity=keep(c), occupation=p, size=size,
                      valid=valid)


def gibbs_grouped(
    energy: torch.Tensor,
    group: torch.Tensor,
    n_groups: int,
    *,
    temperature: TemperatureLike,
    mask: torch.Tensor | None = None,
    log_measure: torch.Tensor | None = None,
    fill: float = math.nan,
    check_finite: bool = True,
) -> GibbsState:
    """Gibbs readouts of several ensembles that share one member axis: member i belongs to group[i].

    energy, group (long, -1 = no group), mask, log_measure: [..., n]; n_groups G; temperature: a float
    or a 0-dimensional tensor. Returns potentials of shape [..., G] and occupations [..., n] (each member's
    p within its own group). Equal to `gibbs` applied to each group separately (tested).
    The per-group maximum used as the log-sum-exp shift is detached: log Z does not depend on the shift,
    so the gradient is exact (d log Z / d a_i = p_i) and needs no gradient of a maximum.
    """
    if n_groups < 0:
        raise ValueError("n_groups must be >= 0")
    e, m, lw = _members(energy, mask, log_measure, check_finite)
    t = _temperature(temperature, e)
    if t.dim() != 0:
        raise ValueError("gibbs_grouped takes one temperature (a float or a 0-dimensional tensor)")
    g = group.to(torch.long).expand_as(e)
    m = m & (g >= 0) & (g < n_groups)
    lead = e.shape[:-1]
    r = int(math.prod(lead))
    n = e.shape[-1]
    ef, mf, lwf, gf = e.reshape(r, n), m.reshape(r, n), lw.reshape(r, n), g.reshape(r, n)
    # Global segment id per member: row * G + group; non-members go to one trash segment r * G.
    rows = torch.arange(r, device=e.device).unsqueeze(-1).expand(r, n)
    seg = torch.where(mf, rows * n_groups + gf, torch.full_like(gf, r * n_groups))
    n_seg = r * n_groups + 1
    # Sanitised arguments: every non-member gets a = 0 and is excluded by the masks below, so no
    # -inf or NaN ever enters exp() and no 0 * inf reaches a gradient.
    e0 = torch.where(mf, ef, torch.zeros_like(ef))
    a = torch.where(mf, -e0 / t + torch.where(mf, lwf, torch.zeros_like(lwf)), torch.zeros_like(e0))
    on = mf.reshape(-1)
    flat_seg, flat_a, flat_e = seg.reshape(-1), a.reshape(-1), e0.reshape(-1)
    shift = torch.full((n_seg,), -math.inf, dtype=torch.float64, device=e.device)
    shift = shift.scatter_reduce(0, flat_seg, torch.where(on, flat_a.detach(), torch.full_like(flat_a, -math.inf)),
                                 reduce="amax", include_self=True)
    count = torch.zeros(n_seg, dtype=torch.long, device=e.device).index_add_(0, flat_seg, on.long())
    valid_seg = count > 0
    shift = torch.where(valid_seg, shift, torch.zeros_like(shift))           # empty segments: finite shift
    arg = torch.where(on, flat_a - shift[flat_seg], torch.zeros_like(flat_a))
    z_terms = torch.exp(arg) * on.to(arg.dtype)
    z = torch.zeros(n_seg, dtype=torch.float64, device=e.device).index_add(0, flat_seg, z_terms)
    log_z = shift + torch.log(torch.where(valid_seg, z, torch.ones_like(z)))  # [n_seg]
    p = torch.exp(torch.where(on, flat_a - log_z[flat_seg], torch.zeros_like(flat_a))) * on.to(flat_a.dtype)
    u = torch.zeros(n_seg, dtype=torch.float64, device=e.device).index_add(0, flat_seg, p * flat_e)
    dev = flat_e - u[flat_seg]
    var = torch.zeros(n_seg, dtype=torch.float64, device=e.device).index_add(0, flat_seg, p * dev * dev)
    keep = slice(0, r * n_groups)                                              # drop the trash segment

    def shape(x: torch.Tensor) -> torch.Tensor:
        return x[keep].reshape(*lead, n_groups)

    valid = shape(valid_seg)
    log_z_g, u_g, var_g = shape(log_z), shape(u), shape(var)

    def fill_(x: torch.Tensor) -> torch.Tensor:
        return torch.where(valid, x, torch.full_like(x, fill))

    return GibbsState(temperature=t.expand_as(log_z_g), log_partition=fill_(log_z_g), free_energy=fill_(-t * log_z_g),
                      mean_energy=fill_(u_g), entropy=fill_(log_z_g + u_g / t), heat_capacity=fill_(var_g / t**2),
                      occupation=p.reshape(*lead, n), size=shape(count), valid=valid)


@dataclass(frozen=True)
class ObservableResponse:
    """Linear response of the Gibbs mean of observables (module docstring), shapes [...] or [..., k].

    mean: <O>_p; susceptibility: Var_p(O) / T = d<O>/dh for E -> E - h O; thermal_response:
    Cov_p(O, E) / T^2 = d<O>/dT.
    """

    mean: torch.Tensor
    susceptibility: torch.Tensor
    thermal_response: torch.Tensor


def _observable(state: GibbsState, observable: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    # Observables [..., n] or [..., n, k] in float64, zero where the occupation is zero (masked members).
    p = state.occupation
    o = observable.to(torch.float64)
    single = o.dim() == p.dim()
    if single:
        o = o.unsqueeze(-1)
    if o.shape[:-1] != p.shape:
        raise ValueError(f"observable must be [..., n] or [..., n, k] with [..., n] = {tuple(p.shape)}")
    on = (p > 0).unsqueeze(-1)
    return torch.where(on, o, torch.zeros_like(o)), p.unsqueeze(-1), single


def observable_response(state: GibbsState, energy: torch.Tensor, observable: torch.Tensor) -> ObservableResponse:
    """Gibbs mean, susceptibility and thermal response of observables (fluctuation-dissipation, Kubo 1966).

    state: from `gibbs` over `energy` [..., n]; observable [..., n] or [..., n, k] (one value per member).
    Values of members outside the ensemble are ignored (they may be NaN).
    """
    o, p, single = _observable(state, observable)
    e = torch.where(p.squeeze(-1) > 0, energy.to(torch.float64), torch.zeros_like(p.squeeze(-1))).unsqueeze(-1)
    t = state.temperature.unsqueeze(-1)
    mean = (p * o).sum(-2)                                                     # [..., k]
    do = o - mean.unsqueeze(-2)
    de = e - state.mean_energy.unsqueeze(-1).unsqueeze(-1).nan_to_num(0.0)
    chi = (p * do * do).sum(-2) / t
    d_t = (p * do * de).sum(-2) / t**2
    ok = state.valid.unsqueeze(-1)
    nan = torch.full_like(mean, math.nan)
    mean, chi, d_t = torch.where(ok, mean, nan), torch.where(ok, chi, nan), torch.where(ok, d_t, nan)
    if single:
        mean, chi, d_t = mean.squeeze(-1), chi.squeeze(-1), d_t.squeeze(-1)
    return ObservableResponse(mean=mean, susceptibility=chi, thermal_response=d_t)


def cross_susceptibility(state: GibbsState, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cov_p(A, B) / T: the response of <A> to a field conjugate to B (and of <B> to one conjugate to A)."""
    oa, p, _ = _observable(state, a)
    ob, _, _ = _observable(state, b)
    if oa.shape[-1] != 1 or ob.shape[-1] != 1:
        raise ValueError("cross_susceptibility takes one observable per member on each side ([..., n])")
    da = oa - (p * oa).sum(-2, keepdim=True)
    db = ob - (p * ob).sum(-2, keepdim=True)
    chi = (p * da * db).sum(-2).squeeze(-1) / state.temperature
    return torch.where(state.valid, chi, torch.full_like(chi, math.nan))


def trailing_slope(values: torch.Tensor, times: torch.Tensor, valid: torch.Tensor, *, window: int
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """Growth rate at every point: least-squares slope over the last `window` valid points (module docstring).

    values, times (float64 seconds), valid (bool): [..., M] trajectories along the last axis. Returns
    (slope [..., M] float64, defined [..., M] bool). A point is defined when it is valid itself, at
    least two valid points enter the fit and their times are not all equal; elsewhere the slope is NaN.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    x = values.to(torch.float64)
    t = times.to(torch.float64)
    ok = valid.to(torch.bool) & torch.isfinite(x) & torch.isfinite(t)
    m_len = x.shape[-1]
    idx = torch.arange(m_len, device=x.device).expand_as(x)
    last = torch.cummax(torch.where(ok, idx, torch.full_like(idx, -1)), dim=-1).values   # last valid index <= m
    prev = torch.cat([torch.full_like(last[..., :1], -1), last[..., :-1]], dim=-1)       # last valid index < m
    # Chain of the window's member indices: k_0 = m (if valid), k_{j+1} = prev(k_j).    [..., M, w]
    chain = [torch.where(ok, idx, torch.full_like(idx, -1))]
    for _ in range(window - 1):
        k = chain[-1]
        chain.append(torch.where(k >= 0, prev.gather(-1, k.clamp_min(0)), torch.full_like(k, -1)))
    kk = torch.stack(chain, dim=-1)
    w = (kk >= 0).to(torch.float64)
    flat = kk.clamp_min(0).flatten(-2)
    tx = t.gather(-1, flat).reshape(kk.shape)
    xx = x.gather(-1, flat).reshape(kk.shape)
    n = w.sum(-1)
    tbar = (w * tx).sum(-1) / n.clamp_min(1.0)
    xbar = (w * xx).sum(-1) / n.clamp_min(1.0)
    dt = (tx - tbar.unsqueeze(-1)) * w
    stt = (dt * dt).sum(-1)
    stx = (dt * (xx - xbar.unsqueeze(-1)) * w).sum(-1)
    defined = ok & (n >= 2) & (stt > 0)
    slope = torch.where(defined, stx / torch.where(stt > 0, stt, torch.ones_like(stt)), torch.full_like(stt, math.nan))
    return slope, defined


class SlopeTracker:
    """Online growth rate of one trajectory: the slope of `trailing_slope` over the last `window` points.

    `update(t, x)` with a non-finite x leaves the window unchanged and returns NaN, as the batch form
    returns NaN at an invalid point without letting it enter later fits.
    """

    def __init__(self, window: int) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self._pts: deque[tuple[float, float]] = deque(maxlen=window)

    def update(self, t: float, x: float) -> float:
        """Add the point (t, x) and return the slope over the window (NaN if not defined)."""
        if not (math.isfinite(x) and math.isfinite(t)):
            return math.nan
        self._pts.append((float(t), float(x)))
        if len(self._pts) < 2:
            return math.nan
        n = len(self._pts)
        tbar = math.fsum(p[0] for p in self._pts) / n
        xbar = math.fsum(p[1] for p in self._pts) / n
        stt = math.fsum((p[0] - tbar) ** 2 for p in self._pts)
        if stt <= 0:
            return math.nan
        stx = math.fsum((p[0] - tbar) * (p[1] - xbar) for p in self._pts)
        return stx / stt


__all__ = ["GibbsState", "ObservableResponse", "SlopeTracker", "cross_susceptibility", "gibbs", "gibbs_grouped",
           "observable_response", "trailing_slope"]
