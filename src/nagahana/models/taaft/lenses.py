"""TAAFT's lenses as energy terms: E_total = sum over lenses of E_l(c, y) + lambda_phys Phi_phys (D-42, ADR-0007).

TAAFT (Topological Anti-Adversary Foundation Transformer) reads the Environment (TSTCT's KV cache) and writes
Imagination [A-14]. CVG-AE and TSTCT are perceptors, not analysers [A-19]; the analysis happens here,
through lenses. Each lens is a term of one energy (D-42): refinement descends their sum, and each lens's share
of every step is an explanation.

Why a sum of energies (a product of experts)
--------------------------------------------
exp(-sum_l E_l) = prod_l exp(-E_l): a sum of energies is a product of the lenses' unnormalised distributions
(Hinton, Neural Computation 14(8), 2002; Du, Li and Mordatch, NeurIPS 2020, arXiv:2004.06030). A hypothesis has
low E_total only when the lenses jointly accept it; a single very low term can offset a high one, which is why
every term is also reported on its own.

Common contract (every lens)
----------------------------
- Input: the hypotheses y [B, N, d_y] at one trigger (N = V entity states, then G adversary slots) and
  `LensInputs` (context c, masks and as-of structure, all fixed during descent).
- Output: `LensTerm` with energy [B] (per trigger) and, where meaningful, per-position contributions [B, N] that sum
  to the energy (an "energy per entity" output, architecture section 6).
- Bounded below (>= 0). Descent on a sum of terms bounded below cannot run away, whatever the learned weights do.
- Learned scale s_l = exp(log_scale_l) (ADR-0007), except the physics term, whose lambda_phys is configuration
  (AS-15): a learned lambda could be trained to zero, removing the boundary; and except the mechanism-design
  term, which shares the game lens's scale (below).
- Inactive positions contribute exactly 0 and receive zero gradient.
- Precision (D-54, AS-451): the elementwise work of a lens runs in float32; the reduction of a lens to its
  per-trigger energy is accumulated in float64, the learned scale is applied in float64, and `TotalEnergy` sums
  the lenses in float64. `LensTerm.energy` and E_total are float64; `LensTerm.per_token` stays float32.

The terms (build-spec section 2.7, AS-14; exact forms AS-202 ... AS-209 and AS-723 ... AS-728)
-----------------------------------------------------------------------------------------------
1. belief-trust (`BeliefTrustLens`): robust, trust-weighted evidence compatibility and priors; under the D-25
   option "boundary + telemetry reliability input", physics violations of the entity's own records lower the
   prior trust.
2. game (`GameLens`): the adversary's exploitability against the defender's committed coverage in a
   Stackelberg security game (distance from the follower's logit quantal-response equilibrium).
3. information (`InformationLens`): informativeness-priced departure from the null hypothesis; the noise
   features enter it under the working D-26 option.
4. topology (`TopologyLens`): robust pairwise MRF on the contact graph as of tau.
5. temporal (`TemporalLens`): compatibility with the previous trigger's hypothesis through f_time.
6. causal (`CausalLens`): directed robust pairwise term weighted by TSTCT's Granger gates.
7. physics (`PhysicsLens`): lambda_phys log(1 + Phi_phys / n) on the decoded believed next state.
Configurable own terms (held decisions, configured in `governance.decisions`):
8. mechanism-design (`MechanismDesignLens`, D-24): the participation (individual-rationality) part of the game
   lens's gap as a term of its own, when "mechanism-design" is listed in `TAAFTConfig.lenses`; the D-24 option
   "Advisor" removes the participation choice from TAAFT's game.
9. noise (`NoiseLens`, D-26): a mixture over white, coloured (long-range dependent) and periodic timing regimes
   as a term of its own under the D-26 option "own lens term"; the information lens then reads the context only.

The registry entry "energy" names the sum (`TotalEnergy`), not a term of its own (ADR-0007).

Extension points
----------------
Register a new `LensEnergy` subclass under a new name and list it in `TAAFTConfig.lenses`; ablate a lens by
removing its name (P-17). A lens may use `prepare` to cache context-only quantities once per trigger, so descent
steps only pay for the y-dependent part.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.core.errors import InvalidOption, InvariantViolation
from nagahana.core.registry import Registry
from nagahana.governance import decisions
from nagahana.governance.assumptions import assume
from nagahana.models.config.components import TAAFTConfig
from nagahana.models.taaft.noise import NOISE_FEATURES
from nagahana.models.taaft.readouts import Readouts
from nagahana.nn.positional import LogDeltaBias

#: D-24 options under which the analysis side of mechanism design sits in TAAFT.
MECHANISM_IN_TAAFT: tuple[str, ...] = ("TAAFT", "both (analysis in TAAFT, design in Advisor)")
#: D-26 option under which noise analysis is a lens term of its own.
NOISE_OWN_TERM = "own lens term"
#: D-25 option under which physics on observed records informs trust.
TELEMETRY_TRUST = "boundary + telemetry reliability input"
#: Timing regimes of the noise lens (D-26), in the order of its class axis.
NOISE_CLASSES: tuple[str, ...] = ("white", "coloured", "periodic")


class DecoderLike(Protocol):
    """What TAAFT needs from the Decoder to put physics on its beliefs.

    decode_fields(z [R, dz], role long [R], planes bool [R, n_planes]) -> decoded (opaque);
    physics_inputs(decoded) -> (values: field -> [R], contributing: field -> bool [R]) for `PhysicsTerm`.
    """

    def decode_fields(self, z: torch.Tensor, role: torch.Tensor, planes: torch.Tensor) -> Any: ...

    def physics_inputs(self, decoded: Any) -> tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]: ...


class PhysicsLike(Protocol):
    """`physics.term.PhysicsTerm`'s call signature: (values, contributing) -> (Phi_phys, breakdown)."""

    def __call__(
        self, values: Mapping[str, torch.Tensor], contributing: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


@dataclass(frozen=True)
class LensSpec:
    """Construction arguments shared by every lens, with the held-decision options resolved once.

    cfg: TAAFT configuration. d_ctx: context width (TAAFT dim). n_planes: relation planes (AS-01).
    The options in force at construction (`governance.decisions`): `mechanism_placement` (D-24),
    `noise_placement` (D-26) and `telemetry_trust` (D-25). They are checked against `cfg.lenses`: the
    mechanism-design term needs a placement in TAAFT and the game lens; the noise term is listed exactly when
    the D-26 option makes it a term of its own.
    """

    cfg: TAAFTConfig
    d_ctx: int
    n_planes: int
    mechanism_placement: str = field(init=False, default="")
    noise_placement: str = field(init=False, default="")
    telemetry_trust: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        m = decisions.require("mechanism-design-placement", by=__name__).value
        n = decisions.require("noise-analysis", by=__name__).value
        t = decisions.require("physics-on-telemetry", by=__name__).value
        assert m is not None and n is not None and t is not None
        object.__setattr__(self, "mechanism_placement", m)
        object.__setattr__(self, "noise_placement", n)
        object.__setattr__(self, "telemetry_trust", t == TELEMETRY_TRUST)
        names = tuple(self.cfg.lenses)
        if "mechanism-design" in names:
            if m not in MECHANISM_IN_TAAFT:
                raise InvalidOption(f"the lens 'mechanism-design' sits in TAAFT, but the D-24 option in force is {m!r}")
            if "game" not in names:
                raise InvalidOption("the lens 'mechanism-design' reads the game lens's equilibrium; list 'game' too")
        if ("noise" in names) != (n == NOISE_OWN_TERM):
            raise InvalidOption(
                f"the D-26 option in force is {n!r}: the lens 'noise' must be listed exactly when the option is "
                f"{NOISE_OWN_TERM!r}"
            )

    @property
    def rank(self) -> int:
        """Rank r of the pairwise and game subspaces: d_hyp // lens_rank_divisor (>= 1; AS-219)."""
        return max(1, self.cfg.d_hyp // self.cfg.lens_rank_divisor)

    @property
    def participation(self) -> bool:
        """Whether TAAFT's game has the adversary's participation choice (D-24 options with TAAFT)."""
        return self.mechanism_placement in MECHANISM_IN_TAAFT

    @property
    def mechanism_own_term(self) -> bool:
        """Whether the participation part of the game is a term of its own (D-42 note, D-24)."""
        return "mechanism-design" in self.cfg.lenses

    @property
    def noise_own_term(self) -> bool:
        """Whether the noise features form a term of their own (D-26)."""
        return self.noise_placement == NOISE_OWN_TERM


@dataclass
class LensInputs:
    """Everything the lenses read at one trigger tau, fixed during descent. Batch-first, per trigger.

    context [B, N, d]: TAAFT context c (null-substituted on dropped windows, AS-214);
    token_mask bool [B, N]: active positions (entities seen by tau; slots of valid triggers);
    n_entities: V (positions [0, V) are entity states, [V, N) adversary slots);
    hop1, hop2 bool [B, V, V]; planes bool [B, V, V, n_planes]: contacts as of tau;
    cause [B, V, V]: aggregated Granger gates from u to v as of tau (row u = cause, column v = effect);
    noise [B, V, F], noise_valid bool [B, V, F]: the noise features as of tau (noise.py);
    y_prev [B, N, d_y] or None, prev_valid bool [B, N], prev_dt float64 [B, N]: the position's refined
        hypothesis at its previous trigger within M_im (read from Imagination; detached);
    dropped bool [B]: the null-context reading E(empty, y) on these windows (P-09, AS-16);
    heads: the readout heads (trust, next latent) shared with `Readouts`;
    entity_role long [B, V], entity_planes bool [B, V, n_planes]: of each entity's latest update, for decoding
        its believed next state; decoder, physics: optional (the physics term is 0 without them);
    telemetry_violation float [B, V] or None: per-entity physics violation of the entity's own observed records
        (`physics.term.telemetry_violation`), accepted only under the D-25 telemetry option.
    """

    context: torch.Tensor
    token_mask: torch.Tensor
    n_entities: int
    hop1: torch.Tensor
    hop2: torch.Tensor
    planes: torch.Tensor
    cause: torch.Tensor
    noise: torch.Tensor
    noise_valid: torch.Tensor
    y_prev: torch.Tensor | None
    prev_valid: torch.Tensor
    prev_dt: torch.Tensor
    dropped: torch.Tensor
    heads: Readouts
    entity_role: torch.Tensor
    entity_planes: torch.Tensor
    decoder: DecoderLike | None = None
    physics: PhysicsLike | None = None
    telemetry_violation: torch.Tensor | None = None

    @property
    def entity_mask(self) -> torch.Tensor:
        return self.token_mask[:, : self.n_entities]

    @property
    def slot_mask(self) -> torch.Tensor:
        return self.token_mask[:, self.n_entities :]


@dataclass
class LensTerm:
    """One lens's value: energy [B]; per_token [B, N] summing to it (or None); aux: explanations."""

    energy: torch.Tensor
    per_token: torch.Tensor | None = None
    aux: dict[str, torch.Tensor] = field(default_factory=dict)


Prepared = dict[str, torch.Tensor]


def _pair_mask(m: torch.Tensor) -> torch.Tensor:
    """Active-pair mask without the diagonal: m bool [B, V] -> [B, V, V]."""
    eye = torch.eye(m.shape[-1], dtype=torch.bool, device=m.device)[None]
    return m[:, :, None] & m[:, None, :] & ~eye


def _sq_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """||a_u - b_v||^2 for all pairs: a [B, U, r], b [B, V, r] -> [B, U, V] (>= 0 after clamping round-off)."""
    d = (a * a).sum(-1)[:, :, None] + (b * b).sum(-1)[:, None, :] - 2.0 * a @ b.transpose(1, 2)
    return d.clamp_min(0.0)


class LensEnergy(nn.Module, abc.ABC):
    """Base class of an energy term. Subclasses implement `raw` (and optionally `prepare`)."""

    #: registry name (also the key in `AnalysisOut.lens_energy`).
    name: str = ""
    #: whether the term carries a learned output scale.
    scaled: bool = True

    def __init__(self, spec: LensSpec) -> None:
        super().__init__()
        self.spec = spec
        if self.scaled:
            self.log_scale = nn.Parameter(torch.zeros(()))

    def prepare(self, inp: LensInputs) -> Prepared:
        """Context-only quantities, computed once per trigger (not per descent step)."""
        return {}

    @abc.abstractmethod
    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        """The unscaled term at hypotheses y [B, N, d_y]."""

    def forward(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        t = self.raw(y, inp, prep)
        # Energy reported in float64 (D-54): `raw` already reduces in float64; `.double()` is exact.
        energy = t.energy.double()
        if not self.scaled:
            return LensTerm(energy, t.per_token, t.aux)
        s = torch.exp(self.log_scale)
        # Learned scale applied in float64 to the energy; the float32 per-position map keeps its dtype.
        return LensTerm(energy * s.double(), None if t.per_token is None else t.per_token * s, t.aux)


class BeliefTrustLens(LensEnergy):
    """E_bt: trust-weighted evidence compatibility and priors (POMDP belief and trust or distrust; AS-202).

    Mathematics (per active entity v; per active position i for the prior)
    ----------------------------------------------------------------
        psi_v = softplus(w . SiLU(W_c c_v + W_y y_v + b) + b_o) >= 0       learned -log p(evidence_v | y_v)
        t_v   = sigmoid(w_t . y_v + b_t)                                trust readout (shared head)
        E_ev  = -log( t_v exp(-psi_v) + (1 - t_v) exp(-beta0) )         robust mixture with an outlier level beta0
        KL_t  = KL( Bern(t_v) || Bern(t0_v) ),  logit t0_v = a(c_v) - g x telemetry_violation_v
        P_i   = 1/2 sum_d (y_id - mu_d(c_i))^2 exp(-log sigma^2_d(c_i))  penalty towards the TAAFT prior
        E_bt  = sum_v (E_ev + KL_t) + sum_i P_i

    Why it is an energy, and what its gradient means
    ------------------------------------------------
    t exp(-psi) + (1 - t) exp(-beta0) is the likelihood of the evidence under "the telemetry is trustworthy
    (probability t) and explained by y" versus "the telemetry is corrupted and explains nothing" (level beta0):
    the outlier-process view of robust estimation (Black and Rangarajan, IJCV 19(1), 1996). Its gradient in y is
    rho_v grad psi_v with responsibility rho_v = t exp(-psi) / (t exp(-psi) + (1 - t) exp(-beta0)): evidence
    pulls the hypothesis in proportion to how much it is trusted. The gradient in t raises trust where the
    hypothesis explains the evidence better than beta0 and lowers it otherwise, against KL_t, so trust cannot be
    lowered for free to escape inconvenient evidence. All parts are >= 0.

    Telemetry reliability (D-25)
    ----------------------------
    Under the option "boundary + telemetry reliability input" the prior trust also reads the physics violation
    of the entity's own observed records (`LensInputs.telemetry_violation`, >= 0): a record that breaks physics
    signals a faulty sensor or forged telemetry, so the prior logit drops by g x violation with a learned
    g = softplus(raw) >= 0. Under the working option "boundary only" (AS-40) the input is refused.
    """

    name = "belief-trust"

    def __init__(self, spec: LensSpec) -> None:
        super().__init__(spec)
        cfg, d, h = spec.cfg, spec.d_ctx, spec.cfg.lens_hidden
        self.w_c = nn.Linear(d, h)
        self.w_y = nn.Linear(cfg.d_hyp, h, bias=False)
        self.w_o = nn.Linear(h, 1)
        self.prior = nn.Linear(d, 2 * cfg.d_hyp)            # mu(c), log sigma^2(c)
        self.trust_prior = nn.Linear(d, 1)                  # logit of t0(c)
        self.beta0_raw = nn.Parameter(torch.tensor(1.0))    # beta0 = softplus(raw) >= 0
        if spec.telemetry_trust:
            self.telemetry_gain_raw = nn.Parameter(torch.tensor(0.0))   # g = softplus(raw) >= 0

    def prepare(self, inp: LensInputs) -> Prepared:
        v = inp.n_entities
        c = inp.context
        mu, lv = self.prior(c).chunk(2, dim=-1)             # [B, N, d_y] each
        t0 = self.trust_prior(c[:, :v]).squeeze(-1)         # [B, V] logit
        if inp.telemetry_violation is not None:
            if not self.spec.telemetry_trust:
                raise InvalidOption("telemetry_violation is an input only under the D-25 option "
                                    f"{TELEMETRY_TRUST!r}; the option in force keeps physics a boundary (AS-40)")
            tv = inp.telemetry_violation.to(t0.dtype)
            if tv.shape != t0.shape or bool((tv < 0).any()):
                raise InvariantViolation("telemetry_violation must be [B, V] and >= 0")
            t0 = t0 - F.softplus(self.telemetry_gain_raw) * tv
        return {
            "hc": self.w_c(c[:, :v]),                       # [B, V, h]
            "mu": mu,
            "lv": 4.0 * torch.tanh(lv / 4.0),               # smooth bound [-4, 4]
            "t0": t0,
        }

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        v = inp.n_entities
        ye = y[:, :v]
        # Evidence compatibility psi >= 0.
        psi = F.softplus(self.w_o(F.silu(prep["hc"] + self.w_y(ye))).squeeze(-1))          # [B, V]
        # Robust mixture with the shared trust head (log space for stability).
        tl = inp.heads.trust_logit(ye)                                                    # [B, V]
        log_t, log_1mt = F.logsigmoid(tl), F.logsigmoid(-tl)
        beta0 = F.softplus(self.beta0_raw)
        e_ev = -torch.logaddexp(log_t - psi, log_1mt - beta0)
        # KL(Bern(t) || Bern(t0)) in logits.
        t = torch.exp(log_t)
        t0l = prep["t0"]
        kl_t = t * (log_t - F.logsigmoid(t0l)) + (1 - t) * (log_1mt - F.logsigmoid(-t0l))
        # Prior penalty for every position (entity states and slots).
        pen = 0.5 * (((y - prep["mu"]) ** 2) * torch.exp(-prep["lv"])).sum(-1)              # [B, N]
        per = pen * inp.token_mask.to(pen.dtype)
        per = per + F.pad((e_ev + kl_t) * inp.entity_mask.to(pen.dtype), (0, y.shape[1] - v))
        # Per-trigger energy: the position sum accumulated in float64 (D-54).
        return LensTerm(per.double().sum(-1), per, {"responsibility": torch.sigmoid(log_t - psi - log_1mt + beta0)})


def quantal_response(payoff: torch.Tensor, valid: torch.Tensor, beta: torch.Tensor | float) -> torch.Tensor:
    """log of the logit quantal response softmax(beta u) over the valid options (last axis); -inf where invalid.

    McKelvey and Palfrey, "Quantal Response Equilibria for Normal Form Games", Games and Economic Behavior
    10(1), 1995. Rows without a valid option return a uniform distribution over all options.
    """
    has = valid.any(-1, keepdim=True)
    ok = valid | ~has
    return torch.log_softmax((beta * payoff).masked_fill(~ok, float("-inf")), dim=-1)


def security_game_qre(
    value: torch.Tensor,
    penalty: torch.Tensor,
    budget: torch.Tensor | float,
    entity_mask: torch.Tensor,
    *,
    beta: float,
    outside: torch.Tensor | None = None,
    iterations: int = 2000,
    tol: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The logit quantal-response equilibrium of the zero-sum security game (`GameLens`), both sides adapting.

    Game per leading index: the attacker picks a target v (or the outside option when `outside` is given), the
    defender spreads a coverage budget m over the targets with distribution x, and the attacker's payoff is
        U_a(sigma, x) = sum_v sigma_v (R_v - m x_v (R_v + P_v)) + sigma_out u0 = sigma^T A x,
        A[v, w] = R_v - m (R_v + P_v) [v = w],   A[out, w] = u0,
    zero-sum with the defender. With entropy regularisation tau = 1 / beta on both sides its unique saddle point
    is the logit QRE. It is computed by the predictive-update (extragradient) method of Cen, Wei and Chi,
    "Fast Policy Extragradient Methods for Competitive Games with Entropy Regularization", NeurIPS 2021
    (arXiv:2105.15186), which converges linearly at rate (1 - eta tau) for eta = 1 / (tau + 2 max|A|):
        log sigma_bar = (1 - eta tau) log sigma + eta A x          (normalised)
        log x_bar     = (1 - eta tau) log x     - eta A^T sigma    (normalised)
        log sigma     = (1 - eta tau) log sigma + eta A x_bar       (normalised)
        log x         = (1 - eta tau) log x     - eta A^T sigma_bar (normalised)
    value R, penalty P: [..., V] >= 0; budget m in (0, 1]; entity_mask bool [..., V]; outside u0 [...] or None.

    Returns (sigma [..., V (+1)], x [..., V], nashconv [...]) with nashconv the entropy-regularised NashConv at the
    returned point (tau KL(sigma || QR_a(x)) + tau KL(x || QR_d(sigma)), zero at the QRE). Every slice needs at
    least one target.
    """
    if not beta > 0:
        raise ValueError("beta must be positive")
    if not bool(entity_mask.any(-1).all()):
        raise InvariantViolation("every game needs at least one target")
    tau = 1.0 / beta
    r, p = value.double(), penalty.double()
    m = torch.as_tensor(budget, dtype=torch.float64)
    mask = entity_mask
    lead_shape = r.shape[:-1]
    u0 = None if outside is None else outside.double().expand(lead_shape)
    coef = m * (r + p)                                                       # m (R + P) [..., V]

    def attacker_payoff(x: torch.Tensor) -> torch.Tensor:
        # (A x)_v = R_v - m (R_v + P_v) x_v ; (A x)_out = u0
        q = r - coef * x
        return q if u0 is None else torch.cat([q, u0.unsqueeze(-1)], dim=-1)

    def defender_loss(sigma: torch.Tensor) -> torch.Tensor:
        # (A^T sigma)_w = sum_v sigma_v R_v + sigma_out u0 - m sigma_w (R_w + P_w): the attacker's payoff of
        # leaving w uncovered; the defender minimises it.
        s_t = sigma[..., : r.shape[-1]]
        base = (s_t * r).sum(-1, keepdim=True)
        if u0 is not None:
            base = base + sigma[..., -1:] * u0.unsqueeze(-1)
        return base - coef * s_t

    a_valid = mask if u0 is None else torch.cat([mask, torch.ones_like(mask[..., :1])], dim=-1)
    a_abs = torch.where(mask, (r - coef).abs().maximum(r.abs()), torch.zeros_like(r)).amax(-1)
    if u0 is not None:
        a_abs = a_abs.maximum(u0.abs())
    eta = (1.0 / (tau + 2.0 * a_abs)).unsqueeze(-1)                          # [..., 1]
    keep = 1.0 - eta * tau
    neg = torch.tensor(float("-inf"), dtype=torch.float64)
    log_s = torch.log_softmax(torch.zeros(a_valid.shape, dtype=torch.float64).masked_fill(~a_valid, neg), -1)
    log_x = torch.log_softmax(torch.zeros(mask.shape, dtype=torch.float64).masked_fill(~mask, neg), -1)

    def step(ls: torch.Tensor, lx: torch.Tensor, x_ref: torch.Tensor, s_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # One multiplicative-weights step of both players against the reference strategies.
        ls_new = torch.where(a_valid, keep * ls + eta * attacker_payoff(x_ref), neg)
        lx_new = torch.where(mask, keep * lx - eta * defender_loss(s_ref), neg)
        return torch.log_softmax(ls_new, -1), torch.log_softmax(lx_new, -1)

    def nashconv(ls: torch.Tensor, lx: torch.Tensor) -> torch.Tensor:
        s, x = ls.exp(), lx.exp()
        qa = quantal_response(attacker_payoff(x), a_valid, beta)
        qd = quantal_response(-defender_loss(s), mask, beta)
        kl_a = torch.where(a_valid, s * (ls - qa), torch.zeros_like(s)).sum(-1)
        kl_d = torch.where(mask, x * (lx - qd), torch.zeros_like(x)).sum(-1)
        return tau * (kl_a + kl_d)

    for it in range(iterations):
        s, x = log_s.exp(), log_x.exp()
        ls_bar, lx_bar = step(log_s, log_x, x, s)                            # prediction
        log_s, log_x = step(log_s, log_x, lx_bar.exp(), ls_bar.exp())        # update against the prediction
        if (it + 1) % 10 == 0 and float(nashconv(log_s, log_x).max()) <= tol:
            break
    return log_s.exp(), log_x.exp(), nashconv(log_s, log_x)


class GameLens(LensEnergy):
    """E_game: the adversary hypotheses' exploitability in a Stackelberg security game (AS-723; D-24 placement).

    The game at one trigger (Kiekintveld et al., AAMAS 2009, the security-game structure)
    -----------------------------------------------------------------------------------
    Targets are the active entities. The defender commits to a coverage c_v = m x_v of every target, with a
    budget m = sigmoid(raw) in (0, 1) and an allocation x = softmax over active v of a(c_v) read from the
    Environment context: the leader's commitment as the adversary can observe it. Adversary hypothesis g (an
    adversary slot) values target v at R_gv = kappa_R sigmoid(<W_a y_g, W_e y_v> / sqrt(r')) and is caught there
    at a cost P_v = kappa_P sigmoid(b(c_v)) (the defender-shaped consequence, from the context). Its expected
    payoff of attacking v is
        u_gv = (1 - c_v) R_gv - c_v P_gv = R_gv - c_v (R_gv + P_v),
    and, when the D-24 option keeps mechanism design in TAAFT ("TAAFT" or "both"), an outside option
    u_out = kappa_R tanh(raw): the participation constraint of mechanism design (Myerson, "Optimal Auction
    Design", Mathematics of Operations Research 6(1), 1981).

    The energy: distance from equilibrium
    -------------------------------------
    Each hypothesis claims a strategy sigma_g = softmax over the options of (<W_s y_g, W_t y_v> / sqrt(r'), b_out).
    Its logit quantal best response (McKelvey and Palfrey 1995) is pi_g = softmax(beta u_g). The adversary's
    regret of playing sigma_g, with entropy regularisation 1 / beta, is its exploitability:
        Gap_g = max_sigma' [sigma'.u_g + H(sigma') / beta] - [sigma_g.u_g + H(sigma_g) / beta]
              = (1 / beta) KL(sigma_g || pi_g) >= 0,
    zero exactly when sigma_g is the follower's quantal-response equilibrium strategy against the commitment.
    E_game = (1 / G_act) sum over active slots of Gap_g: hypotheses whose claimed behaviour is rational against
    the observed defence get low energy ("consistent with a rational progression", build-spec section 2.7).
    This is the follower's part of NashConv (Lanctot et al., "A Unified Game-Theoretic Approach to Multiagent
    Reinforcement Learning", NeurIPS 2017, arXiv:1711.00832). The leader's part is not an energy: the
    defender's commitment is observed, and its optimality is not evidence about the adversary. It is reported
    instead (`defender_regret`, the defender's gap against the average claimed threat), and
    `equilibrium_report` gives the two-sided QRE (`security_game_qre`) the interaction would settle at if both
    sides adapted. In security games the defender's Stackelberg strategies are Nash strategies as well
    (Korzhyk, Yin, Kiekintveld, Conitzer and Tambe, JAIR 41, 2011), so the two readings agree on the leader.

    Mechanism design as a term of its own (D-24, D-42 note)
    ------------------------------------------------------
    By the chain rule of the Kullback-Leibler divergence (Cover and Thomas, Elements of Information Theory,
    2nd ed., theorem 2.5.3), with act = 1 - out,
        KL(sigma || pi) = KL(Bern(sigma_act) || Bern(pi_act)) + sigma_act KL(sigma_act-conditional || pi_act-conditional):
    the participation (individual-rationality) gap plus the incentive-compatibility gap of the target choice.
    When "mechanism-design" is listed in `TAAFTConfig.lenses`, this lens reports the second part and
    `MechanismDesignLens` the first, at this lens's scale, so E_total is the same either way (tested).

    Parameters and compute
    ----------------------
    The valuation and the claimed targeting each use rank r' = r // 2 (r = AS-219's rank), so the bilinear
    budget and the pairwise compute equal those of one rank-r interaction (AS-723). Aux: "targets" (pi),
    "claimed" (sigma), "regret" (Gap_g), "coverage" (c), "defender_response" (the defender's quantal response to
    the average claimed threat) and "defender_regret".
    """

    name = "game"

    def __init__(self, spec: LensSpec) -> None:
        super().__init__(spec)
        assume("AS-14", by=__name__)
        d_y = spec.cfg.d_hyp
        r = max(1, spec.rank // 2)
        self.r_game = r
        self.w_s = nn.Linear(d_y, r, bias=False)            # slot: claimed targeting
        self.w_t = nn.Linear(d_y, r, bias=False)            # entity: claimed targeting
        self.w_a = nn.Linear(d_y, r, bias=False)            # slot: valuation
        self.w_e = nn.Linear(d_y, r, bias=False)            # entity: valuation
        self.ctx = nn.Linear(spec.d_ctx, 2)                 # per entity: (coverage logit, capture-cost logit)
        self.kappa_raw = nn.Parameter(torch.tensor(1.0))    # kappa_R = softplus(raw): scale of valuations
        self.kappa_p_raw = nn.Parameter(torch.tensor(1.0))  # kappa_P = softplus(raw): scale of capture costs
        self.beta_raw = nn.Parameter(torch.tensor(1.0))     # beta = softplus(raw) + 1e-3: rationality
        self.budget_raw = nn.Parameter(torch.tensor(0.0))   # m = sigmoid(raw): coverage budget
        self.participation = spec.participation
        if self.participation:
            self.u_abstain = nn.Parameter(torch.zeros(()))  # outside option u_out = kappa_R tanh(raw)
            self.claim_abstain = nn.Parameter(torch.zeros(()))   # claimed logit of abstaining
        self._cache: tuple[torch.Tensor, LensInputs, Prepared, dict[str, torch.Tensor]] | None = None

    def scalars(self) -> dict[str, torch.Tensor]:
        """The game's learned scalars after their links: kappa_R, kappa_P, beta, m (and u_out with participation)."""
        out = {"kappa": F.softplus(self.kappa_raw), "kappa_p": F.softplus(self.kappa_p_raw),
               "beta": F.softplus(self.beta_raw) + 1e-3, "budget": torch.sigmoid(self.budget_raw)}
        if self.participation:
            out["outside"] = out["kappa"] * torch.tanh(self.u_abstain)
        return out

    def prepare(self, inp: LensInputs) -> Prepared:
        v = inp.n_entities
        h = self.ctx(inp.context[:, :v])                                                # [B, V, 2]
        ent = inp.entity_mask
        has = ent.any(-1, keepdim=True)
        logit = h[..., 0].masked_fill(~(ent | ~has), float("-inf"))
        alloc = torch.softmax(logit, dim=-1) * ent.to(h.dtype)                          # x [B, V], zeros if none active
        return {"alloc": alloc, "cost_unit": torch.sigmoid(h[..., 1])}                  # P_v / kappa_P [B, V]

    def payoffs(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> dict[str, torch.Tensor]:
        """Valuations R [B, G, V], capture costs P [B, 1, V], coverage c [B, 1, V], payoffs u [B, G, V], claims [B, G, V]."""
        v = inp.n_entities
        ye, ys = y[:, :v], y[:, v:]
        sc = self.scalars()
        scale = math.sqrt(self.r_game)
        value = sc["kappa"] * torch.sigmoid(self.w_a(ys) @ self.w_e(ye).transpose(1, 2) / scale)
        cost = sc["kappa_p"] * prep["cost_unit"][:, None, :]
        cover = sc["budget"] * prep["alloc"][:, None, :]
        payoff = value - cover * (value + cost)
        claim = self.w_s(ys) @ self.w_t(ye).transpose(1, 2) / scale
        return {"value": value, "cost": cost, "cover": cover, "payoff": payoff, "claim": claim}

    def equilibrium(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> dict[str, torch.Tensor]:
        """The follower's quantal response, the claimed strategies and the gap split (class docstring).

        The result of the latest call is kept while its inputs are the same objects, so the game term and the
        mechanism-design term of one energy evaluation share one computation and one autograd graph.
        """
        c = self._cache
        if c is not None and c[0] is y and c[1] is inp and c[2] is prep:
            return c[3]
        v = inp.n_entities
        b, g = y.shape[0], y.shape[1] - v
        pay = self.payoffs(y, inp, prep)
        sc = self.scalars()
        beta = sc["beta"]
        ent_on = inp.entity_mask[:, None, :].expand(b, g, v)
        if self.participation:
            outside = sc["outside"].expand(b, g, 1)
            u = torch.cat([pay["payoff"], outside], dim=-1)
            claim = torch.cat([pay["claim"], self.claim_abstain.expand(b, g, 1)], dim=-1)
            valid = torch.cat([ent_on, torch.ones_like(ent_on[..., :1])], dim=-1)
        else:
            u, claim, valid = pay["payoff"], pay["claim"], ent_on
        has = valid.any(-1)                                                             # [B, G]
        ok = valid | ~has[..., None]
        log_pi = quantal_response(u, valid, beta)                                       # [B, G, O]
        log_sigma = torch.log_softmax(claim.masked_fill(~ok, float("-inf")), dim=-1)
        sigma = torch.exp(log_sigma)
        ls = torch.where(ok, log_sigma, torch.zeros_like(log_sigma))
        lp = torch.where(ok, log_pi, torch.zeros_like(log_pi))
        terms = torch.where(ok, sigma * (ls - lp), torch.zeros_like(sigma)) / beta      # per option, sums to Gap
        # Participation split (chain rule): act_term = sigma_act (log sigma_act - log pi_act), over target options.
        ent_ok = ok[..., :v] & ent_on
        has_ent = ent_ok.any(-1)
        neg = torch.full_like(ls[..., :v], float("-inf"))
        log_s_act = torch.logsumexp(torch.where(ent_ok, ls[..., :v], neg), dim=-1)
        log_p_act = torch.logsumexp(torch.where(ent_ok, lp[..., :v], neg), dim=-1)
        safe_s = torch.where(has_ent, log_s_act, torch.zeros_like(log_s_act))
        safe_p = torch.where(has_ent, log_p_act, torch.zeros_like(log_p_act))
        act_term = torch.where(has_ent, torch.exp(safe_s) * (safe_s - safe_p), torch.zeros_like(safe_s)) / beta
        if self.participation:
            participation = terms[..., -1] + act_term                                   # [B, G] IR gap
        else:
            participation = torch.zeros_like(act_term)
        gap = terms.sum(-1)                                                             # [B, G] (1/beta) KL
        slot_on = (inp.slot_mask & has).to(gap.dtype)
        weight = slot_on / slot_on.sum(-1, keepdim=True).clamp_min(1.0)                 # [B, G] uniform over active
        # Per-entity incentive-compatibility shares: sigma_gv [(log sigma - log pi) - (log sigma_act - log pi_act)] / beta.
        ic_ent = terms[..., :v] - sigma[..., :v] * (safe_s - safe_p)[..., None] / beta
        st = {"gap": gap, "participation": participation, "weight": weight, "terms": terms, "ic_entity": ic_ent,
              "log_pi": log_pi, "sigma": sigma, "valid": valid, "cover": pay["cover"][:, 0], "value": pay["value"],
              "cost": pay["cost"], "beta": beta}
        self._cache = (y, inp, prep, st)
        return st

    def clear_cache(self) -> None:
        """Drop the kept equilibrium (and the tensors it references)."""
        self._cache = None

    def defender_view(self, st: dict[str, torch.Tensor], inp: LensInputs) -> dict[str, torch.Tensor]:
        """The defender's quantal response to the average claimed threat and the regret of the commitment."""
        v = inp.n_entities
        w = st["weight"]                                                                # [B, G]
        threat = (w[..., None] * st["sigma"][..., :v] * (st["value"] + st["cost"])).sum(1)   # [B, V]
        ent = inp.entity_mask
        x = st["cover"] / self.scalars()["budget"]                                      # allocation [B, V]
        beta = st["beta"]
        log_rho = quantal_response(threat, ent, beta)
        has = ent.any(-1)
        safe_x = torch.where(ent, torch.log(x.clamp_min(1e-30)), torch.zeros_like(x))
        safe_r = torch.where(ent, log_rho, torch.zeros_like(log_rho))
        regret = torch.where(ent, x * (safe_x - safe_r), torch.zeros_like(x)).sum(-1) / beta
        return {"defender_response": torch.where(ent, log_rho.exp(), torch.zeros_like(x)),
                "defender_regret": torch.where(has, regret, torch.zeros_like(regret))}

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        v = inp.n_entities
        st = self.equilibrium(y, inp, prep)
        w = st["weight"]
        if self.spec.mechanism_own_term:
            # Incentive compatibility only: the participation part is the mechanism-design term.
            per_ent = (w[..., None] * st["ic_entity"]).sum(1)                           # [B, V]
            per_slot = torch.zeros_like(w)
            energy = (w * (st["gap"] - st["participation"])).double().sum(-1)
        else:
            per_ent = (w[..., None] * st["terms"][..., :v]).sum(1)
            per_slot = w * st["terms"][..., -1] if self.participation else torch.zeros_like(w)
            energy = (w * st["gap"]).double().sum(-1)
        per = torch.cat([per_ent, per_slot], dim=-1)
        aux = {"targets": st["log_pi"].exp(), "claimed": st["sigma"], "regret": st["gap"], "coverage": st["cover"]}
        aux.update(self.defender_view(st, inp))
        return LensTerm(energy, per, aux)

    @torch.no_grad()
    def equilibrium_report(self, y: torch.Tensor, inp: LensInputs, prep: Prepared, *, iterations: int = 2000,
                           tol: float = 1e-10) -> dict[str, torch.Tensor]:
        """The two-sided logit QRE of every slot's game at y (`security_game_qre`), for analysts and the Advisor.

        Returns "attacker" [B, G, V (+1)] and "coverage" [B, G, V] (the defender's equilibrium allocation against
        that slot's valuation) and "nashconv" [B, G] at the returned point. Slots of triggers without an active
        entity get zeros.
        """
        v = inp.n_entities
        pay = self.payoffs(y, inp, prep)
        sc = self.scalars()
        b, g = y.shape[0], y.shape[1] - v
        mask = inp.entity_mask[:, None, :].expand(b, g, v)
        has = mask.any(-1)
        safe_mask = mask | ~has[..., None]
        outside = sc["outside"].expand(b, g) if self.participation else None
        cost = pay["cost"].expand(b, g, v)
        sigma, x, nc = security_game_qre(pay["value"], cost, float(sc["budget"]), safe_mask, beta=float(sc["beta"]),
                                         outside=outside, iterations=iterations, tol=tol)
        z = has[..., None].to(sigma.dtype)
        return {"attacker": sigma * z, "coverage": x * has[..., None].to(x.dtype), "nashconv": nc * has.to(nc.dtype)}


class MechanismDesignLens(LensEnergy):
    """E_mech: the participation (individual-rationality) gap of the adversary hypotheses (D-24 own term, AS-724).

        E_mech = s_game (1 / G_act) sum over active slots of (1 / beta) KL( Bern(sigma_g,act) || Bern(pi_g,act) )

    the first part of the chain-rule split of the game lens's gap (`GameLens`): whether a hypothesis's claim to
    act at all, rather than take the outside option, is a rational response to the incentives the defender's
    mechanism sets. It carries the game lens's learned scale s_game (it has none of its own), so listing it moves
    energy between the two terms and leaves E_total unchanged. It exists only under the D-24 options that keep
    mechanism design in TAAFT, and reads the game lens's equilibrium (attached by `TotalEnergy`).
    """

    name = "mechanism-design"
    scaled = False

    def __init__(self, spec: LensSpec) -> None:
        super().__init__(spec)
        if not spec.participation:
            raise InvalidOption("the mechanism-design term needs the D-24 option to keep mechanism design in TAAFT")
        self._game: GameLens | None = None

    def attach(self, game: GameLens) -> None:
        """Read `game`'s equilibrium (kept outside this module's parameters: the game owns them)."""
        object.__setattr__(self, "_game", game)

    def prepare(self, inp: LensInputs) -> Prepared:
        if self._game is None:
            raise InvariantViolation("attach the game lens before use (TotalEnergy does)")
        return self._game.prepare(inp)

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        game = self._game
        if game is None:
            raise InvariantViolation("attach the game lens before use (TotalEnergy does)")
        st = game.equilibrium(y, inp, prep)
        s = torch.exp(game.log_scale)
        w = st["weight"]
        per_slot = w * st["participation"] * s
        per = torch.cat([torch.zeros(w.shape[0], inp.n_entities, dtype=per_slot.dtype, device=per_slot.device), per_slot], -1)
        energy = (w * st["participation"]).double().sum(-1) * s.double()
        return LensTerm(energy, per, {"participation_gap": st["participation"]})


class InformationLens(LensEnergy):
    """E_info: cost of departing from the null hypothesis, priced by how informative the evidence is (AS-204).

    Mathematics (per active entity v)
    --------------------------------
        e_v    = W_nu [nu_v * valid_v ; valid_v]                         noise features with their flags (D-41)
        kappa_v = softplus( w . SiLU(W_c c_v + e_v + b) + b_o )          precision of the null prior
        E_info = sum_v 1/2 kappa_v ||W_i (y_v - y_null)||^2 / r

    Why it is an energy, and what its gradient means
    ------------------------------------------------
    1/2 kappa ||Delta||^2 is, up to constants, the code length in nats of describing a departure Delta from the
    null hypothesis y_null under a Gaussian prior of precision kappa: the minimum-description-length reading
    (Rissanen, Automatica 14(5), 1978). kappa_v is learned from the context and from nu_v: where the evidence is
    uninformative (few events, white-noise timing, low SNR) kappa is high and a strong claim costs a lot; where it
    is informative (a sharp periodic peak, many structured events) kappa is low and the hypothesis may move. The
    gradient kappa_v W_i^T W_i (y_v - y_null) / r pulls unsupported claims back to "nothing special here".

    Noise (D-26): under the working option "inside the information lens" (AS-36) the noise features
    nu_v = (log periodogram peak ratio, log dominant period, aggregated-variance Hurst, log event count) enter
    kappa_v. Under the option "own lens term" they form `NoiseLens` and this lens reads the context only.
    kappa_v is reported in aux["precision"].
    """

    name = "information"

    def __init__(self, spec: LensSpec) -> None:
        super().__init__(spec)
        cfg, d, h, r = spec.cfg, spec.d_ctx, spec.cfg.lens_hidden, spec.rank
        self.noise_inside = not spec.noise_own_term
        if self.noise_inside:
            self.w_nu = nn.Linear(2 * len(NOISE_FEATURES), h, bias=False)
        self.w_c = nn.Linear(d, h)
        self.w_o = nn.Linear(h, 1)
        self.w_i = nn.Linear(cfg.d_hyp, r, bias=False)
        self.y_null = nn.Parameter(torch.zeros(cfg.d_hyp))

    def prepare(self, inp: LensInputs) -> Prepared:
        v = inp.n_entities
        hidden = self.w_c(inp.context[:, :v])
        if self.noise_inside:
            assume("AS-36", by=__name__)
            valid = inp.noise_valid.to(inp.noise.dtype)
            x = torch.cat([inp.noise * valid, valid], dim=-1)          # [B, V, 2F]; absent is not zero (D-41)
            hidden = hidden + self.w_nu(x)
        kappa = F.softplus(self.w_o(F.silu(hidden)).squeeze(-1))
        return {"kappa": kappa}                                        # [B, V]

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        v = inp.n_entities
        r = self.w_i.out_features
        dev = self.w_i(y[:, :v] - self.y_null)                                         # [B, V, r]
        e = 0.5 * prep["kappa"] * (dev * dev).sum(-1) / r * inp.entity_mask.to(dev.dtype)
        per = F.pad(e, (0, y.shape[1] - v))
        # Per-trigger energy: the entity sum accumulated in float64 (D-54).
        return LensTerm(e.double().sum(-1), per, {"precision": prep["kappa"]})


class NoiseLens(LensEnergy):
    """E_noise: the evidence of an entity's timing under the timing regime its hypothesis believes (D-26, AS-725).

    Mathematics (per active entity v; features f with validity flags)
    -----------------------------------------------------------------
    Three regimes k (`NOISE_CLASSES`): white (independent gaps, Hurst H = 0.5), coloured (long-range dependent,
    H = 0.8, the self-similarity of aggregate traffic found by Leland, Taqqu, Willinger and Wilson, IEEE/ACM ToN
    2(1), 1994) and periodic (beacon-like, a high periodogram peak; Hu et al., BAYWATCH, DSN 2016). The
    hypothesis believes pi_k(y_v) = softmax(W_n y_v + b_n); each regime models the features by a diagonal
    Gaussian N(mu_kf, sigma_kf^2) with log sigma_kf in [s_lo, s_hi]. With l_kf = log N(nu_vf; mu_kf, sigma_kf^2)
    and l_max = -1/2 log(2 pi) - s_lo the largest density,
        E_v = -log sum_k pi_k(y_v) exp( sum_f valid_vf (l_kf - l_max) )  >= 0.
    An invalid feature contributes no factor (absence is not zero, D-41), so an entity without timing evidence has
    E_v = 0. The Hurst means are fixed at the regime's definition (0.5, 0.8, 0.5); the other means and every scale
    are learned. The gradient moves y_v towards the regime that explains the entity's timing: the belief carries
    whether the timing is white, coloured or periodic. Responsibilities r_vk (the posterior regime) are reported
    in aux["regime"]: the coloured-versus-white reading of D-26.
    """

    name = "noise"
    S_LO, S_HI = -3.0, 3.0
    HURST = (0.5, 0.8, 0.5)

    def __init__(self, spec: LensSpec) -> None:
        super().__init__(spec)
        nf = len(NOISE_FEATURES)
        self.regime = nn.Linear(spec.cfg.d_hyp, len(NOISE_CLASSES))
        # Initial means: peak ratio about log(H_64) for white and coloured timing, higher for periodic timing.
        init = torch.zeros(len(NOISE_CLASSES), nf)
        if "log_peak_ratio" in NOISE_FEATURES:
            init[:, NOISE_FEATURES.index("log_peak_ratio")] = torch.tensor([1.5, 1.5, 3.5])
        self.mu = nn.Parameter(init)
        self.scale_raw = nn.Parameter(torch.zeros(len(NOISE_CLASSES), nf))
        hurst = torch.zeros(len(NOISE_CLASSES), nf, dtype=torch.bool)
        fixed = torch.zeros(len(NOISE_CLASSES), nf)
        if "hurst" in NOISE_FEATURES:
            j = NOISE_FEATURES.index("hurst")
            hurst[:, j] = True
            fixed[:, j] = torch.tensor(self.HURST)
        self.register_buffer("_hurst_mask", hurst, persistent=False)
        self.register_buffer("_hurst_value", fixed, persistent=False)
        self._hurst_mask: torch.Tensor
        self._hurst_value: torch.Tensor

    def prepare(self, inp: LensInputs) -> Prepared:
        assume("AS-36", by=__name__)
        mu = torch.where(self._hurst_mask, self._hurst_value, self.mu)                  # [K, F]
        log_s = self.S_LO + (self.S_HI - self.S_LO) * torch.sigmoid(self.scale_raw)     # [K, F]
        nu = inp.noise.to(mu.dtype)                                                     # [B, V, F]
        valid = inp.noise_valid
        z = (nu[:, :, None, :] - mu[None, None]) * torch.exp(-log_s)[None, None]        # [B, V, K, F]
        excess = -(log_s - self.S_LO)[None, None] - 0.5 * z * z                         # l_kf - l_max <= 0
        excess = torch.where(valid[:, :, None, :], excess, torch.zeros_like(excess))
        return {"loglik": excess.sum(-1)}                                               # [B, V, K]

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        v = inp.n_entities
        log_pi = torch.log_softmax(self.regime(y[:, :v]), dim=-1)                       # [B, V, K]
        joint = log_pi + prep["loglik"]
        e = -torch.logsumexp(joint, dim=-1) * inp.entity_mask.to(joint.dtype)           # [B, V] >= 0
        per = F.pad(e, (0, y.shape[1] - v))
        regime = torch.softmax(joint, dim=-1)
        return LensTerm(e.double().sum(-1), per, {"regime": regime})


class TopologyLens(LensEnergy):
    """E_top: robust pairwise MRF on the contact graph as of tau; compromise propagates along edges (AS-206).

    Mathematics (active entities u != v with hop-1 or hop-2 contact as of tau)
    ------------------------------------------------------------------------
        w_uv  = softplus( b_hop(1) + sum_p b_p [plane-p contact <= tau] )   for hop-1 pairs
              = softplus( b_hop(2) )                                        for hop-2 pairs
        s_uv  = ||A (y_u - y_v)||^2 / r
        E_top = 1/2 sum over u != v of w_uv log(1 + s_uv)

    Why it is an energy, and what its gradient means
    ------------------------------------------------
    A pairwise Markov random field over the contact graph: hosts that talk share risk, so their hypotheses should
    agree in the subspace A. The Lorentzian potential log(1 + s) is robust (Black and Rangarajan 1996): its
    gradient w / (1 + s) grad s fades for pairs that disagree strongly, so a compromised host does not drag every
    benign neighbour with it. The weights are learned from plane membership and hop. Null reading (AS-214): with
    the evidence removed, every active pair gets one learned weight w_null / (n - 1) (a mean-field coherence
    prior). Per-position contribution: half of each edge to each endpoint. >= 0.
    """

    name = "topology"

    def __init__(self, spec: LensSpec) -> None:
        super().__init__(spec)
        self.a = nn.Linear(spec.cfg.d_hyp, spec.rank, bias=False)
        self.b_hop = nn.Parameter(torch.zeros(2))
        self.b_plane = nn.Parameter(torch.zeros(spec.n_planes))
        self.b_null = nn.Parameter(torch.tensor(-2.0))

    def prepare(self, inp: LensInputs) -> Prepared:
        pairs = _pair_mask(inp.entity_mask)                                            # [B, V, V]
        logit1 = self.b_hop[0] + (inp.planes.to(self.b_plane.dtype) * self.b_plane).sum(-1)
        w = torch.where(inp.hop1, F.softplus(logit1), torch.zeros_like(logit1))
        w = w + torch.where(inp.hop2, F.softplus(self.b_hop[1]).expand_as(w), torch.zeros_like(w))
        n_act = inp.entity_mask.sum(-1).clamp_min(2).to(w.dtype)                      # [B]
        w_null = (F.softplus(self.b_null) / (n_act - 1.0))[:, None, None].expand_as(w)
        w = torch.where(inp.dropped[:, None, None], w_null, w)
        w = 0.5 * (w + w.transpose(1, 2)) * pairs.to(w.dtype)
        return {"w": w}

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        v = inp.n_entities
        r = self.a.out_features
        a = self.a(y[:, :v])                                                           # [B, V, r]
        phi = torch.log1p(_sq_dist(a, a) / r)                                          # [B, V, V]
        we = prep["w"] * phi
        per_ent = 0.5 * we.sum(-1)                                                     # half of each edge
        per = F.pad(per_ent, (0, y.shape[1] - v))
        # Per-trigger energy 1/2 sum over u != v of w phi: the pair sum accumulated in float64 (D-54).
        return LensTerm(0.5 * we.double().sum(dim=(1, 2)), per, {"edge_energy": we})


class TemporalLens(LensEnergy):
    """E_time: compatibility with the position's previous hypothesis, read from Imagination (AS-207).

    Mathematics (active positions i with a previous refined hypothesis y_i^prev within M_im triggers)
    -------------------------------------------------------------------------------------------
        e(dtau)     = learned embedding of the log-time bucket of dtau (D-49 buckets)
        f_time      = y^prev + MLP([y^prev ; e(dtau)])                    learned transition
        log pi(dtau) = 8 tanh( (W_pi e(dtau) + b_pi) / 8 )                diagonal precision
        E_time      = sum_i 1/2 sum_d pi_d (y_id - f_time,d)^2

    Why it is an energy, and what its gradient means
    ------------------------------------------------
    It is the negative log of a Gaussian transition p(y_tau | y_tau', dtau) without its normaliser: the belief at
    tau should be a plausible evolution of the belief at tau'. The precision depends on dtau, so a belief may
    change more after a long gap than after a short one. y^prev is detached (a stored belief, not a path for
    gradients across triggers). >= 0.
    """

    name = "temporal"

    def __init__(self, spec: LensSpec) -> None:
        super().__init__(spec)
        cfg = spec.cfg
        d_e = max(8, cfg.d_hyp // 4)
        self.dt_emb = LogDeltaBias(d_e, n_buckets=cfg.delta_buckets)   # a per-bucket learned vector
        with torch.no_grad():
            self.dt_emb.table.normal_(0.0, 0.02)
        self.mlp_in = nn.Linear(cfg.d_hyp + d_e, cfg.lens_hidden)
        self.mlp_out = nn.Linear(cfg.lens_hidden, cfg.d_hyp)
        self.log_prec = nn.Linear(d_e, cfg.d_hyp)

    def prepare(self, inp: LensInputs) -> Prepared:
        if inp.y_prev is None:
            return {}
        e = self.dt_emb(inp.prev_dt).to(inp.y_prev.dtype)                              # [B, N, d_e]
        f = inp.y_prev + self.mlp_out(F.silu(self.mlp_in(torch.cat([inp.y_prev, e], dim=-1))))
        lp = 8.0 * torch.tanh(self.log_prec(e) / 8.0)
        on = (inp.token_mask & inp.prev_valid).to(f.dtype)
        return {"f": f, "prec": torch.exp(lp), "on": on}

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        if not prep:
            z = y.new_zeros(y.shape[:2])
            # Exactly 0 in float64, still connected to y (so autograd.grad sees the input).
            return LensTerm((z.sum(-1) + 0.0 * y.sum(dim=(1, 2))).double(), z)
        per = 0.5 * (prep["prec"] * (y - prep["f"]) ** 2).sum(-1) * prep["on"]        # [B, N]
        # Per-trigger energy: the position sum accumulated in float64 (D-54).
        return LensTerm(per.double().sum(-1), per)


class CausalLens(LensEnergy):
    """E_cause: directed robust pairwise term weighted by TSTCT's Granger gates (AS-208).

    Mathematics (active entities u != v)
    -----------------------------------
        g_uv     = mean candidate gate of the TSTCT causal heads from u's states to v's states, as of tau
        s_uv     = ||A y_v - B y_u||^2 / r                      (A != B: the term is directed)
        E_cause  = gamma sum over u != v of g_uv log(1 + s_uv),   gamma = softplus(learned)

    Why it is an energy, and what its gradient means
    ------------------------------------------------
    Where TSTCT found lagged predictive influence from u to v (Granger 1969; Tank et al., TPAMI 2021, AS-10), the
    effect's hypothesis should be predictable from the cause's through a learned map (B y_u close to A y_v). The
    gradient moves y_v towards what its causes imply (and y_u towards explaining its effects), with the robust
    Lorentzian fading for pairs that disagree strongly. This is predictive (Granger) influence, not identified
    causal structure (AS-10). Per-position contribution: half of each directed pair to each endpoint. >= 0.
    """

    name = "causal"

    def __init__(self, spec: LensSpec) -> None:
        super().__init__(spec)
        self.a = nn.Linear(spec.cfg.d_hyp, spec.rank, bias=False)   # effect map
        self.b = nn.Linear(spec.cfg.d_hyp, spec.rank, bias=False)   # cause map
        self.gamma_raw = nn.Parameter(torch.tensor(1.0))

    def prepare(self, inp: LensInputs) -> Prepared:
        pairs = _pair_mask(inp.entity_mask)
        w = F.softplus(self.gamma_raw) * inp.cause.to(self.b.weight.dtype) * pairs.to(self.b.weight.dtype)
        return {"w": w}                                                                # [B, u, v]

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        v = inp.n_entities
        r = self.a.out_features
        ye = y[:, :v]
        phi = torch.log1p(_sq_dist(self.b(ye), self.a(ye)) / r)                        # [B, u, v]
        we = prep["w"] * phi
        per_ent = 0.5 * (we.sum(-1) + we.sum(-2))
        per = F.pad(per_ent, (0, y.shape[1] - v))
        # Per-trigger energy: the directed-pair sum accumulated in float64 (D-54).
        return LensTerm(we.double().sum(dim=(1, 2)), per, {"edge_energy": we})


class PhysicsLens(LensEnergy):
    """lambda_phys Phi_phys on the decoded believed next state of each entity (D-18, D-37, AS-15, AS-209).

    Mathematics (per trigger, R = active entity rows of window b)
    -------------------------------------------------------------
        z_v     = [mu_next(y_v) ; softmax(l_next(y_v))]           expected next latent (differentiable)
        x_v     = Decoder(z_v | role_v, planes_v)                 decoded fields
        Phi_b   = sum_c w_c ||m_c * r_c(x)||^2 over the rows of b  the shared physics term (physics/term.py)
        E_phys  = lambda_phys log(1 + Phi_b / |R_b|)

    Why it is an energy, and what its gradient means
    ------------------------------------------------
    Phi = 0 inside the physical boundary and grows with the violation, so the term never moves a hypothesis whose
    decoded next state is possible; its gradient pushes beliefs whose decoded consequences are impossible back to
    the boundary: physics as a boundary against hallucination, never as a detector (D-18). Dividing by the row
    count makes it per entity, and log(1 + .) keeps the zero set and the ordering while bounding the gradient, so
    one absurd decode cannot swamp the other lenses. lambda_phys is configuration (not learned). Without a
    decoder or a physics term the term is exactly 0.
    """

    name = "physics"
    scaled = False

    def raw(self, y: torch.Tensor, inp: LensInputs, prep: Prepared) -> LensTerm:
        b = y.shape[0]
        v = inp.n_entities
        zero = y.sum(dim=(1, 2)).double() * 0.0                                       # [B] float64, keeps the graph
        if inp.decoder is None or inp.physics is None:
            return LensTerm(zero, None)
        assume("AS-15", by=__name__)
        mean, _, logits = inp.heads.next_latent(y[:, :v])
        z = inp.heads.latent_vector(mean, logits)                                     # [B, V, dz]
        rows = z.reshape(b * v, -1)
        decoded = inp.decoder.decode_fields(rows, inp.entity_role.reshape(-1), inp.entity_planes.reshape(b * v, -1))
        values, contributing = inp.decoder.physics_inputs(decoded)
        active = inp.entity_mask.reshape(-1)
        row_b = torch.arange(b, device=y.device).repeat_interleave(v)
        terms = []
        for bi in range(b):
            # Restrict Phi to this window's active rows by masking `contributing` (PhysicsTerm sums rows).
            sel = active & (row_b == bi)
            contrib_b = {f: m.bool() & sel for f, m in contributing.items()}
            phi_b, _ = inp.physics(values, contrib_b)
            # lambda log(1 + Phi / n) in float64 (D-54): Phi_b may be large (raw units), the log is the output.
            n_b = sel.sum().clamp_min(1).to(torch.float64)
            terms.append(self.spec.cfg.lambda_phys * torch.log1p(phi_b.to(torch.float64) / n_b))
        return LensTerm(torch.stack(terms) + zero, None)


class TotalEnergy(nn.Module):
    """E_total = sum of the configured lens terms + lambda_phys Phi_phys (D-42; ADR-0007).

    `prepare(inp)` once per trigger; `forward(y, inp, prep)` -> (E_total [B], {name: LensTerm}). The
    mechanism-design term, when listed, is attached to the game lens and shares its prepared quantities.
    """

    def __init__(self, spec: LensSpec, names: Sequence[str]) -> None:
        super().__init__()
        if len(set(names)) != len(names):
            raise InvariantViolation(f"lens names must be distinct: {tuple(names)}")
        terms: dict[str, nn.Module] = {}
        for n in names:
            terms[n] = LENSES.build(n, spec)
        if "physics" not in terms:
            terms["physics"] = LENSES.build("physics", spec)
        self.terms = nn.ModuleDict(terms)
        if "mechanism-design" in self.terms:
            if "game" not in self.terms:
                raise InvalidOption("the mechanism-design term needs the game lens")
            cast(MechanismDesignLens, self.terms["mechanism-design"]).attach(cast(GameLens, self.terms["game"]))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.terms.keys())

    def lens(self, name: str) -> LensEnergy:
        return cast(LensEnergy, self.terms[name])

    def prepare(self, inp: LensInputs) -> dict[str, Prepared]:
        out = {n: self.lens(n).prepare(inp) for n in self.names if n != "mechanism-design"}
        if "mechanism-design" in self.terms:
            out["mechanism-design"] = out["game"]          # one prepared object: the equilibrium is computed once
        return out

    def forward(self, y: torch.Tensor, inp: LensInputs, prep: Mapping[str, Prepared]) -> tuple[torch.Tensor, dict[str, LensTerm]]:
        """-> (E_total float64 [B], {name: LensTerm with float64 energy [B]}) (D-54)."""
        out: dict[str, LensTerm] = {n: self.lens(n)(y, inp, prep[n]) for n in self.names}
        if "game" in self.terms:
            cast(GameLens, self.terms["game"]).clear_cache()
        # The lens sum accumulated in float64 (each term is float64 already; `.double()` is exact).
        total = torch.stack([t.energy.double() for t in out.values()], dim=0).sum(0)
        return total, out


LENSES: Registry[Any] = Registry("TAAFT lens")

LENSES.register("belief-trust", summary="robust trust-weighted evidence energy and priors (POMDP belief, trust)")(BeliefTrustLens)
LENSES.register("game", summary="adversary exploitability against the defender's commitment in a security game (AS-723)")(GameLens)
LENSES.register("information", summary="informativeness-priced departure from the null hypothesis (AS-204)")(InformationLens)
LENSES.register("topology", summary="robust pairwise MRF on the contact graph as of tau (AS-206)")(TopologyLens)
LENSES.register("temporal", summary="transition compatibility with the previous belief (AS-207)")(TemporalLens)
LENSES.register("causal", summary="directed robust term weighted by TSTCT Granger gates (AS-208)")(CausalLens)
LENSES.register("physics", summary="lambda_phys log(1 + Phi_phys / n) on decoded believed next states (D-18, AS-209)")(PhysicsLens)
LENSES.register("energy", summary="E_total: the sum of the lens terms and lambda Phi_phys (D-42), not a term of its own")(TotalEnergy)
LENSES.register("mechanism-design", requires={"D-24": MECHANISM_IN_TAAFT},
                summary="participation gap of the adversary hypotheses as its own term (D-24, AS-724)")(MechanismDesignLens)
LENSES.register("noise", requires={"D-26": (NOISE_OWN_TERM,)},
                summary="white, coloured or periodic timing regime as its own term (D-26, AS-725)")(NoiseLens)
