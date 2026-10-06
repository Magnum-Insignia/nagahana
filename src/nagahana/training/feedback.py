"""Human feedback in training: stage 3 (training-phase feedback) and stage 4 (confidence calibration).

Purpose
-------
Stage 3 trains the complete architecture "with training-phase human feedback" and stage 4 calibrates
confidence "with human feedback". The Verifier's learning methods (process reward, trust value,
calibration policy; RLHF and RLVR as they are added to models/verifier) plug into the stages through
two contracts defined here, so a new method is a registered name in the stage configuration and never a
change to the training loop:

FeedbackLearner (stage 3)
    name                          registry key, listed in `Stage3Options.feedback`
    target                        "policy": its loss joins the stage-3 objective and trains the
                                  Forecaster, Advisor or TAAFT through the same optimiser and sharding;
                                  "verifier": its loss trains the Verifier in a step of its own
    modules()                     the modules whose parameters the loss trains
    loss(batch) -> (loss, logs)   the loss on one trigger-bearing micro-batch (`FeedbackBatch`)
    state_dict() / load_state_dict(state)   learner state kept across steps (checkpointed)

Calibrator (stage 4)
    name                          registry key, listed in `Stage4Config.calibration`
    fit(pairs) -> proposal | None  a `CalibrationProposal` from resolved outcome-forecast pairs

A learner factory is called as `factory(model, options)` with the stage-3 options (`Stage3Options`), a
calibrator factory as `factory(model, settings)` with the stage-4 settings (`Stage4Config`); a factory
reads its own fields from them and nothing else.

The human gate (D-21; ADR-0004)
-------------------------------
No feedback changes a weight or a temperature without a human: every learner runs only when the run
holds a HumanCommand("update-weights"), checked with `models.verifier.gate.require_command` before any
loss is computed; a calibration proposal is applied only through `gate.apply_calibration` with a
HumanCommand("apply-calibration"). Without the command, stage 3 skips the learners and records why.
Dataset annotations count as human-supplied truth for public datasets (AS-25, D-17).

Registered implementations
--------------------------
Defined here:

- "verifier-heads" (target "verifier"; AS-415): the process-reward model on step labels (an imagined step
  is plausible when its target is the realised one), the trust value head on resolved triggers (did the
  decision P_inf(K) >= 1/2 match the outcome), the calibration policy head towards the maximum-likelihood
  temperature once `min_pairs` resolved pairs with both outcomes exist (Lightman et al., arXiv:2305.20050;
  Guo et al., ICML 2017, arXiv:1706.04599). The Monitor (drift statistics) observes the latents.
- "ml-temperature" (calibrator): the maximum-likelihood temperature of each output family on its
  resolved pairs (`verifier.calibration.ml_temperature`).

The Verifier's reinforcement learning from human and verifiable rewards (D-65, models/verifier/hooks.py),
imported when a run names them:

- "verifier-rlvr" (target "policy"): group-relative policy optimisation of the Forecaster with the batch's
  confirmed outcomes as verifiable rewards.
- "verifier-rlhf" (target "policy"): direct preference optimisation of the Forecaster or the Advisor on
  the analyst preferences of a feedback ledger (`Stage3Options.rlhf_ledger`, `rlhf_situations`).
- "verifier-rlcd" (target "verifier"): the process-reward model, the trust head on the Brier score of
  decision correctness and the calibration head toward the calibrated-decision temperature; it trains the
  same heads as "verifier-heads", so a run names one of the two (`Stage3Options` refuses both).
- "rlcd-temperature" (calibrator): the calibrated-decision temperature of every output family.

Their settings come from the Verifier's `FeedbackLearningConfig` (a YAML file named by
`Stage3Options.feedback_config` and `Stage4Config.feedback_config`; its defaults when None).

Decisions: D-07, D-17, D-21, D-45, D-65. Assumptions: AS-25, AS-415, AS-595.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.models.batch import AnalysisOut, ForecastOut, LabelBatch, LatentOut, WindowBatch
from nagahana.models.forecaster.losses import NO_TARGET, ForecasterTargets
from nagahana.models.verifier.calibration import logit, ml_temperature
from nagahana.models.verifier.gate import UPDATE_WEIGHTS, require_command
from nagahana.models.verifier.heads import forecast_features, monitor_features, reliability_features
from nagahana.models.verifier.monitor import Monitor
from nagahana.models.verifier.reports import OUTPUT_FAMILIES, CalibrationProposal
from nagahana.roles.contracts import HumanCommand, OutcomeForecastPair
from nagahana.training.assumptions import use

TARGETS: tuple[str, ...] = ("policy", "verifier")


@dataclass
class FeedbackBatch:
    """What stage 3 hands to feedback learners for one trigger-bearing micro-batch.

    analysis: TAAFT's output (with gradient for "policy" learners, detached for "verifier" learners).
    forecast: imagined routes at every trigger (no gradient). targets: the realised steps, events and
    censoring (`forecaster.losses`). outcomes: [B, M] float, 1 / 0 / NaN (unresolved) per trigger.
    latents: the frozen perception's posterior (for the Monitor). window, labels: the batch.
    step: the optimiser step. command: the run's HumanCommand.
    """

    analysis: AnalysisOut
    forecast: ForecastOut
    targets: ForecasterTargets
    outcomes: torch.Tensor
    latents: LatentOut
    window: WindowBatch
    labels: LabelBatch
    step: int
    command: HumanCommand


@runtime_checkable
class FeedbackLearner(Protocol):
    """Stage-3 contract (module docstring)."""

    name: str
    target: str

    def modules(self) -> Sequence[nn.Module]: ...

    def loss(self, batch: FeedbackBatch) -> tuple[torch.Tensor, dict[str, float]]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


@dataclass
class CalibrationPairs:
    """Resolved outcome-forecast pairs per output family: family -> (p float64 [n], y float64 [n])."""

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]

    def n(self, family: str) -> int:
        return int(self.pairs[family][0].numel()) if family in self.pairs else 0


@runtime_checkable
class Calibrator(Protocol):
    """Stage-4 contract (module docstring)."""

    name: str

    def fit(self, pairs: CalibrationPairs) -> CalibrationProposal | None: ...


LearnerFactory = Callable[[Any, Any], FeedbackLearner]
CalibratorFactory = Callable[[Any, Any], Calibrator]
_LEARNERS: dict[str, tuple[LearnerFactory, str]] = {}
_CALIBRATORS: dict[str, tuple[CalibratorFactory, str]] = {}


def register_learner(name: str, *, summary: str) -> Callable[[LearnerFactory], LearnerFactory]:
    """Register a stage-3 feedback learner factory `factory(model, settings) -> FeedbackLearner`."""
    def deco(factory: LearnerFactory) -> LearnerFactory:
        if name in _LEARNERS:
            raise InvariantViolation(f"feedback learner {name!r} is registered twice")
        _LEARNERS[name] = (factory, summary)
        return factory
    return deco


def register_calibrator(name: str, *, summary: str) -> Callable[[CalibratorFactory], CalibratorFactory]:
    """Register a stage-4 calibrator factory `factory(model, settings) -> Calibrator`."""
    def deco(factory: CalibratorFactory) -> CalibratorFactory:
        if name in _CALIBRATORS:
            raise InvariantViolation(f"calibrator {name!r} is registered twice")
        _CALIBRATORS[name] = (factory, summary)
        return factory
    return deco


def learners() -> dict[str, str]:
    """Registered learners -> summary."""
    return {k: v[1] for k, v in _LEARNERS.items()}


def calibrators() -> dict[str, str]:
    """Registered calibrators -> summary."""
    return {k: v[1] for k, v in _CALIBRATORS.items()}


def build_learners(names: Sequence[str], model: Any, settings: Any) -> list[FeedbackLearner]:
    """The configured learners, checked against the contract."""
    use("AS-595", by=__name__)
    out: list[FeedbackLearner] = []
    for n in names:
        if n not in _LEARNERS:
            raise InvariantViolation(f"unknown feedback learner {n!r}; registered: {sorted(_LEARNERS)}")
        learner = _LEARNERS[n][0](model, settings)
        if not isinstance(learner, FeedbackLearner) or learner.target not in TARGETS:
            raise InvariantViolation(f"{n!r} does not satisfy the FeedbackLearner contract (target in {TARGETS})")
        out.append(learner)
    return out


def build_calibrators(names: Sequence[str], model: Any, settings: Any = None) -> list[Calibrator]:
    """The configured calibrators, checked against the contract (`settings`: the stage-4 settings)."""
    out: list[Calibrator] = []
    for n in names:
        if n not in _CALIBRATORS:
            raise InvariantViolation(f"unknown calibrator {n!r}; registered: {sorted(_CALIBRATORS)}")
        cal = _CALIBRATORS[n][0](model, settings)
        if not isinstance(cal, Calibrator):
            raise InvariantViolation(f"{n!r} does not satisfy the Calibrator contract")
        out.append(cal)
    return out


def authorise(command: HumanCommand | None) -> HumanCommand:
    """The run's feedback command, refused unless it is a HumanCommand("update-weights") (D-21)."""
    return require_command(command, UPDATE_WEIGHTS)


def prm_step_labels(fo: ForecastOut, target_entity: torch.Tensor) -> torch.Tensor:
    """[B, M, N, K] step labels (AS-415) from imagined targets and realised targets [B, M, K]."""
    imagined = fo.route_actions[..., 1]                                          # [B, M, N, K] (-1 = no target)
    real = target_entity[:, :, None, :].expand_as(imagined)
    known = (real >= 0) | (real == NO_TARGET)
    match = torch.where(real == NO_TARGET, imagined < 0, imagined == real)
    return torch.where(known, match.long(), torch.full_like(imagined, -1))


@dataclass
class _VerifierHeadsState:
    """Resolved pairs per output family (the ledger) and the Monitor, across steps."""

    monitor: Monitor
    pairs: dict[str, tuple[list[float], list[float]]] = field(default_factory=lambda: {f: ([], []) for f in OUTPUT_FAMILIES})


class VerifierHeadsLearner:
    """Built-in "verifier-heads" learner (module docstring; AS-415)."""

    name = "verifier-heads"
    target = "verifier"

    def __init__(self, model: Any, settings: Any = None) -> None:
        use("AS-595", by=__name__)
        self.model = model
        cfg = model.cfg
        self.cfg = cfg
        self.state = _VerifierHeadsState(monitor=Monitor(cfg.verifier, region_dims={"environment": cfg.latent_dim,
                                                                                      "imagination": cfg.taaft.d_hyp}))

    def modules(self) -> Sequence[nn.Module]:
        return [self.model.verifier]

    def loss(self, batch: FeedbackBatch) -> tuple[torch.Tensor, dict[str, float]]:
        """PRM + trust head (+ calibration head when enough pairs), on detached inputs (module docstring)."""
        authorise(batch.command)
        cfg = self.cfg
        v = self.model.verifier
        an, fo = batch.analysis, batch.forecast
        logits = v.prm(an, fo)
        labels = prm_step_labels(fo, batch.targets.target_entity)
        l_prm = v.prm.loss(logits, labels)
        y = batch.outcomes
        ff = forecast_features(fo, torch.sigmoid(logits.detach()))
        mf = monitor_features(self.state.monitor.report(), cusum_h=cfg.verifier.cusum_h, ph_lambda=cfg.verifier.ph_lambda)
        trust_logit = v.trust(ff, mf)                                                   # [B, M]
        ok = torch.isfinite(y)
        p_k = fo.p_inf[..., -1]
        correct = ((p_k >= 0.5).to(y.dtype) == y).to(trust_logit.dtype)
        l_trust = (F.binary_cross_entropy_with_logits(trust_logit[ok], correct[ok]) if bool(ok.any())
                   else trust_logit.sum() * 0.0)
        # The ledger of resolved pairs (P_inf family) and the Monitor's statistics of both regions.
        for pv, yv in zip(p_k[ok].tolist(), y[ok].tolist(), strict=True):
            pc = float(min(max(pv, 0.0), 1.0))
            self.state.pairs["p_inf"][0].append(pc)
            self.state.pairs["p_inf"][1].append(float(yv))
            self.state.monitor.record_resolution(OutcomeForecastPair("train", pc, bool(yv > 0.5)))
        real = batch.window.positions.mask
        self.state.monitor.observe_latents("environment", batch.latents.z.detach()[real])
        self.state.monitor.observe_latents("imagination", an.y.detach()[an.token_mask])
        l_cal = l_prm * 0.0
        pv_all, yv_all = self.state.pairs["p_inf"]
        n_pairs = len(pv_all)
        if n_pairs >= cfg.verifier.min_pairs and 0 < sum(yv_all) < n_pairs:
            # float64 (D-54): the stored probabilities are Python doubles.
            p_t, y_t = torch.tensor(pv_all, dtype=torch.float64), torch.tensor(yv_all, dtype=torch.float64)
            t_ml = ml_temperature(logit(p_t), y_t, t_min=cfg.verifier.temperature_min, t_max=cfg.verifier.temperature_max)
            feats = reliability_features(p_t, y_t, bins=cfg.verifier.reliability_bins)[None]
            log_t = v.calibration(feats.to(trust_logit.dtype), torch.tensor([OUTPUT_FAMILIES.index("p_inf")]))
            l_cal = v.calibration.loss(log_t, torch.tensor([t_ml], dtype=log_t.dtype))
        total = l_prm + l_trust + l_cal
        return total, {"prm": float(l_prm.detach()), "trust": float(l_trust.detach()), "calibration": float(l_cal.detach()),
                       "resolved_pairs": float(n_pairs), "prm_labelled": float((labels >= 0).sum())}

    def state_dict(self) -> dict[str, Any]:
        return {"monitor": self.state.monitor, "pairs": {k: (list(a), list(b)) for k, (a, b) in self.state.pairs.items()}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        monitor = state["monitor"]
        if not isinstance(monitor, Monitor):
            raise InvariantViolation("malformed verifier-heads state")
        self.state = _VerifierHeadsState(monitor=monitor,
                                         pairs={k: (list(a), list(b)) for k, (a, b) in dict(state["pairs"]).items()})


class MLTemperatureCalibrator:
    """Built-in "ml-temperature" calibrator (module docstring)."""

    name = "ml-temperature"

    def __init__(self, model: Any, settings: Any = None) -> None:
        self.vc = model.cfg.verifier

    def fit(self, pairs: CalibrationPairs) -> CalibrationProposal | None:
        temps: dict[str, float] = {}
        counts: dict[str, int] = {}
        for fam, (p, y) in pairs.pairs.items():
            yd = y.to(torch.float64)
            if p.numel() < 2 or not (0 < float(yd.sum()) < float(yd.numel())):
                continue                                                             # both outcomes are needed
            temps[fam] = ml_temperature(logit(p.to(torch.float64)), yd, t_min=self.vc.temperature_min,
                                        t_max=self.vc.temperature_max)
            counts[fam] = int(p.numel())
        if not temps:
            return None
        return CalibrationProposal(temperatures=temps, source="ml-fit", n_pairs=counts, t_min=self.vc.temperature_min,
                                   t_max=self.vc.temperature_max)


def _verifier_hooks() -> Any:
    """models/verifier/hooks.py, imported when a run names one of its learners or calibrators (D-65)."""
    try:
        from nagahana.models.verifier import hooks
    except ImportError as exc:                                                  # an installation without the module
        raise InvariantViolation("the Verifier's feedback hooks (models/verifier/hooks.py) cannot be imported") from exc
    return hooks


def feedback_learning_config(path: str | None) -> Any:
    """The Verifier's `FeedbackLearningConfig`: the YAML file at `path`, validated, or its defaults when None."""
    from nagahana.models.verifier.config import FeedbackLearningConfig, feedback_config_from_mapping

    if path is None:
        return FeedbackLearningConfig()
    from nagahana.core.config import load_yaml

    return feedback_config_from_mapping(load_yaml(path))


def _setting(settings: Any, name: str) -> Any:
    # The factories read named fields of the stage settings; a settings object without the field is a
    # wiring error, never a silent default.
    if settings is None or not hasattr(settings, name):
        raise InvariantViolation(f"the stage settings handed to the factory have no field {name!r}")
    return getattr(settings, name)


def _rlvr(model: Any, settings: Any) -> FeedbackLearner:
    return _verifier_hooks().RLVRStageLearner(model, settings, cfg=feedback_learning_config(_setting(settings, "feedback_config")))


def _rlhf(model: Any, settings: Any) -> FeedbackLearner:
    ledger, situations = _setting(settings, "rlhf_ledger"), _setting(settings, "rlhf_situations")
    if ledger is None or situations is None:
        raise InvariantViolation("verifier-rlhf needs stage3.options.rlhf_ledger and rlhf_situations")
    return _verifier_hooks().RLHFStageLearner(model, settings, ledger_path=ledger, situations_path=situations,
                                              policy=_setting(settings, "rlhf_policy"),
                                              per_step=_setting(settings, "rlhf_per_step"),
                                              cfg=feedback_learning_config(_setting(settings, "feedback_config")))


def _rlcd(model: Any, settings: Any) -> FeedbackLearner:
    return _verifier_hooks().RLCDStageLearner(model, settings, cfg=feedback_learning_config(_setting(settings, "feedback_config")))


def _rlcd_temperature(model: Any, settings: Any) -> Calibrator:
    return _verifier_hooks().RLCDCalibrator(model, cfg=feedback_learning_config(_setting(settings, "feedback_config")))


register_learner("verifier-heads", summary="PRM on step labels, trust head on resolved triggers, calibration head (AS-415)")(
    lambda model, settings: VerifierHeadsLearner(model, settings))
register_learner("verifier-rlvr", summary="GRPO of the Forecaster policy with confirmed outcomes as verifiable rewards (D-65)")(
    _rlvr)
register_learner("verifier-rlhf", summary="DPO of the Forecaster or Advisor on analyst preferences of a ledger (D-65)")(_rlhf)
register_learner("verifier-rlcd", summary="PRM, Brier trust head and calibrated-decision temperature head (D-65)")(_rlcd)
register_calibrator("ml-temperature", summary="maximum-likelihood temperature per output family")(
    lambda model, settings: MLTemperatureCalibrator(model, settings))
register_calibrator("rlcd-temperature", summary="calibrated-decision temperature per output family (D-65)")(_rlcd_temperature)

#: Learners that train the same Verifier heads with different objectives: a run names at most one.
VERIFIER_HEAD_ALTERNATIVES: tuple[str, ...] = ("verifier-heads", "verifier-rlcd")


__all__ = ["CalibrationPairs", "Calibrator", "FeedbackBatch", "FeedbackLearner", "MLTemperatureCalibrator", "TARGETS",
           "VERIFIER_HEAD_ALTERNATIVES", "VerifierHeadsLearner", "authorise", "build_calibrators", "build_learners",
           "calibrators", "feedback_learning_config", "learners", "prm_step_labels", "register_calibrator", "register_learner"]
