"""Hooks of the Verifier's feedback learning into the training stages (D-62 numbering; D-21, D-65; AS-847).

Stage 3 (full training) runs feedback learners on every trigger-bearing micro-batch and Stage 4
(zero-shot validation and calibration) runs calibrators on resolved outcome-forecast pairs. The training
package defines both contracts (a learner: name, target "policy" or "verifier", modules(), loss(batch),
state_dict(), load_state_dict(); a calibrator: name, fit(pairs)) and registers implementations by name.
The classes below satisfy those contracts structurally, without importing the training package:

    RLVRStageLearner    "verifier-rlvr", target "policy": GRPO of the Forecaster's policy with the batch's
                        confirmed outcomes as verifiable rewards (online.OnlineFeedback.rlvr_loss)
    RLHFStageLearner    "verifier-rlhf", target "policy": DPO on the analyst preferences of a ledger
    RLCDStageLearner    "verifier-rlcd", target "verifier": the process-reward model on step labels
                        (AS-415), the trust head on the RLCR Brier score of decision correctness and the
                        calibration head toward the RLCD temperature (rlcd.py), with the Monitor observing the
                        latents; it takes the place of the maximum-likelihood variant of the same heads
    RLCDCalibrator      "rlcd-temperature": the RLCD temperature per output family from resolved pairs

Every learner checks the run's HumanCommand("update-weights") before computing anything (D-21), exactly as
the stage does. A batch is read through the attributes the training contract documents: analysis,
forecast, targets (forecaster.losses.ForecasterTargets), outcomes, latents, window, labels, step, command.

Dataset annotations count as human-supplied truth (AS-25, D-17): a trigger's survival targets become an
OutcomeConfirmation (rlvr.outcome_from_survival) whose stages are the step labels of the window.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.models.batch import ForecastOut
from nagahana.models.forecaster.losses import NO_TARGET
from nagahana.models.verifier.config import FeedbackLearningConfig
from nagahana.models.verifier.feedback import OutcomeConfirmation, PreferenceFeedback
from nagahana.models.verifier.gate import UPDATE_WEIGHTS, command_id, require_command
from nagahana.models.verifier.heads import forecast_features, monitor_features, reliability_features
from nagahana.models.verifier.learning import generator_for
from nagahana.models.verifier.ledger import FeedbackLedger
from nagahana.models.verifier.monitor import Monitor
from nagahana.models.verifier.online import OnlineFeedback
from nagahana.models.verifier.reports import OUTPUT_FAMILIES, CalibrationProposal
from nagahana.models.verifier.rlcd import rlcd_temperature, rlcr_trust_loss
from nagahana.models.verifier.rlvr import outcome_from_survival
from nagahana.models.verifier.situations import Situation, load_situations, trigger_situation
from nagahana.roles.contracts import OutcomeForecastPair


def step_plausibility_labels(fo: ForecastOut, target_entity: torch.Tensor) -> torch.Tensor:
    """[B, M, N, K] PRM step labels (AS-415) from imagined targets and realised targets [B, M, K].

    An imagined step is plausible (1) when its target is the realised target of that step (the responder
    of the first malicious update, or no target), implausible (0) when the realised target is known and
    differs, unknown (-1) otherwise.
    """
    imagined = fo.route_actions[..., 1]                                          # [B, M, N, K] (-1 = no target)
    real = target_entity[:, :, None, :].expand_as(imagined)
    known = (real >= 0) | (real == NO_TARGET)
    match = torch.where(real == NO_TARGET, imagined < 0, imagined == real)
    return torch.where(known, match.long(), torch.full_like(imagined, -1))


def training_triggers(batch: Any, *, max_triggers: int, seed: int) -> tuple[list[Situation], list[OutcomeConfirmation]]:
    """Situations and dataset outcomes of the usable triggers of a training micro-batch (module docstring).

    At most `max_triggers` are taken, a seeded random subset when there are more (AS-848).
    """
    targets = batch.targets
    mask = batch.window.triggers.mask & targets.usable
    pos = torch.nonzero(mask).tolist()
    if len(pos) > max_triggers:
        gen = generator_for(seed, "stage3", "triggers", int(batch.step))
        pos = [pos[i] for i in sorted(torch.randperm(len(pos), generator=gen)[:max_triggers].tolist())]
    cid = command_id(batch.command)
    k = int(targets.stage.shape[-1]) - 1
    sits: list[Situation] = []
    outs: list[OutcomeConfirmation] = []
    for b, m in pos:
        sid = f"stage3:{int(batch.step)}:{b}:{m}"
        t_abs = float(batch.window.triggers.time[b, m]) + float(batch.window.origin[b])
        o = outcome_from_survival(refers_to=sid, event_step=int(targets.event_step[b, m]), censored=bool(targets.censored[b, m]),
                                  horizon_k=k, stage_labels=[int(x) for x in targets.stage[b, m].tolist()],
                                  analyst="dataset-annotation", time=t_abs, command_id=cid)
        if o is None:
            continue
        sits.append(trigger_situation(sid, batch.analysis, b, m, forecast=batch.forecast, time=t_abs))
        outs.append(o)
    return sits, outs


class RLVRStageLearner:
    """Stage-3 learner "verifier-rlvr" (module docstring). The reference is the policy at construction."""

    name = "verifier-rlvr"
    target = "policy"

    def __init__(self, model: Any, settings: Any = None, *, cfg: FeedbackLearningConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg if cfg is not None else FeedbackLearningConfig()
        self.online = OnlineFeedback(self.cfg, forecaster=model.forecaster)

    def modules(self) -> Sequence[nn.Module]:
        return [self.model.forecaster]

    def loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        require_command(batch.command, UPDATE_WEIGHTS)                           # D-21: before anything is computed
        sits, outs = training_triggers(batch, max_triggers=self.cfg.grpo.triggers_per_iteration, seed=self.cfg.seed)
        loss, stats = self.online.rlvr_loss(sits, outs, step=int(batch.step))
        return loss, stats

    def state_dict(self) -> dict[str, Any]:
        return {"reference_hash": self.online.ref_forecaster.hash, "reference": dict(self.online.ref_forecaster.tensors)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ref = dict(state["reference"])
        if set(ref) != set(self.online.ref_forecaster.tensors):
            raise InvariantViolation("the saved RLVR reference does not match the Forecaster's parameters")
        self.online.ref_forecaster.tensors = {k: v.detach().clone() for k, v in ref.items()}
        self.online.ref_forecaster.hash = str(state["reference_hash"])


class RLHFStageLearner:
    """Stage-3 learner "verifier-rlhf": DPO on the preferences of a feedback ledger (module docstring).

    ledger_path, situations_path: the ledger file and the situations file (situations.save_situations);
    target: "forecaster" or "advisor"; per_step: preferences sampled per micro-batch.
    """

    name = "verifier-rlhf"
    target = "policy"

    def __init__(self, model: Any, settings: Any = None, *, ledger_path: str | Path, situations_path: str | Path,
                 policy: str = "forecaster", per_step: int = 16, cfg: FeedbackLearningConfig | None = None) -> None:
        if policy not in ("forecaster", "advisor"):
            raise InvariantViolation("policy must be 'forecaster' or 'advisor'")
        self.model, self.policy = model, policy
        self.cfg = cfg if cfg is not None else FeedbackLearningConfig()
        ledger = FeedbackLedger(ledger_path, fsync=False)
        ledger.verify()
        prefs = [e for _, e in ledger.feedback(("preference",)) if isinstance(e, PreferenceFeedback) and e.target == policy]
        if not prefs:
            raise InvariantViolation(f"the ledger holds no preference for the {policy}")
        self.preferences: list[PreferenceFeedback] = prefs
        self.situations: dict[str, Situation] = load_situations(situations_path)
        self.per_step = int(per_step)
        self.online = OnlineFeedback(self.cfg, forecaster=model.forecaster, advisor=model.advisor)

    def modules(self) -> Sequence[nn.Module]:
        return [self.model.forecaster if self.policy == "forecaster" else self.model.advisor]

    def loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        require_command(batch.command, UPDATE_WEIGHTS)
        gen = generator_for(self.cfg.seed, "stage3", "rlhf", int(batch.step))
        idx = torch.randperm(len(self.preferences), generator=gen)[: self.per_step].tolist()
        return self.online.dpo_loss([self.preferences[i] for i in idx], self.situations, target=self.policy)

    def state_dict(self) -> dict[str, Any]:
        snap = self.online.ref_forecaster if self.policy == "forecaster" else self.online.ref_advisor
        assert snap is not None
        return {"reference_hash": snap.hash, "reference": dict(snap.tensors)}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        snap = self.online.ref_forecaster if self.policy == "forecaster" else self.online.ref_advisor
        assert snap is not None
        snap.tensors = {k: v.detach().clone() for k, v in dict(state["reference"]).items()}
        snap.hash = str(state["reference_hash"])


class RLCDStageLearner:
    """Stage-3 learner "verifier-rlcd" (module docstring): PRM + RLCR trust head + RLCD calibration head."""

    name = "verifier-rlcd"
    target = "verifier"

    def __init__(self, model: Any, settings: Any = None, *, cfg: FeedbackLearningConfig | None = None) -> None:
        self.model = model
        self.cfg = cfg if cfg is not None else FeedbackLearningConfig()
        mc = model.cfg
        self.monitor = Monitor(mc.verifier, region_dims={"environment": mc.latent_dim, "imagination": mc.taaft.d_hyp})
        self.pairs: tuple[list[float], list[float]] = ([], [])

    def modules(self) -> Sequence[nn.Module]:
        return [self.model.verifier]

    def loss(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        require_command(batch.command, UPDATE_WEIGHTS)                           # D-21: before anything is computed
        vc = self.model.cfg.verifier
        rc = self.cfg.rlcd
        v = self.model.verifier
        an, fo = batch.analysis, batch.forecast
        logits = v.prm(an, fo)
        labels = step_plausibility_labels(fo, batch.targets.target_entity)
        l_prm = v.prm.loss(logits, labels)
        y = batch.outcomes
        ff = forecast_features(fo, torch.sigmoid(logits.detach()))
        mf = monitor_features(self.monitor.report(), cusum_h=vc.cusum_h, ph_lambda=vc.ph_lambda)
        trust_logit = v.trust(ff, mf)                                             # [B, M]
        ok = torch.isfinite(y)
        p_k = fo.p_inf[..., -1]
        correct = ((p_k >= rc.decision_threshold).to(y.dtype) == y).to(torch.float64)
        l_trust = rlcr_trust_loss(trust_logit[ok], correct[ok]) if bool(ok.any()) else trust_logit.sum() * 0.0
        for pv, yv in zip(p_k[ok].tolist(), y[ok].tolist(), strict=True):
            pc = float(min(max(pv, 0.0), 1.0))
            self.pairs[0].append(pc)
            self.pairs[1].append(float(yv))
            self.monitor.record_resolution(OutcomeForecastPair("train", pc, bool(yv > 0.5)))
        real = batch.window.positions.mask
        self.monitor.observe_latents("environment", batch.latents.z.detach()[real])
        self.monitor.observe_latents("imagination", an.y.detach()[an.token_mask])
        l_cal = l_prm * 0.0
        n_pairs = len(self.pairs[0])
        if n_pairs >= rc.min_pairs and 0 < sum(self.pairs[1]) < n_pairs:
            p_t = torch.tensor(self.pairs[0], dtype=torch.float64)
            y_t = torch.tensor(self.pairs[1], dtype=torch.float64)
            t_star = rlcd_temperature(p_t, y_t, score=rc.score, t_min=vc.temperature_min, t_max=vc.temperature_max,
                                      grid=rc.grid, tol=rc.tol)
            feats = reliability_features(p_t, y_t, bins=vc.reliability_bins)[None]
            log_t = v.calibration(feats, torch.tensor([OUTPUT_FAMILIES.index("p_inf")]))
            l_cal = v.calibration.loss(log_t, torch.tensor([t_star], dtype=log_t.dtype))
        total = l_prm.double() + l_trust + l_cal.double()                         # float64 (D-54)
        return total, {"prm": float(l_prm.detach()), "trust_brier": float(l_trust.detach()), "calibration": float(l_cal.detach()),
                       "resolved_pairs": float(n_pairs), "prm_labelled": float((labels >= 0).sum())}

    def state_dict(self) -> dict[str, Any]:
        return {"monitor": self.monitor, "pairs": (list(self.pairs[0]), list(self.pairs[1]))}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        monitor = state["monitor"]
        if not isinstance(monitor, Monitor):
            raise InvariantViolation("malformed verifier-rlcd state")
        self.monitor = monitor
        p, y = state["pairs"]
        self.pairs = (list(p), list(y))


class RLCDCalibrator:
    """Stage-4 calibrator "rlcd-temperature": the RLCD temperature of every family with enough pairs."""

    name = "rlcd-temperature"

    def __init__(self, model: Any, *, cfg: FeedbackLearningConfig | None = None) -> None:
        self.vc = model.cfg.verifier
        self.cfg = cfg if cfg is not None else FeedbackLearningConfig()

    def fit(self, pairs: Any) -> CalibrationProposal | None:
        """pairs.pairs: family -> (p float64 [n], y float64 [n]); families outside the configured ones are ignored."""
        rc = self.cfg.rlcd
        temps: dict[str, float] = {}
        counts: dict[str, int] = {}
        for fam, (p, y) in dict(pairs.pairs).items():
            if fam not in rc.families:
                continue
            yd = y.to(torch.float64)
            if p.numel() < rc.min_pairs or not (0 < float(yd.sum()) < float(yd.numel())):
                continue
            temps[fam] = rlcd_temperature(p.to(torch.float64), yd, score=rc.score, t_min=self.vc.temperature_min,
                                          t_max=self.vc.temperature_max, grid=rc.grid, tol=rc.tol)
            counts[fam] = int(p.numel())
        if not temps:
            return None
        return CalibrationProposal(temperatures=temps, source="rlcd", n_pairs=counts, t_min=self.vc.temperature_min,
                                   t_max=self.vc.temperature_max)


__all__ = ["RLCDCalibrator", "RLCDStageLearner", "RLHFStageLearner", "RLVRStageLearner", "step_plausibility_labels",
           "training_triggers"]
