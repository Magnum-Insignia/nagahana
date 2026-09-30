"""Forecast-specific evaluation: skill, lead time, and time-to-event concordance (ARCH §6.2, #26).

Skill score: does the forecaster beat a naive reference?
--------------------------------------------------------
    SS = 1 − S_model / S_reference
S is a mean proper score (Brier or CRPS, lower is better). The reference is persistence ("like
now") or climatology (the base rate). SS > 0 means the model adds value; SS ≤ 0 means it does not.
This is the honest bar from forecast verification [ARCH #26.6].

Lead time: how early is the warning?
------------------------------------
For an attack whose stage is completed at time t_c, the lead time is t_c − t_alert, where t_alert is
the first time P_inf crosses the alert threshold before t_c. It is undefined (NaN) if the alert
never fired in time. Report it at a fixed operating point (e.g. a fixed FPR); lead time bought with
more false alarms is not progress.

Concordance (Harrell's C-index): does the model rank *when* correctly?
----------------------------------------------------------------------
For time-to-event outputs (hazard curves) with right-censoring (attacks that never completed):

    C = P( risk_i > risk_j  |  T_i < T_j and event_i observed )

This counts comparable pairs whose earlier time is an observed event. Ties in risk count as ½.
C = 0.5 is random and C = 1 is perfect ordering. This is the reference O(n²) implementation.
"""

from __future__ import annotations

import math

import torch


def skill_score(model_score: float, reference_score: float) -> float:
    """1 − S_model / S_reference (lower-is-better scores). NaN if the reference score is 0."""
    if reference_score == 0:
        return math.nan
    return 1.0 - model_score / reference_score


def lead_time(times: torch.Tensor, p_inf: torch.Tensor, *, threshold: float, completion_time: float) -> float:
    """t_c − (first time p_inf ≥ threshold before t_c); NaN if no timely alert."""
    before = times < completion_time
    hits = torch.nonzero(before & (p_inf >= threshold)).flatten()
    if hits.numel() == 0:
        return math.nan
    return float(completion_time - times[hits[0]])


def concordance_index(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> float:
    """Harrell's C-index for right-censored data (higher risk should mean earlier event)."""
    n = risk.numel()
    num, den = 0.0, 0
    r, t, e = risk.tolist(), time.tolist(), event.bool().tolist()
    for i in range(n):
        if not e[i]:
            continue
        for j in range(n):
            if t[i] < t[j]:
                den += 1
                if r[i] > r[j]:
                    num += 1.0
                elif r[i] == r[j]:
                    num += 0.5
    return num / den if den else math.nan
