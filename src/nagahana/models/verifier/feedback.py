"""Human feedback the Verifier learns from: typed, validated, content-addressed events (D-17, D-21, D-65).

Human feedback is trusted supplied truth (D-17, [A-21]); it is the only signal the Verifier's learning
reads besides confirmed outcomes. Each event carries its provenance and refers to a model record (a
forecast, an alert or an advisory) by id; the model record itself (the analysis tensors and the
forecast) is resolved at fit time from a situation source (`situations.py`, AS-832), while the content
the analyst judged (the routes of a forecast, the steps of an advisory) is stored in the event, so the
ledger alone says exactly what was preferred over what.

Event kinds

    preference  pairwise preference between two forecasts, two route sets or two advisories of the
                same situation: first, second or tie, with the analyst's confidence (AS-833, AS-834)
    alert       accept, reject or edit of an alert (AS-846)
    stage       correction of a stage: the entity's current stage (step 0) or forecast step k >= 1
    outcome     the confirmed outcome after the horizon: first infiltration step (0 = none), observed
                steps (right-censoring), responded-to flag with the known response propensity, and
                the stages confirmed per step (AS-839, AS-840)

Provenance (every event): the analyst id, the event time (epoch seconds), the id of the record it
refers to, the id of the HumanCommand under which it was admitted into the ledger (stamped at ingest,
AS-831), an optional SHA-256 digest of the situation the analyst saw, and a free-text note.

Preference labels. A preference becomes the soft label s = P(first preferred): s = c when the first item
is preferred with confidence c in (0.5, 1], s = 1 - c when the second is, s = 1/2 for a tie. The
Bradley-Terry likelihood and the DPO loss are cross-entropies against s, so a certain preference (c = 1)
is the usual hard label and a tie pulls the two implicit rewards together.

Identity. `event_id(e)` is the SHA-256 of the canonical JSON record of the event (every field, the
command id included), so an event is addressed by its content and a resubmission at another time is a
new observation with a new id.

Invariants (tested): every constructor rejects malformed content (empty analyst, non-finite time,
out-of-range steps, an event step after the last observed step, a propensity outside [0, 1), unknown
stage names, preferences between identical items); `event_from_record(event_to_record(e)) == e`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from nagahana.core.errors import InvariantViolation
from nagahana.models.verifier.canonical import digest_of
from nagahana.models.vocab import N_STAGES, STAGE_CODE, STAGES

FEEDBACK_KINDS: tuple[str, ...] = ("preference", "alert", "stage", "outcome")
ITEM_KINDS: tuple[str, ...] = ("forecast", "route_set", "advisory")
PREFERENCES: tuple[str, ...] = ("first", "second", "tie")
VERDICTS: tuple[str, ...] = ("accept", "reject", "edit")
OUTCOME_SOURCES: tuple[str, ...] = ("forensics", "analyst")
STAGE_NAMES: tuple[str, ...] = tuple(name for name, _ in STAGES)
#: Target code of a route step without a target (the Forecaster's imagination writes -1 there).
ROUTE_NO_TARGET = -1


def _is_hex64(text: str) -> bool:
    return len(text) == 64 and all(c in "0123456789abcdef" for c in text)


def _finite(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise InvariantViolation(f"{name} must be finite, got {value!r}")
    return v


@dataclass(frozen=True)
class Provenance:
    """Who said what about which record, when, and under which command (module docstring)."""

    analyst: str
    time: float
    refers_to: str
    command_id: str = ""
    situation_digest: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.analyst.strip():
            raise InvariantViolation("feedback needs the analyst who gave it")
        _finite("feedback time", self.time)
        if not self.refers_to.strip():
            raise InvariantViolation("feedback must name the forecast, alert or advisory it refers to")
        if self.command_id and not _is_hex64(self.command_id):
            raise InvariantViolation("command_id must be a lowercase SHA-256 hex digest (gate.command_id)")
        if self.situation_digest and not _is_hex64(self.situation_digest):
            raise InvariantViolation("situation_digest must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class RouteSpec:
    """One imagined route as the analyst saw it: K steps of (technique slot, target entity).

    targets use the Forecaster's imagination convention: a window entity index, or -1 for no target.
    count is the number of the forecast's draws merged into this route (D-46); cumulative is the route's
    own infiltration curve F_n(k) and stage_codes its stage per step, both as shown (optional).
    """

    techniques: tuple[int, ...]
    targets: tuple[int, ...]
    count: int = 1
    cumulative: tuple[float, ...] = ()
    stage_codes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        k = len(self.techniques)
        if k < 1 or len(self.targets) != k:
            raise InvariantViolation("a route needs K >= 1 techniques and K targets")
        if any(int(t) < 0 for t in self.techniques):
            raise InvariantViolation("technique slots of a route must be >= 0")
        if any(int(v) < ROUTE_NO_TARGET for v in self.targets):
            raise InvariantViolation("route targets are entity indices >= 0 or -1 (no target)")
        if self.count < 1:
            raise InvariantViolation("a route's merge count must be >= 1")
        if self.cumulative:
            if len(self.cumulative) != k:
                raise InvariantViolation("a route's cumulative curve needs one value per step")
            prev = 0.0
            for x in self.cumulative:
                v = _finite("cumulative probability", x)
                if not 0.0 <= v <= 1.0 or v + 1e-12 < prev:
                    raise InvariantViolation("a route's cumulative curve must lie in [0, 1] and never decrease")
                prev = v
        if self.stage_codes and (len(self.stage_codes) != k or any(not 0 <= int(s) < N_STAGES for s in self.stage_codes)):
            raise InvariantViolation(f"stage codes need one code in [0, {N_STAGES}) per step")

    @property
    def horizon(self) -> int:
        """K, the number of imagined steps."""
        return len(self.techniques)


@dataclass(frozen=True)
class PlanSpec:
    """One counter-measure sequence as the analyst saw it, with the evidence it was ranked by (D-33)."""

    steps: tuple[tuple[int, int], ...]
    delta_p_inf: float
    delta_p_inf_cvar: float
    delta_p_inf_worst: float
    disruption_cost: float
    information_value: float
    feasible: bool

    def __post_init__(self) -> None:
        if not self.steps:
            raise InvariantViolation("an advisory needs at least one step")
        for step in self.steps:
            if len(step) != 2 or int(step[0]) < 0 or int(step[1]) < 0:
                raise InvariantViolation("an advisory step is (D3FEND slot >= 0, target entity >= 0)")
        for name in ("delta_p_inf", "delta_p_inf_cvar", "delta_p_inf_worst", "disruption_cost", "information_value"):
            _finite(f"advisory {name}", getattr(self, name))
        if self.disruption_cost < 0 or self.information_value < 0:
            raise InvariantViolation("disruption cost and information value are non-negative")


@dataclass(frozen=True)
class PreferenceItem:
    """One side of a preference: a forecast or route set (routes) or an advisory (plan)."""

    item_id: str
    routes: tuple[RouteSpec, ...] = ()
    plan: PlanSpec | None = None
    p_inf: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.item_id.strip():
            raise InvariantViolation("a preference item needs its id")
        if self.p_inf:
            prev = 0.0
            for x in self.p_inf:
                v = _finite("P_inf", x)
                if not 0.0 <= v <= 1.0 or v + 1e-12 < prev:
                    raise InvariantViolation("an item's P_inf curve must lie in [0, 1] and never decrease")
                prev = v


@dataclass(frozen=True)
class PreferenceFeedback:
    """Pairwise preference between two items of one kind for the same situation (provenance.refers_to)."""

    provenance: Provenance
    item_kind: str
    first: PreferenceItem
    second: PreferenceItem
    preferred: str
    confidence: float = 1.0
    kind: ClassVar[str] = "preference"

    def __post_init__(self) -> None:
        if self.item_kind not in ITEM_KINDS:
            raise InvariantViolation(f"item_kind must be one of {list(ITEM_KINDS)}, got {self.item_kind!r}")
        if self.preferred not in PREFERENCES:
            raise InvariantViolation(f"preferred must be one of {list(PREFERENCES)}, got {self.preferred!r}")
        c = _finite("confidence", self.confidence)
        if not 0.5 < c <= 1.0:
            raise InvariantViolation("confidence must lie in (0.5, 1]; a tie is preferred='tie'")
        if self.first.item_id == self.second.item_id:
            raise InvariantViolation("a preference compares two different items")
        for item in (self.first, self.second):
            if self.item_kind == "advisory":
                if item.plan is None or item.routes:
                    raise InvariantViolation("an advisory item carries a plan and no routes")
            else:
                if not item.routes or item.plan is not None:
                    raise InvariantViolation(f"a {self.item_kind} item carries routes and no plan")
                ks = {r.horizon for r in item.routes}
                if len(ks) != 1:
                    raise InvariantViolation("all routes of an item share one horizon K")
                if item.p_inf and len(item.p_inf) != next(iter(ks)):
                    raise InvariantViolation("an item's P_inf curve needs one value per step")
                if self.item_kind == "forecast" and not item.p_inf:
                    raise InvariantViolation("a forecast item carries the P_inf curve that was shown")
        if self.item_kind == "advisory":
            assert self.first.plan is not None and self.second.plan is not None
            if self.first.plan.steps == self.second.plan.steps:
                raise InvariantViolation("a preference between identical advisories carries no information")
        elif (sorted((r.techniques, r.targets, r.count) for r in self.first.routes)
              == sorted((r.techniques, r.targets, r.count) for r in self.second.routes)):
            raise InvariantViolation("a preference between identical route sets carries no information")

    @property
    def target(self) -> str:
        """The policy this preference trains: "forecaster" (forecasts, route sets) or "advisor" (advisories)."""
        return "advisor" if self.item_kind == "advisory" else "forecaster"

    @property
    def label(self) -> float:
        """Soft label s = P(first preferred) (module docstring)."""
        if self.preferred == "tie":
            return 0.5
        return float(self.confidence) if self.preferred == "first" else 1.0 - float(self.confidence)


@dataclass(frozen=True)
class AlertFeedback:
    """Accept, reject or edit of an alert (AS-846).

    entity is the window entity index the alert named. An edit corrects the entity (the alert named the
    wrong machine) and/or the stage; at least one correction is required.
    """

    provenance: Provenance
    verdict: str
    entity: int | None = None
    corrected_entity: int | None = None
    corrected_stage: str | None = None
    kind: ClassVar[str] = "alert"

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise InvariantViolation(f"verdict must be one of {list(VERDICTS)}, got {self.verdict!r}")
        if self.verdict != "edit" and (self.corrected_entity is not None or self.corrected_stage is not None):
            raise InvariantViolation("only an edit carries corrections")
        if self.verdict == "edit" and self.corrected_entity is None and self.corrected_stage is None:
            raise InvariantViolation("an edit corrects the entity, the stage, or both")
        for name in ("entity", "corrected_entity"):
            v = getattr(self, name)
            if v is not None and int(v) < 0:
                raise InvariantViolation(f"{name} is a window entity index >= 0")
        if self.corrected_entity is not None and (self.entity is None or self.corrected_entity == self.entity):
            raise InvariantViolation("an entity correction names the alert's entity and a different corrected one")
        if self.corrected_stage is not None and self.corrected_stage not in STAGE_CODE:
            raise InvariantViolation(f"unknown stage {self.corrected_stage!r}; stages: {list(STAGE_NAMES)}")

    @property
    def decision_correct(self) -> bool:
        """Was the alert decision right? accept: yes; reject: no; edit: yes unless it named the wrong entity."""
        if self.verdict == "accept":
            return True
        if self.verdict == "reject":
            return False
        return self.corrected_entity is None


@dataclass(frozen=True)
class StageCorrection:
    """The stage that really held: of an entity now (step 0, needs the entity) or at forecast step k >= 1."""

    provenance: Provenance
    step: int
    stage: str
    entity: int | None = None
    kind: ClassVar[str] = "stage"

    def __post_init__(self) -> None:
        if self.step < 0:
            raise InvariantViolation("a stage correction's step is 0 (now) or a forecast step k >= 1")
        if self.step == 0 and self.entity is None:
            raise InvariantViolation("a correction of the current stage names its entity")
        if self.entity is not None and int(self.entity) < 0:
            raise InvariantViolation("entity is a window entity index >= 0")
        if self.stage not in STAGE_CODE:
            raise InvariantViolation(f"unknown stage {self.stage!r}; stages: {list(STAGE_NAMES)}")

    @property
    def stage_code(self) -> int:
        """The class code of the corrected stage (models/vocab.py)."""
        return STAGE_CODE[self.stage]


@dataclass(frozen=True)
class OutcomeConfirmation:
    """What happened after the horizon of a forecast (the outcome contract of evaluation/predictions.py).

    event_step: first step with infiltration (1 ... K), 0 when none occurred inside the observed steps.
    observed_steps: how many of the K steps were observed; event_step = 0 with observed_steps < K is
    right-censored after observed_steps steps. responded_to: the SOC acted on the forecast, so the
    outcome is counterfactual and is not scored as right or wrong [Q-38]; response_propensity is the
    known probability that the response policy acts on such a forecast (AS-840). stages: (step, stage)
    pairs confirmed by forensics or the analyst.
    """

    provenance: Provenance
    horizon_k: int
    event_step: int
    observed_steps: int
    responded_to: bool = False
    response_propensity: float | None = None
    source: str = "analyst"
    stages: tuple[tuple[int, str], ...] = ()
    kind: ClassVar[str] = "outcome"

    def __post_init__(self) -> None:
        k = self.horizon_k
        if k < 1:
            raise InvariantViolation("horizon_k must be >= 1")
        if not 0 <= self.event_step <= k or not 0 <= self.observed_steps <= k:
            raise InvariantViolation("event_step and observed_steps must lie in 0 ... K")
        if self.event_step > 0 and self.event_step > self.observed_steps:
            raise InvariantViolation("an event cannot fall after the last observed step")
        if self.response_propensity is not None:
            rho = _finite("response_propensity", self.response_propensity)
            if not 0.0 <= rho < 1.0:
                raise InvariantViolation("response_propensity must lie in [0, 1)")
        if self.source not in OUTCOME_SOURCES:
            raise InvariantViolation(f"source must be one of {list(OUTCOME_SOURCES)}, got {self.source!r}")
        steps = [int(s) for s, _ in self.stages]
        if len(set(steps)) != len(steps) or any(not 1 <= s <= k for s in steps):
            raise InvariantViolation("confirmed stages name distinct forecast steps in 1 ... K")
        for _, name in self.stages:
            if name not in STAGE_CODE:
                raise InvariantViolation(f"unknown stage {name!r}; stages: {list(STAGE_NAMES)}")

    @property
    def occurred(self) -> bool:
        """Infiltration happened within the horizon."""
        return self.event_step > 0

    @property
    def known_at_horizon(self) -> bool:
        """The outcome of 'infiltration within K steps' is known (an event, or all K steps observed)."""
        return self.event_step > 0 or self.observed_steps >= self.horizon_k

    @property
    def stage_codes(self) -> dict[int, int]:
        """Confirmed stage code per forecast step."""
        return {int(s): STAGE_CODE[name] for s, name in self.stages}


FeedbackEvent = PreferenceFeedback | AlertFeedback | StageCorrection | OutcomeConfirmation
_KIND_OF: dict[str, type] = {"preference": PreferenceFeedback, "alert": AlertFeedback, "stage": StageCorrection,
                             "outcome": OutcomeConfirmation}


def _provenance_record(p: Provenance) -> dict[str, Any]:
    return {"analyst": p.analyst, "time": float(p.time), "refers_to": p.refers_to, "command_id": p.command_id,
            "situation_digest": p.situation_digest, "note": p.note}


def _route_record(r: RouteSpec) -> dict[str, Any]:
    return {"techniques": [int(t) for t in r.techniques], "targets": [int(v) for v in r.targets], "count": int(r.count),
            "cumulative": [float(x) for x in r.cumulative], "stage_codes": [int(s) for s in r.stage_codes]}


def _plan_record(p: PlanSpec) -> dict[str, Any]:
    return {"steps": [[int(a), int(v)] for a, v in p.steps], "delta_p_inf": float(p.delta_p_inf),
            "delta_p_inf_cvar": float(p.delta_p_inf_cvar), "delta_p_inf_worst": float(p.delta_p_inf_worst),
            "disruption_cost": float(p.disruption_cost), "information_value": float(p.information_value),
            "feasible": bool(p.feasible)}


def _item_record(i: PreferenceItem) -> dict[str, Any]:
    return {"item_id": i.item_id, "routes": [_route_record(r) for r in i.routes],
            "plan": None if i.plan is None else _plan_record(i.plan), "p_inf": [float(x) for x in i.p_inf]}


def event_to_record(event: FeedbackEvent) -> dict[str, Any]:
    """The JSON record of an event (the ledger payload body); `event_from_record` inverts it exactly."""
    body: dict[str, Any] = {"kind": event.kind, "provenance": _provenance_record(event.provenance)}
    if isinstance(event, PreferenceFeedback):
        body |= {"item_kind": event.item_kind, "first": _item_record(event.first), "second": _item_record(event.second),
                 "preferred": event.preferred, "confidence": float(event.confidence)}
    elif isinstance(event, AlertFeedback):
        body |= {"verdict": event.verdict, "entity": event.entity, "corrected_entity": event.corrected_entity,
                 "corrected_stage": event.corrected_stage}
    elif isinstance(event, StageCorrection):
        body |= {"step": int(event.step), "stage": event.stage, "entity": event.entity}
    elif isinstance(event, OutcomeConfirmation):
        body |= {"horizon_k": int(event.horizon_k), "event_step": int(event.event_step),
                 "observed_steps": int(event.observed_steps), "responded_to": bool(event.responded_to),
                 "response_propensity": None if event.response_propensity is None else float(event.response_propensity),
                 "source": event.source, "stages": [[int(s), name] for s, name in event.stages]}
    else:
        raise InvariantViolation(f"not a feedback event: {type(event).__name__}")
    return body


def _take(record: Mapping[str, Any], keys: Sequence[str], where: str) -> dict[str, Any]:
    # Strict field set: an unknown or missing key is an error, never silently dropped or defaulted.
    got = set(record)
    want = set(keys)
    if got != want:
        raise InvariantViolation(f"{where}: fields {sorted(got)} differ from {sorted(want)}")
    return dict(record)


def _provenance_from(r: Mapping[str, Any]) -> Provenance:
    d = _take(r, ("analyst", "time", "refers_to", "command_id", "situation_digest", "note"), "provenance")
    return Provenance(analyst=str(d["analyst"]), time=float(d["time"]), refers_to=str(d["refers_to"]),
                      command_id=str(d["command_id"]), situation_digest=str(d["situation_digest"]), note=str(d["note"]))


def _route_from(r: Mapping[str, Any]) -> RouteSpec:
    d = _take(r, ("techniques", "targets", "count", "cumulative", "stage_codes"), "route")
    return RouteSpec(techniques=tuple(int(t) for t in d["techniques"]), targets=tuple(int(v) for v in d["targets"]),
                     count=int(d["count"]), cumulative=tuple(float(x) for x in d["cumulative"]),
                     stage_codes=tuple(int(s) for s in d["stage_codes"]))


def _plan_from(r: Mapping[str, Any]) -> PlanSpec:
    d = _take(r, ("steps", "delta_p_inf", "delta_p_inf_cvar", "delta_p_inf_worst", "disruption_cost",
                  "information_value", "feasible"), "plan")
    return PlanSpec(steps=tuple((int(a), int(v)) for a, v in d["steps"]), delta_p_inf=float(d["delta_p_inf"]),
                    delta_p_inf_cvar=float(d["delta_p_inf_cvar"]), delta_p_inf_worst=float(d["delta_p_inf_worst"]),
                    disruption_cost=float(d["disruption_cost"]), information_value=float(d["information_value"]),
                    feasible=bool(d["feasible"]))


def _item_from(r: Mapping[str, Any]) -> PreferenceItem:
    d = _take(r, ("item_id", "routes", "plan", "p_inf"), "item")
    return PreferenceItem(item_id=str(d["item_id"]), routes=tuple(_route_from(x) for x in d["routes"]),
                          plan=None if d["plan"] is None else _plan_from(d["plan"]),
                          p_inf=tuple(float(x) for x in d["p_inf"]))


def _opt_int(v: Any) -> int | None:
    return None if v is None else int(v)


def event_from_record(record: Mapping[str, Any]) -> FeedbackEvent:
    """Rebuild an event from its record, validating every field (unknown or missing keys raise)."""
    kind = record.get("kind")
    if kind not in _KIND_OF:
        raise InvariantViolation(f"unknown feedback kind {kind!r}; kinds: {list(FEEDBACK_KINDS)}")
    prov = _provenance_from(record["provenance"]) if "provenance" in record else None
    if prov is None:
        raise InvariantViolation("a feedback record needs its provenance")
    if kind == "preference":
        d = _take(record, ("kind", "provenance", "item_kind", "first", "second", "preferred", "confidence"), "preference")
        return PreferenceFeedback(provenance=prov, item_kind=str(d["item_kind"]), first=_item_from(d["first"]),
                                  second=_item_from(d["second"]), preferred=str(d["preferred"]),
                                  confidence=float(d["confidence"]))
    if kind == "alert":
        d = _take(record, ("kind", "provenance", "verdict", "entity", "corrected_entity", "corrected_stage"), "alert")
        return AlertFeedback(provenance=prov, verdict=str(d["verdict"]), entity=_opt_int(d["entity"]),
                             corrected_entity=_opt_int(d["corrected_entity"]),
                             corrected_stage=None if d["corrected_stage"] is None else str(d["corrected_stage"]))
    if kind == "stage":
        d = _take(record, ("kind", "provenance", "step", "stage", "entity"), "stage")
        return StageCorrection(provenance=prov, step=int(d["step"]), stage=str(d["stage"]), entity=_opt_int(d["entity"]))
    d = _take(record, ("kind", "provenance", "horizon_k", "event_step", "observed_steps", "responded_to",
                       "response_propensity", "source", "stages"), "outcome")
    return OutcomeConfirmation(provenance=prov, horizon_k=int(d["horizon_k"]), event_step=int(d["event_step"]),
                               observed_steps=int(d["observed_steps"]), responded_to=bool(d["responded_to"]),
                               response_propensity=None if d["response_propensity"] is None else float(d["response_propensity"]),
                               source=str(d["source"]), stages=tuple((int(s), str(n)) for s, n in d["stages"]))


def event_id(event: FeedbackEvent) -> str:
    """SHA-256 of the event's canonical record (module docstring, Identity)."""
    return digest_of(event_to_record(event))


def stamp(event: FeedbackEvent, command_id: str) -> FeedbackEvent:
    """The event with the id of the HumanCommand that admits it (an event stamped by another command raises)."""
    if not _is_hex64(command_id):
        raise InvariantViolation("command_id must be a lowercase SHA-256 hex digest")
    have = event.provenance.command_id
    if have and have != command_id:
        raise InvariantViolation(f"event already admitted under command {have[:12]}..., not {command_id[:12]}...")
    return replace(event, provenance=replace(event.provenance, command_id=command_id))


__all__ = [
    "FEEDBACK_KINDS", "ITEM_KINDS", "OUTCOME_SOURCES", "PREFERENCES", "ROUTE_NO_TARGET", "STAGE_NAMES", "VERDICTS",
    "AlertFeedback", "FeedbackEvent", "OutcomeConfirmation", "PlanSpec", "PreferenceFeedback", "PreferenceItem",
    "Provenance", "RouteSpec", "StageCorrection", "event_from_record", "event_id", "event_to_record", "stamp",
]
