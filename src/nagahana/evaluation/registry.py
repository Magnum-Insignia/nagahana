"""Pre-registration of evaluation protocols: hypotheses, metrics, analysis plan, seeds and splits, before any run.

Before a protocol is run, its experiment is registered: the hypotheses, the primary and secondary
metrics, the analysis plan (the complete evaluation configuration: intervals, resampling schemes,
tests, multiple-comparison control, operating points), the models, seeds, splits, datasets and
protocol variants. The registration document is serialised canonically (JSON with sorted keys, no
whitespace, ASCII only, no NaN), hashed with SHA-256 (NIST FIPS 180-4) and time-stamped in UTC, and its
entry is appended to an append-only log in which every entry carries the hash of the previous entry.
Changing a registered document, or editing or reordering the log, breaks a hash and is reported by
`Registry.verify` (the linked time-stamping construction of Haber and Stornetta, Journal of Cryptology
3:99-111, 1991, doi:10.1007/BF00196791).

Pre-registration separates confirmatory from exploratory results (Nosek, Ebersole, DeHaven and Mellor,
"The preregistration revolution", PNAS 115:2600-2606, 2018, doi:10.1073/pnas.1708274114). The thesis
evaluation chapter fixes metrics, thresholds, protocols and tests before the zero-shot split is first
evaluated and evaluates that split once per fixed configuration; the registry makes that commitment
checkable.

Deviations. The scorer refuses a protocol run without a registration of that protocol, then compares
the run with its registration and reports every deviation:

    major   the analysis plan (evaluation configuration) differs from the registered one; a run started
            before the registration time, or its start time (ModelOutputs.config["started_utc"], an
            ISO-8601 time) is not recorded; a registered model or primary metric is missing
    minor   seeds, models, splits, datasets or variants that were not registered, or registered seeds
            that are missing

Hypotheses. Registered hypotheses are tested as one confirmatory family: each is read from the paired
comparison of its two models on its cell, its two-sided p-value is converted to the hypothesised
direction (p / 2 when the estimate lies on the hypothesised side, 1 - p / 2 otherwise; the tests used
here have symmetric two-sided p-values), and the configured multiple-comparison procedure is applied
to the hypotheses alone. A hypothesis is supported when its adjusted p-value is at most the level and
its difference lies beyond the registered margin on the hypothesised side. Its verdict is confirmatory
only when the run has no major deviation; otherwise it is exploratory.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import typing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nagahana.core.errors import InvariantViolation

ZERO_HASH = "0" * 64
DIRECTIONS: tuple[str, ...] = ("greater", "less", "two-sided")


def canonical_json(obj: Any) -> bytes:
    """Canonical serialisation: sorted keys, no whitespace, ASCII, finite numbers only."""
    try:
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except ValueError as exc:
        raise InvariantViolation(f"registration contains a non-finite number: {exc}") from exc
    return text.encode("ascii")


def sha256(obj: Any) -> str:
    """SHA-256 hex digest of the canonical serialisation of obj."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


@dataclass(frozen=True)
class PlannedMetric:
    """A metric fixed in advance: task, metric name and the cell it is read on ("" matches any value)."""

    task: str
    metric: str
    group: str = ""
    novelty: str = ""
    variant: str = ""
    horizon: str = ""


@dataclass(frozen=True)
class Hypothesis:
    """A directional hypothesis about the difference model_a - model_b of one metric on one cell."""

    id: str
    statement: str
    task: str
    metric: str
    model_a: str
    model_b: str
    direction: str
    group: str = ""
    novelty: str = ""
    variant: str = ""
    horizon: str = ""
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise InvariantViolation(f"hypothesis {self.id}: direction must be one of {DIRECTIONS}")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise InvariantViolation(f"hypothesis {self.id}: margin must be finite and non-negative")


@dataclass(frozen=True)
class Registration:
    """The pre-registered plan of one experiment (one protocol)."""

    id: str
    protocol: str
    title: str
    hypotheses: tuple[Hypothesis, ...]
    primary: tuple[PlannedMetric, ...]
    secondary: tuple[PlannedMetric, ...]
    models: tuple[str, ...]
    seeds: tuple[int, ...]
    splits: tuple[str, ...]
    datasets: tuple[str, ...]
    variants: tuple[str, ...] = ()
    data_manifest_sha256: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or any(c in self.id for c in "/\\:*?\"<>| "):
            raise InvariantViolation("registration id must be non-empty and contain no path or space characters")
        if not self.primary:
            raise InvariantViolation("a registration needs at least one primary metric")
        if not self.models or not self.seeds:
            raise InvariantViolation("a registration names its models and seeds")
        if len({h.id for h in self.hypotheses}) != len(self.hypotheses):
            raise InvariantViolation("hypothesis ids must be unique")
        if self.data_manifest_sha256 and (len(self.data_manifest_sha256) != 64
                                          or any(c not in "0123456789abcdef" for c in self.data_manifest_sha256)):
            raise InvariantViolation("data_manifest_sha256 must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class RegistryEntry:
    """One line of the registry log."""

    id: str
    protocol: str
    created_utc: str
    document_sha256: str
    previous_sha256: str
    entry_sha256: str

    def created(self) -> datetime:
        """The registration time as an aware UTC datetime."""
        return parse_utc(self.created_utc)


@dataclass(frozen=True)
class Registered:
    """A registration with its log entry (what the scorer receives)."""

    registration: Registration
    entry: RegistryEntry


@dataclass(frozen=True)
class Deviation:
    """A difference between a run and its registration."""

    severity: str
    kind: str
    detail: str


def to_plain(obj: Any) -> Any:
    """Plain-data form (dicts, lists, numbers, strings) of a registration object."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, tuple | list):
        return [to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    return obj


def _from_plain(tp: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if origin is tuple:
        (inner, _) = typing.get_args(tp)
        if not isinstance(value, list):
            raise InvariantViolation(f"registration: {path} must be a list")
        return tuple(_from_plain(inner, v, f"{path}[{i}]") for i, v in enumerate(value))
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise InvariantViolation(f"registration: {path} must be a mapping")
        hints = typing.get_type_hints(tp)
        names = {f.name for f in dataclasses.fields(tp)}
        unknown = sorted(set(value) - names)
        if unknown:
            raise InvariantViolation(f"registration: unknown keys at {path}: {unknown}")
        kwargs = {k: _from_plain(hints[k], v, f"{path}.{k}") for k, v in value.items()}
        return tp(**kwargs)
    if origin is dict or tp is dict:
        if not isinstance(value, dict):
            raise InvariantViolation(f"registration: {path} must be a mapping")
        return dict(value)
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InvariantViolation(f"registration: {path} must be a number")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise InvariantViolation(f"registration: {path} must be an integer")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise InvariantViolation(f"registration: {path} must be a string")
        return value
    return value


def registration_from_plain(data: dict[str, Any]) -> Registration:
    """Rebuild a Registration from its plain-data form (strict: unknown keys are refused)."""
    reg = _from_plain(Registration, data, "registration")
    assert isinstance(reg, Registration)
    return reg


def parse_utc(text: str) -> datetime:
    """Parse an ISO-8601 time; a time without an offset is refused (it would be ambiguous)."""
    t = datetime.fromisoformat(text)
    if t.tzinfo is None:
        raise InvariantViolation(f"time {text!r} has no UTC offset")
    return t.astimezone(UTC)


class Registry:
    """A directory holding registered documents and the hash-chained log registry.jsonl."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.log = self.root / "registry.jsonl"
        self.docs = self.root / "documents"

    def entries(self) -> list[RegistryEntry]:
        """All log entries in order."""
        if not self.log.exists():
            return []
        out = []
        for line in self.log.read_text(encoding="ascii").splitlines():
            if line.strip():
                out.append(RegistryEntry(**json.loads(line)))
        return out

    def register(self, reg: Registration, *, now: datetime | None = None) -> RegistryEntry:
        """Store the document, time-stamp it and append its entry to the log (refuses a duplicate id)."""
        problems = self.verify()
        if problems:
            raise InvariantViolation("the registry is inconsistent: " + "; ".join(problems))
        existing = self.entries()
        if any(e.id == reg.id for e in existing):
            raise InvariantViolation(f"registration {reg.id!r} already exists; register a new id instead of editing")
        doc = to_plain(reg)
        doc_hash = sha256(doc)
        when = (now if now is not None else datetime.now(UTC)).astimezone(UTC).isoformat(timespec="microseconds")
        previous = existing[-1].entry_sha256 if existing else ZERO_HASH
        body = {"id": reg.id, "protocol": reg.protocol, "created_utc": when, "document_sha256": doc_hash,
                "previous_sha256": previous}
        entry = RegistryEntry(**body, entry_sha256=sha256(body))
        self.docs.mkdir(parents=True, exist_ok=True)
        (self.docs / f"{reg.id}.json").write_bytes(canonical_json(doc))
        with self.log.open("a", encoding="ascii") as fh:
            fh.write(canonical_json(to_plain(entry)).decode("ascii") + "\n")
        return entry

    def verify(self) -> list[str]:
        """Problems of the registry: broken chain, altered or missing documents (empty when consistent)."""
        problems: list[str] = []
        previous = ZERO_HASH
        for i, e in enumerate(self.entries()):
            body = {"id": e.id, "protocol": e.protocol, "created_utc": e.created_utc,
                    "document_sha256": e.document_sha256, "previous_sha256": e.previous_sha256}
            if e.previous_sha256 != previous:
                problems.append(f"entry {i} ({e.id}) does not chain to the previous entry")
            if sha256(body) != e.entry_sha256:
                problems.append(f"entry {i} ({e.id}) was altered")
            doc_path = self.docs / f"{e.id}.json"
            if not doc_path.exists():
                problems.append(f"document of {e.id} is missing")
            elif hashlib.sha256(doc_path.read_bytes()).hexdigest() != e.document_sha256:
                problems.append(f"document of {e.id} was altered after registration")
            previous = e.entry_sha256
        return problems

    def get(self, rid: str) -> Registered:
        """The registration with id rid and its entry (after verifying the registry)."""
        problems = self.verify()
        if problems:
            raise InvariantViolation("the registry is inconsistent: " + "; ".join(problems))
        for e in self.entries():
            if e.id == rid:
                data = json.loads((self.docs / f"{rid}.json").read_bytes())
                return Registered(registration_from_plain(data), e)
        raise InvariantViolation(f"no registration {rid!r} in {self.root}")

    def latest(self, protocol: str) -> Registered | None:
        """The most recent registration of a protocol, or None."""
        found = [e for e in self.entries() if e.protocol.upper() == protocol.upper()]
        return self.get(found[-1].id) if found else None


def _diff_keys(a: Any, b: Any, prefix: str = "", depth: int = 3) -> list[str]:
    # Paths at which two plain-data trees differ (down to `depth` levels).
    if isinstance(a, dict) and isinstance(b, dict) and depth > 0:
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{prefix}{k}")
            else:
                out.extend(_diff_keys(a[k], b[k], f"{prefix}{k}.", depth - 1))
        return out
    return [] if canonical_json(a) == canonical_json(b) else [prefix.rstrip(".") or "(root)"]


def run_deviations(reg: Registered, *, config: dict[str, Any], models: dict[str, list[int]],
                   splits: set[str], datasets: set[str], variants: set[str],
                   started: list[tuple[str, int, str | None]]) -> list[Deviation]:
    """Deviations of a run from its registration (module docstring).

    models: run key -> seeds; started: (run key, seed, started_utc or None) per bundle.
    """
    r, e = reg.registration, reg.entry
    out: list[Deviation] = []
    if r.config:
        diff = _diff_keys(r.config, config)
        if diff:
            out.append(Deviation("major", "analysis_plan", "evaluation configuration differs from the registered one at: "
                                 + ", ".join(diff[:20]) + (" ..." if len(diff) > 20 else "")))
    else:
        out.append(Deviation("major", "analysis_plan", "the registration carries no evaluation configuration"))
    created = e.created()
    for key, seed, when in started:
        if when is None:
            out.append(Deviation("major", "timing", f"{key} seed {seed}: run start time not recorded "
                                 "(config['started_utc']); precedence of the registration cannot be verified"))
            continue
        try:
            t = parse_utc(when)
        except (ValueError, InvariantViolation) as exc:
            out.append(Deviation("major", "timing", f"{key} seed {seed}: unreadable start time ({exc})"))
            continue
        if t < created:
            out.append(Deviation("major", "timing", f"{key} seed {seed}: run started {when}, before the registration "
                                 f"time {e.created_utc}"))
    for m in r.models:
        if m not in models:
            out.append(Deviation("major", "models", f"registered model {m} has no outputs"))
    for m, seeds in models.items():
        if m not in r.models:
            out.append(Deviation("minor", "models", f"model {m} was not registered"))
            continue
        extra = sorted(set(seeds) - set(r.seeds))
        missing = sorted(set(r.seeds) - set(seeds))
        if extra:
            out.append(Deviation("minor", "seeds", f"{m}: seeds {extra} were not registered"))
        if missing:
            out.append(Deviation("minor", "seeds", f"{m}: registered seeds {missing} are missing"))
    for name, present, registered in (("splits", splits, set(r.splits)), ("datasets", datasets, set(r.datasets)),
                                      ("variants", variants, set(r.variants))):
        if registered:
            extra = sorted(present - registered)
            if extra:
                out.append(Deviation("minor", name, f"{name} {extra} were not registered"))
    return out


def directional_p(p_two_sided: float, estimate: float, direction: str) -> float:
    """One-sided p-value in the hypothesised direction from a symmetric two-sided p-value."""
    if not math.isfinite(p_two_sided) or not math.isfinite(estimate):
        return math.nan
    if direction == "two-sided":
        return p_two_sided
    toward = estimate > 0 if direction == "greater" else estimate < 0
    return p_two_sided / 2.0 if toward else 1.0 - p_two_sided / 2.0
