"""The Verifier's ledger: one hash-chained, append-only, persisted record of feedback and audit steps (AS-830).

Why one chain. Feedback (what analysts said) and audit records (which commands were shown, which
candidate was fitted from which feedback, how it was evaluated, who promoted it, what a rollback
restored) live in one chain, so every audit record commits to the exact feedback prefix before it: a fit
record names the ledger head it was computed from, and any later edit, deletion or reordering of that
feedback breaks every subsequent hash.

The chain. Each record's payload is the canonical JSON text (canonical.py) of

    {"v": 1, "kind": <record kind>, "time": <epoch seconds>, "body": <record body>}

and is committed by `memory.eventlog.HashChainLog` (P-02's primitive), whose entry hash is

    h_i = SHA-256( h_(i-1) || SHA-256(payload_i) ),   h_(-1) = 0^256   (as hex text)

D-65 makes this ledger mandatory for the Verifier's own records; the primitive is enabled for these
records only, which leaves the held question of the Environment's persistence (D-15, P-02) untouched.

Persistence. The ledger file holds one JSON line per record: index, previous hash, payload hash, entry
hash and the payload text. Appends are written as one line with flush and, by default, fsync (durable on
return). Opening a file replays every payload through a fresh chain and compares every stored hash with
the recomputed one, so an edited, deleted, reordered or torn line raises `InvariantViolation` naming the
line. A truncated tail is a valid shorter chain; it is detected by comparing the head with an anchor
kept elsewhere (`expected_head`; candidate files record the head they were fitted from). Before each
append the file size is compared with the size this ledger last wrote, so a second writer is refused
(one writer per ledger file, AS-830).

Record kinds

    command                  a HumanCommand shown to the ledger (its fields and id, gate.command_id)
    feedback                 one admitted feedback event (feedback.event_to_record) with its event id
    audit.ingest             a batch of feedback admitted under an "ingest-feedback" command
    audit.fit                a candidate fitted under a "fit-feedback-update" command
    audit.evaluate           a held-out evaluation of a candidate
    audit.promote            a promotion of weight deltas under an "update-weights" command
    audit.apply-calibration  temperatures applied under an "apply-calibration" command
    audit.rollback           a rollback under a "rollback-update" command
    audit.refused            an attempted change refused by the gate (no or wrong command, reuse, stale state)

Single use (AS-831). A command authorises at most one state-changing record (ingest, fit, promote,
apply-calibration, rollback); `assert_unused` raises `HumanCommandRequired` for a command that already
authorised one.

Invariants (tested): `verify` passes on an untouched ledger and raises on any in-memory or on-disk
tampering; reopening a file reproduces the same records and head; duplicate feedback events (same id)
are refused.
"""

from __future__ import annotations

import json
import os
import time as _time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from nagahana.core.errors import HumanCommandRequired, InvariantViolation
from nagahana.memory.eventlog import HashChainLog
from nagahana.models.verifier.canonical import canonical_json, from_canonical, sha256_hex
from nagahana.models.verifier.feedback import FeedbackEvent, event_from_record, event_id, event_to_record
from nagahana.models.verifier.gate import command_id, command_record
from nagahana.roles.contracts import HumanCommand

#: The proposal whose hash-chain primitive the ledger reuses (memory/eventlog.py); enabled for the ledger only.
EVENT_LOG_PROPOSAL = "P-02"
LEDGER_VERSION = 1
RECORD_KINDS: tuple[str, ...] = ("command", "feedback", "audit.ingest", "audit.fit", "audit.evaluate", "audit.promote",
                                 "audit.apply-calibration", "audit.rollback", "audit.refused")
#: Audit kinds that consume the authorising command (single use).
STATE_CHANGING: frozenset[str] = frozenset({"audit.ingest", "audit.fit", "audit.promote", "audit.apply-calibration",
                                            "audit.rollback"})
_LINE_KEYS = ("entry_hash", "index", "payload", "payload_hash", "prev_hash")


@dataclass(frozen=True)
class LedgerRecord:
    """One committed record: its chain position and hashes, and the decoded payload."""

    index: int
    kind: str
    time: float
    body: Mapping[str, Any] = field(repr=False)
    prev_hash: str = ""
    payload_hash: str = ""
    entry_hash: str = ""


def _payload_text(kind: str, body: Mapping[str, Any], when: float) -> str:
    return canonical_json({"v": LEDGER_VERSION, "kind": kind, "time": float(when), "body": dict(body)})


def _freeze(obj: Any) -> Any:
    # Read-only views of decoded bodies, so a record's body cannot be edited in place by a caller.
    if isinstance(obj, dict):
        return MappingProxyType({k: _freeze(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return tuple(_freeze(v) for v in obj)
    return obj


def _decode_payload(text: str, where: str) -> tuple[str, float, dict[str, Any]]:
    try:
        obj = from_canonical(text)
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvariantViolation(f"{where}: payload is not canonical JSON ({exc})") from exc
    if not isinstance(obj, dict) or set(obj) != {"v", "kind", "time", "body"}:
        raise InvariantViolation(f"{where}: payload must hold exactly v, kind, time and body")
    if obj["v"] != LEDGER_VERSION:
        raise InvariantViolation(f"{where}: unknown ledger version {obj['v']!r}")
    if obj["kind"] not in RECORD_KINDS:
        raise InvariantViolation(f"{where}: unknown record kind {obj['kind']!r}")
    if not isinstance(obj["body"], dict):
        raise InvariantViolation(f"{where}: body must be a mapping")
    if canonical_json(obj) != text:
        raise InvariantViolation(f"{where}: payload is not in canonical form")
    return str(obj["kind"]), float(obj["time"]), obj["body"]


class FeedbackLedger:
    """Hash-chained feedback and audit ledger, in memory or persisted to a JSON Lines file (module docstring).

    Parameters
    ----------
    path:
        The ledger file. An existing file is replayed and verified; a missing one is created on the first
        append. None keeps the ledger in memory (tests, in-loop training).
    fsync:
        Force every append to stable storage before returning.
    expected_head:
        When given, the replayed head must equal it (detects a truncated tail against an external anchor).
    """

    def __init__(self, path: str | Path | None = None, *, fsync: bool = True, expected_head: str | None = None) -> None:
        self._log = HashChainLog(enabled_proposals=(EVENT_LOG_PROPOSAL,))
        self._records: list[LedgerRecord] = []
        self._commands: dict[str, LedgerRecord] = {}
        self._event_index: dict[str, int] = {}
        self._used: dict[str, LedgerRecord] = {}
        self._path = None if path is None else Path(path)
        self._fsync = bool(fsync)
        self._size = 0
        if self._path is not None and self._path.exists():
            self._replay(self._path)
        if expected_head is not None and self.head != expected_head:
            raise InvariantViolation(f"ledger head {self.head[:16]}... differs from the anchor {expected_head[:16]}... "
                                     "(records missing or replaced)")

    # ---- reading and verification
    def _replay(self, path: Path) -> None:
        # Rebuild the chain from the payloads and compare every stored hash with the recomputed one.
        raw = path.read_bytes()
        self._size = len(raw)
        if not raw:
            return
        lines = raw.split(b"\n")
        if lines[-1] != b"":
            raise InvariantViolation(f"{path}: the last line is incomplete (torn write); the ledger is not opened")
        for n, line in enumerate(lines[:-1], start=1):
            where = f"{path}:{n}"
            try:
                obj = json.loads(line.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InvariantViolation(f"{where}: not a ledger line ({exc})") from exc
            if not isinstance(obj, dict) or sorted(obj) != list(_LINE_KEYS):
                raise InvariantViolation(f"{where}: a ledger line holds exactly {list(_LINE_KEYS)}")
            payload = obj["payload"]
            if not isinstance(payload, str):
                raise InvariantViolation(f"{where}: payload must be text")
            entry = self._log.append(payload.encode("utf-8"))
            if (obj["index"], obj["prev_hash"], obj["payload_hash"], obj["entry_hash"]) != (
                    entry.index, entry.prev_hash, entry.payload_hash, entry.entry_hash):
                raise InvariantViolation(f"{where}: stored hashes do not match the chain recomputed from the payloads "
                                         "(edited, deleted or reordered record)")
            kind, when, body = _decode_payload(payload, where)
            self._index(LedgerRecord(entry.index, kind, when, _freeze(body), entry.prev_hash, entry.payload_hash,
                                     entry.entry_hash))

    def _index(self, rec: LedgerRecord) -> None:
        # Secondary indices: commands by id, feedback by event id, consumed commands.
        if rec.kind == "command":
            self._commands[str(rec.body["id"])] = rec
        elif rec.kind == "feedback":
            eid = str(rec.body["event_id"])
            if eid in self._event_index:
                raise InvariantViolation(f"record {rec.index}: duplicate feedback event {eid[:12]}...")
            self._event_index[eid] = rec.index
        if rec.kind in STATE_CHANGING:
            cid = str(rec.body.get("command_id", ""))
            if cid in self._used:
                raise InvariantViolation(f"record {rec.index}: command {cid[:12]}... authorised two changes")
            self._used[cid] = rec
        self._records.append(rec)

    @property
    def head(self) -> str:
        """Hash of the last record (the value to anchor or sign)."""
        return self._log.head

    @property
    def path(self) -> Path | None:
        """The ledger file, or None in memory."""
        return self._path

    def __len__(self) -> int:
        return len(self._records)

    def records(self, kind: str | None = None) -> tuple[LedgerRecord, ...]:
        """All records in order, optionally of one kind (or kind prefix ending in '.', e.g. 'audit.')."""
        if kind is None:
            return tuple(self._records)
        if kind.endswith("."):
            return tuple(r for r in self._records if r.kind.startswith(kind))
        return tuple(r for r in self._records if r.kind == kind)

    def record(self, index: int) -> LedgerRecord:
        """The record at a chain index."""
        if not 0 <= index < len(self._records):
            raise KeyError(f"no ledger record {index}")
        return self._records[index]

    def verify(self) -> None:
        """Recompute the chain and every record's payload; raise `InvariantViolation` at the first mismatch."""
        self._log.verify()
        entries = self._log.entries()
        if len(entries) != len(self._records):
            raise InvariantViolation("ledger records and chain entries differ in number")
        for rec, entry in zip(self._records, entries, strict=True):
            text = _payload_text(rec.kind, _thaw(rec.body), rec.time)
            if text.encode("utf-8") != entry.payload or (rec.prev_hash, rec.payload_hash, rec.entry_hash) != (
                    entry.prev_hash, entry.payload_hash, entry.entry_hash):
                raise InvariantViolation(f"ledger record {rec.index} does not match its committed payload")
        if self._path is not None and self._path.exists():
            FeedbackLedger(self._path, fsync=False, expected_head=self.head)

    # ---- writing
    def append(self, kind: str, body: Mapping[str, Any], *, when: float | None = None) -> LedgerRecord:
        """Commit one record and persist it (module docstring); returns the committed record."""
        if kind not in RECORD_KINDS:
            raise InvariantViolation(f"unknown record kind {kind!r}; kinds: {list(RECORD_KINDS)}")
        t = _time.time() if when is None else float(when)
        text = _payload_text(kind, body, t)
        _, _, decoded = _decode_payload(text, "new record")
        rec_body = _freeze(decoded)
        # Validate the secondary indices before anything is committed.
        if kind == "feedback" and str(decoded["event_id"]) in self._event_index:
            raise InvariantViolation(f"feedback event {str(decoded['event_id'])[:12]}... is already in the ledger")
        if kind in STATE_CHANGING and str(decoded.get("command_id", "")) in self._used:
            raise HumanCommandRequired("this HumanCommand has already authorised a change (single use, AS-831)")
        if self._path is not None:
            self._check_writer()
        entry = self._log.append(text.encode("utf-8"))
        rec = LedgerRecord(entry.index, kind, t, rec_body, entry.prev_hash, entry.payload_hash, entry.entry_hash)
        if self._path is not None:
            line = canonical_json({"index": entry.index, "prev_hash": entry.prev_hash, "payload_hash": entry.payload_hash,
                                   "entry_hash": entry.entry_hash, "payload": text}).encode("ascii") + b"\n"
            self._write(line)
        self._index(rec)
        return rec

    def _check_writer(self) -> None:
        assert self._path is not None
        size = self._path.stat().st_size if self._path.exists() else 0
        if size != self._size:
            raise InvariantViolation(f"{self._path} changed outside this ledger ({size} bytes on disk, {self._size} written); "
                                     "one writer per ledger file")

    def _write(self, line: bytes) -> None:
        assert self._path is not None
        new = not self._path.exists()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("ab") as fh:
            fh.write(line)
            fh.flush()
            if self._fsync:
                os.fsync(fh.fileno())
        if new and self._fsync and os.name == "posix":
            # Make the new directory entry durable as well (POSIX; Windows commits it with the file).
            fd = os.open(str(self._path.parent), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        self._size += len(line)

    # ---- commands
    def register_command(self, command: HumanCommand, *, when: float | None = None) -> str:
        """Record a command (once per id) and return its id."""
        cid = command_id(command)
        if cid not in self._commands:
            self.append("command", command_record(command) | {"id": cid}, when=when)
        return cid

    def command(self, cid: str) -> HumanCommand:
        """The command recorded under an id."""
        if cid not in self._commands:
            raise KeyError(f"no command {cid[:12]}... in the ledger")
        b = self._commands[cid].body
        return HumanCommand(approver=str(b["approver"]), action=str(b["action"]), reason=str(b["reason"]), time=float(b["time"]))

    def assert_unused(self, cid: str) -> None:
        """Raise `HumanCommandRequired` when the command already authorised a state-changing record."""
        if cid in self._used:
            rec = self._used[cid]
            raise HumanCommandRequired(f"the HumanCommand {cid[:12]}... already authorised {rec.kind} (record {rec.index}); "
                                       "a command authorises one change (AS-831)")

    def used_by(self, cid: str) -> LedgerRecord | None:
        """The state-changing record a command authorised, if any."""
        return self._used.get(cid)

    # ---- feedback
    def admit(self, events: Iterable[FeedbackEvent], *, when: float | None = None) -> list[int]:
        """Append already-stamped feedback events; returns their record indices (use `service.ingest`)."""
        out: list[int] = []
        for e in events:
            if not e.provenance.command_id:
                raise InvariantViolation("feedback must be stamped with the id of the command that admits it")
            rec = self.append("feedback", event_to_record(e) | {"event_id": event_id(e)}, when=when)
            out.append(rec.index)
        return out

    def feedback(self, kinds: Iterable[str] | None = None) -> list[tuple[int, FeedbackEvent]]:
        """Admitted feedback events in ledger order, optionally of some kinds: (record index, event)."""
        wanted = None if kinds is None else set(kinds)
        out: list[tuple[int, FeedbackEvent]] = []
        for rec in self._records:
            if rec.kind != "feedback":
                continue
            body = _thaw(rec.body)
            body.pop("event_id")
            if wanted is not None and body.get("kind") not in wanted:
                continue
            out.append((rec.index, event_from_record(body)))
        return out

    def has_event(self, eid: str) -> bool:
        """True when a feedback event with this id is in the ledger."""
        return eid in self._event_index

    def file_digest(self) -> str:
        """SHA-256 of the ledger file as written (empty-string digest in memory)."""
        if self._path is None or not self._path.exists():
            return sha256_hex(b"")
        return sha256_hex(self._path.read_bytes())


def _thaw(obj: Any) -> Any:
    # Plain dicts and lists from the read-only views of `_freeze` (for re-encoding and event decoding).
    if isinstance(obj, Mapping):
        return {k: _thaw(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return [_thaw(v) for v in obj]
    return obj


__all__ = ["EVENT_LOG_PROPOSAL", "LEDGER_VERSION", "RECORD_KINDS", "STATE_CHANGING", "FeedbackLedger", "LedgerRecord"]
