"""The `Assumption` record type, shared by the hand-written registry and the generated one."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumption:
    """One engineering assumption.

    id, slug: stable identifiers. title: short name. stands_for: held decision / proposal IDs it
    stands in for ("detail" when it fills an unspecified detail). value: what is assumed.
    why: reasoning and justification (or where they are written). evidence: citations and code-run
    findings. affects: modules or areas.
    """

    id: str
    slug: str
    title: str
    stands_for: tuple[str, ...]
    value: str
    why: str
    evidence: tuple[str, ...] = ()
    affects: tuple[str, ...] = ()
