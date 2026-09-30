"""Who may read and write which memory region (D-35, D-01; [Q-31], [Q-32], [A-12]–[A-16]).

The three regions
-----------------
- **Environment**: observed facts and their transition history. Built by the Simulator (CVG-AE
  encodes the state, TSTCT writes the spatio-temporal-causal KV cache [A-12]). Beliefs never enter
  it: "imagine you learnt a definition but you want to imagine something of a change on it you
  won't forget this definition and rewrite completely" [Q-31].
- **Imagination**: belief (and suspicion) plus forecasts. Written by the Forecaster; TAAFT writes
  "the kv cache of the analysis" [A-14]. Isolated from facts.
- **Monitor**: deviations and variances of the other two, i.e. memory drift from outcome–forecast
  pairs. Written by the Verifier [A-16], [Q-32].

The matrix (R = read, W = write, – = none)
------------------------------------------
                  Environment  Imagination  Monitor
    Simulator        R W           –           –
    Forecaster       R            R W          –
    Advisor          R            R            –      (memory-less; reads both: D-01, [A-15])
    Verifier         R            R           R W
    Decoder          R            R           (R)     (Monitor read is proposal P-12)
    Generator        –            –            –      (training only; works on datasets)

Everything not listed is denied. The matrix is data (`_DECIDED`, `_PROPOSED`), so changing a rule is
a one-line edit that tests pick up.
"""

from __future__ import annotations

import enum
from collections.abc import Collection

from nagahana.core.errors import AccessDenied
from nagahana.core.roles import Role
from nagahana.governance import decisions


class Region(enum.Enum):
    """The three memory regions."""

    ENVIRONMENT = "environment"
    IMAGINATION = "imagination"
    MONITOR = "monitor"


class Op(enum.Flag):
    """Access rights."""

    NONE = 0
    READ = enum.auto()
    WRITE = enum.auto()


_RW = Op.READ | Op.WRITE

_DECIDED: dict[tuple[Role, Region], Op] = {
    (Role.SIMULATOR, Region.ENVIRONMENT): _RW,
    (Role.FORECASTER, Region.ENVIRONMENT): Op.READ,
    (Role.FORECASTER, Region.IMAGINATION): _RW,
    (Role.ADVISOR, Region.ENVIRONMENT): Op.READ,      # D-01 decided 2026-09-29 [A-15]
    (Role.ADVISOR, Region.IMAGINATION): Op.READ,
    (Role.VERIFIER, Region.ENVIRONMENT): Op.READ,     # outcomes for outcome–forecast pairs [A-16]
    (Role.VERIFIER, Region.IMAGINATION): Op.READ,     # forecasts for the same pairs
    (Role.VERIFIER, Region.MONITOR): _RW,
    (Role.DECODER, Region.ENVIRONMENT): Op.READ,      # "direct view into its memory" [A-13]
    (Role.DECODER, Region.IMAGINATION): Op.READ,      # "we can deconstruct the imagination space also"
}

#: Rights that exist only while a proposal is enabled: (proposal id, role, region) → op.
_PROPOSED: dict[tuple[str, Role, Region], Op] = {
    ("P-12", Role.DECODER, Region.MONITOR): Op.READ,
}


def allowed(role: Role, region: Region, enabled_proposals: Collection[str] = ()) -> Op:
    """Rights of `role` on `region`, including rights granted by enabled proposals."""
    op = _DECIDED.get((role, region), Op.NONE)
    for (pid, r, reg), extra in _PROPOSED.items():
        if r is role and reg is region:
            try:
                decisions.require_proposal(pid, enabled_proposals)
            except Exception:  # proposal not enabled: no extra rights
                continue
            op |= extra
    return op


def check(role: Role, region: Region, op: Op, enabled_proposals: Collection[str] = ()) -> None:
    """Raise `AccessDenied` unless `role` holds every right in `op` on `region`."""
    have = allowed(role, region, enabled_proposals)
    if (have & op) != op:
        raise AccessDenied(
            f"{role.value} may not {op.name or op} {region.value} "
            f"(has: {have.name or 'none'}). See memory/access.py for the matrix and its sources."
        )


def matrix(enabled_proposals: Collection[str] = ()) -> dict[Role, dict[Region, Op]]:
    """The full matrix, e.g. for the CLI or docs."""
    return {role: {reg: allowed(role, reg, enabled_proposals) for reg in Region} for role in Role}
