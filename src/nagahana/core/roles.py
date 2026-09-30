"""Roles of the architecture, with the current names and aliases for the old ones.

The names (D-19, [A-10])
------------------------
- **Simulator** takes input and generates, updates and manages the Environment. It holds CVG-AE,
  TSTCT and the physics units ([A-03], [A-05], [A-11], [A-12]).
- **Forecaster** (formerly "Renderer") holds TAAFT plus policy/value heads. It forecasts the next
  K states over N samples into Imagination ([A-14]).
- **Advisor** (formerly "Planner") is a policy/value agent on both caches that produces
  D3FEND-based counter-measure sequences, advisory only ([A-15], [A-20], [Q-13], [Q-30]).
- **Verifier** is the regulator. It holds human feedback ("supplied truth"), analyses memory drift,
  and changes weights only on a human's command ([A-16], [A-21]).
- **Decoder** is a parallel output: a view into the Environment and Imagination through the shared
  latent space ([A-13]).
- **Generator** is training-only data augmentation by event variants ([A-17], [Q-37]).

Why aliases
-----------
Older documents, diagrams and notes say "renderer" and "planner". `resolve()` accepts them and emits
a `DeprecationWarning`, so old configs keep working while every new artefact uses the new names.
"""

from __future__ import annotations

import enum
import warnings


class Role(enum.Enum):
    """The six roles. Values are the canonical config spellings."""

    SIMULATOR = "simulator"
    FORECASTER = "forecaster"
    ADVISOR = "advisor"
    VERIFIER = "verifier"
    DECODER = "decoder"
    GENERATOR = "generator"


#: Old names still found in documents written before 2026-09-29 [A-10].
LEGACY_NAMES: dict[str, Role] = {
    "renderer": Role.FORECASTER,
    "planner": Role.ADVISOR,
}


def resolve(name: str | Role) -> Role:
    """Map a role name (current or legacy, any case) to `Role`.

    Legacy names resolve with a `DeprecationWarning` pointing at the new name.
    """
    if isinstance(name, Role):
        return name
    key = name.strip().lower()
    for role in Role:
        if role.value == key:
            return role
    if key in LEGACY_NAMES:
        new = LEGACY_NAMES[key]
        warnings.warn(
            f"Role name {name!r} is legacy; use {new.value!r} (renamed 2026-09-29, A-10).",
            DeprecationWarning,
            stacklevel=2,
        )
        return new
    raise ValueError(f"Unknown role {name!r}. Roles: {', '.join(r.value for r in Role)}")
