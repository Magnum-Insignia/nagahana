"""Ground-truth world simulator (proposal P-14; template, JAX + Equinox).

Purpose: a controllable network world where the *hidden* state is known, so that belief accuracy,
trust estimation, forecast skill and information ceilings (info_audit.py) can be measured exactly.
Public datasets cannot provide this: they have label errors (Engelen et al., IEEE SPW 2021) and no
hidden-state truth.

Planned structure (for review before building):
- **topology**: typed entities (hosts, services, accounts, OT devices) and relation planes, with
  site archetypes (office enterprise, plant with Purdue levels).
- **benign dynamics**: sessions, polling cycles (OT), diurnal load; self-similar aggregate traffic.
- **adversary**: a policy over ATT&CK techniques with preconditions and effects (P-05 skeleton),
  including stealth and pacing options (fast/slow, loud/quiet).
- **observation model g**: taps, sampling, encryption, NetFlow-only regimes, sensor loss. This is the
  lever for "observability sliding" experiments.
- **physics**: the same residual catalogue as physics/residuals.py, so simulated data obey the same
  boundary the model learns.

JAX is imported lazily by the functions that will need it; this module imports nothing heavy.
"""

from __future__ import annotations

from nagahana.core.errors import NotBuiltYet


def simulate(*_args: object, **_kwargs: object) -> object:
    """Simulate one world trajectory with known hidden state (template)."""
    raise NotBuiltYet("ground-truth world simulator", waiting_on=("P-14 approval",))
