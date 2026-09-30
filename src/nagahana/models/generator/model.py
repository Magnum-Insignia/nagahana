"""Generator: training-only, physics-bounded event variants (D-40; [A-17], [Q-37], [Q-06]).

The owner's design
------------------
- Purpose: "for a particular event we can slide the observability, signatures, attack traces, etc as
  similar samples for it to have clarity of the partial observable concept" [Q-37]. It is
  "variants of the same event with varying signatures, traces, impacts as observability sliding"
  (ai-mod-arch, component 4), varying "features & relationships that the encoder is capturing".
- Methods: "even the generator uses the same concept of physics informed concept to not hallucinate
  impossible ones, and further uses energy based modeling via ssl, further using Joint Energy Models,
  generative training (where we make the model generate the missing parts etc to make it our expert
  network data generator), autoregressive & diffusion methods" [A-17].
- World-model framing: "the data is just guidance and the generative sampling with self supervision
  is it (or us modeling) to fill the gaps with awareness of boundaries of capabilities" [Q-06].

Contract
--------
    (x̃, 𝒢̃) = G_φ(x, 𝒢, c, ε),        y(x̃, 𝒢̃) = y(x, 𝒢)      (label-preserving)

The controls c are observability (drop packet fields, hide a sensor, sample 1-in-n), signature
(ports, timing jitter, tool fingerprints), trace (another path or technique to the same objective)
and impact (targets, volumes, duration).

Guards
------
- **Training only** (D-40): `PhysicsBoundedSampler.sample` requires RunMode.TRAIN.
- **Physics-bounded** ([A-17], decided): hard limits applied to every variant; Φ_phys reported per
  variant. A hard acceptance threshold Φ_phys ≤ τ is proposal P-11.
- **No leakage** (P-23): the Generator trains on the real *training* split only (pipeline/splits.py).
- Which families come first is held (D-14). Every family entry requires it.

Family lineage: JEM (Grathwohl et al., ICLR 2020; refs.md#L924); denoising diffusion and score SDEs
(refs.md#L938); masked generative modelling (fill in missing parts); autoregressive sequence models.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from nagahana.core.errors import NotBuiltYet
from nagahana.core.modes import RunMode, require_mode
from nagahana.core.registry import Registry
from nagahana.governance import decisions

GENERATORS: Registry[Any] = Registry("generator family")


def _family(name: str, what: str) -> None:
    def factory(**_config: object) -> Any:
        raise NotBuiltYet(f"generator family '{name}': {what}", waiting_on=("D-14",))

    GENERATORS.register(name, requires=("D-14",), summary=what)(factory)


_family("jem", "Joint Energy Model: classifier as energy model, SGLD sampling")
_family("energy-ssl", "energy-based self-supervised generation")
_family("masked-generative", "generate the missing parts (masked modelling)")
_family("autoregressive", "event-by-event sequence generation")
_family("diffusion", "denoising diffusion / score-based generation")


class PhysicsBoundedSampler:
    """Wraps a family with the decided guards. `family` is a built generator (once D-14 is decided).

    Parameters
    ----------
    family: object with `.sample(event, controls)` returning candidate variants.
    enabled_proposals: proposals enabled in this run (the hard gate needs P-11).
    """

    def __init__(self, family: Any, *, enabled_proposals: Collection[str] = ()) -> None:
        self.family = family
        self.enabled = frozenset(enabled_proposals)

    def sample(self, event: Any, controls: Any) -> Any:
        """Draw variants of one real training event. Training only."""
        require_mode(RunMode.TRAIN, component="Generator")
        raise NotBuiltYet("physics-bounded variant sampling", waiting_on=("D-14", "P-11"))

    def hard_gate_enabled(self) -> bool:
        """True only if proposal P-11 (accept iff Φ_phys ≤ τ) is enabled in this run."""
        try:
            decisions.require_proposal("generator-physics-gate", self.enabled)
        except Exception:
            return False
        return True
