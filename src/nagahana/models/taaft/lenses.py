"""TAAFT's analysis lenses: where "everything converges that others fail over" [A-19].

TAAFT (Topological Anti-Adversary Foundation Transformer) reads the Environment (TSTCT's KV cache)
and writes the analysis KV cache, which is Imagination [A-14]. CVG-AE and TSTCT "are simply
perceptors not analyzers, they just show what the senses gave in" [A-19]. The analysis happens
here, through a set of lenses. Each lens is a pluggable module in the `LENSES` registry, so lenses
can be added, removed or ablated independently (P-17).

The lenses the owner named [A-19]:

1. **belief-trust**: "the pomdp and observability trust/distrust management, along with belief
   computation". The POSG belief over hidden state, including the adversary's stage, goal and type
   (P-13). Its adversarial marginal is the *suspicion*, with an assume-breach floor: never zero
   [Q-10], ARCH §4.7. Per-source and per-field trust estimates [Q-08]. Update (diagram 04):
       b_t(s) ∝ p(o_t | s, m_t) Σ_{s'} Σ_{a^A} T(s | s', a^A, a^D) π_A(a^A | s') b_{t−1}(s')
2. **energy**: "the energy based transformer as its core energy based modeling approach"
   (energy.py). The energy landscape of stability versus adversarial activity [A-02 item 3].
3. **game**: "game theory, mechanism design". Two-team zero-sum at the abstraction level, with
   coordination inside teams [Q-12], ARCH §3.1. Policy populations with exploiters (PSRO,
   refs.md#L2080; AlphaStar league, #L2087); Stackelberg commitment (#L2094). Where mechanism
   design lives is held (D-24).
4. **information**: "info theory & snr". Mutual information between observables and hidden
   stage; signal-to-noise of evidence.
5. **noise**: "colored noise, white noise, various noise analysis". The owner is "not sure yet"
   (held, D-26). Evidence it is meaningful: benign aggregate traffic is self-similar (Leland et al.,
   IEEE/ACM ToN 1994), and malware beaconing is periodic under jitter (BAYWATCH, DSN 2016).
6. **topology**: "graph & math topology for spatiality with some social networks analysis concpets
   (very slight on sna)". Spectral structure, brokerage and community change (ARCH #25.4).
7. **temporal**: "time series & temporal analysis". Change-point detection (CUSUM/Shiryaev,
   ARCH #25.1), extreme-value thresholds (#25.2), time-to-event hazards (#26.3).
8. **causal**: "causal inference". Interventional reasoning over kill-chain enablement (ARCH §4.4).

Every lens is a template until its internals are specified. Lenses tied to held decisions declare
them in `requires`, so they cannot be built by accident.
"""

from __future__ import annotations

from typing import Any, Protocol

from nagahana.core.errors import NotBuiltYet
from nagahana.core.registry import Registry


class Lens(Protocol):
    """An analysis lens: Environment view (+ current analysis) in, lens features out."""

    name: str

    def analyse(self, environment_view: Any, analysis_state: Any) -> Any:
        """Return this lens's contribution to the analysis cache."""
        ...


LENSES: Registry[Any] = Registry("TAAFT lens")


def _lens(name: str, what: str, *, requires: tuple[str, ...] = (), waiting_on: tuple[str, ...] = ()) -> None:
    class _Template:
        def __init__(self, **_config: object) -> None:
            self.name = name

        def analyse(self, environment_view: Any, analysis_state: Any) -> Any:
            raise NotBuiltYet(f"TAAFT lens '{name}': {what}", waiting_on=waiting_on or requires)

    _Template.__name__ = f"Lens_{name.replace('-', '_')}"
    LENSES.register(name, requires=requires, summary=what)(_Template)


_lens("belief-trust", "POSG belief, suspicion floor, telemetry trust/distrust", waiting_on=("lens design",))
_lens("energy", "energy landscape and refinement (EBT core)", waiting_on=("D-11b",))
_lens("game", "team-level POSG, policy populations, Stackelberg commitment", waiting_on=("D-11c",))
_lens("mechanism-design", "incentive design (placement held)", requires=("D-24",))
_lens("information", "mutual information and SNR of evidence", waiting_on=("lens design",))
_lens("noise", "coloured vs white noise; periodicity; self-similarity", requires=("D-26",))
_lens("topology", "spectral structure, brokerage, community change (SNA-lite)", waiting_on=("lens design",))
_lens("temporal", "change points, extreme-value thresholds, hazards", waiting_on=("lens design",))
_lens("causal", "interventional reasoning over kill-chain enablement", waiting_on=("lens design",))
