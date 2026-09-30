"""TSTCT: Topological Spatio-Temporal Causal Transformer (D-19, [A-10], [A-12]). Template.

Role
----
"Next TSTCT will run by taking this state as input, to generate the spatio-temporal-causal kv cache;
this basically stores the state model & updates of it … which makes up the environment" [A-12].
With CVG-AE, TSTCT is a *perceptor, not an analyser* [A-19]: it organises what the senses gave,
across space, time and cause. Analysis (belief, energy, games) is TAAFT's job.

Structure (diagram 03)
----------------------
Input: per-entity latents z from CVG-AE (models/latent.py) at irregular event times, plus a time
encoding of the gaps Δt with a linear part (trend) and periodic parts (beacons, OT polling cycles):

    τ(Δt) = [ ω_0·Δt + φ_0,  sin(ω_i·Δt + φ_i) ]_{i = 1..k}

Blocks: three head types in parallel (spatial, temporal, causal; masks in `masks.py`), concatenated
and projected, then add & norm · MLP · add & norm, stacked L times. Topology enters as the attention
bias B_h: hypergraph distance and entity/hyperedge kinds. That is the "Topological" in the name.

Output: the Environment KV cache (memory/kvcache.py) plus the refined latent sequence handed to
TAAFT [A-14] and the Decoder [A-13].

Open items this template waits on
---------------------------------
- D-15: is the KV cache the durable Environment, or a view rebuilt from an event log (P-02, P-18)?
- The causal-head structure-learning method (masks.causal_allowed).
- D-11a: what TSTCT's self-supervised pretraining (stage 3) predicts. Candidates: masked states,
  next latent (JEPA-style, ARCH §4.2, refs.md#L735).

Prior art to borrow from (composition method, P-07)
----------------------------------------------------
Dreamer 4's efficient transformer world model (refs.md#L228); TGN event-time memory (Rossi et al.
2020); Transformer-XL segment caching (refs.md#L2388); spatial-temporal-causal transformers such as
STDCformer and CaST (named in the 2026-09-28 review).
"""

from __future__ import annotations

from torch import nn

from nagahana.core.errors import NotBuiltYet
from nagahana.core.registry import Registry

TRANSFORMERS: Registry[type] = Registry("TSTCT")


@TRANSFORMERS.register("tstct", requires=(), summary="template: three head types + topology bias + KV cache")
class TSTCT(nn.Module):
    """Template. See the module docstring."""

    def __init__(self, **config: object) -> None:
        super().__init__()
        self.config = dict(config)

    def forward(self, *args: object, **kwargs: object) -> object:
        raise NotBuiltYet("TSTCT forward (heads, topology bias, KV cache writes)", waiting_on=("D-15", "D-11a"))
