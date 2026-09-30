"""Decoder: the parallel output that makes memory visible ([A-13]; D-05, D-11a, P-01, P-20). Template.

Role
----
"the decoder is a parallel output, which also lets us determine/explain & interpret that the
encoder & TSTCT is working correctly while also giving us the direct view into its memory; and
since the latent space is same everywhere, we can deconstruct the imagination space also" [A-13].

So the Decoder serves two uses:
1. **Interpretation.** Decode Environment latents and compare with what was observed. A faithful
   decode is evidence that CVG-AE and TSTCT work.
2. **Live view.** Decode Environment *and* Imagination latents into one readable network graph.
   Every element carries its provenance: observed / believed / forecast (P-01). Beliefs and forecasts
   never render as facts; that is the trust property.

Maths (diagram 02)
------------------
A statistical inverse, i.e. a likelihood. It is not an exact inverse, because encoding loses
detail:

    p_ψ(x | z) = Π_{p ∈ 𝒫} [ Π_{v ∈ V} p_ψ(x_v^p | z)   Π_{e ∈ ℰ_p^cand} p_ψ(e ∈ ℰ_p | z) ]

- Only *contributing* fields enter the node likelihood (P-20; D-11a decides the full target).
- Only *candidate* hyperedges are scored (known topology + pairs the shared hard limits 𝒞 allow),
  never all n² pairs.
- No physics of its own (D-37). Φ_phys is scored on what the Decoder outputs inside the Simulator's
  and Forecaster's losses.
- One graph for people: planes are flattened by a method still held (D-05): typed multigraph,
  supra-graph, reducibility merge (De Domenico et al., Nat. Commun. 2015), or a combination.

Input guard: a latent from another latent space is refused (`LatentState.space`, P-19).
"""

from __future__ import annotations

from torch import nn

from nagahana.core.errors import InvariantViolation, NotBuiltYet
from nagahana.models.latent import LatentState


class ParallelDecoder(nn.Module):
    """Template. See the module docstring."""

    def __init__(self, *, latent_space: str) -> None:
        super().__init__()
        self.latent_space = latent_space

    def _check(self, z: LatentState) -> None:
        if z.space != self.latent_space:
            raise InvariantViolation(f"decoder for space {self.latent_space!r} got a latent from {z.space!r}")

    def decode_fields(self, z: LatentState) -> object:
        """p_ψ(x_v^p | z) for contributing fields."""
        self._check(z)
        raise NotBuiltYet("per-plane field likelihood", waiting_on=("D-11a", "P-20"))

    def decode_edges(self, z: LatentState, candidates: object) -> object:
        """p_ψ(e ∈ ℰ_p | z) for candidate hyperedges."""
        self._check(z)
        raise NotBuiltYet("candidate hyperedge scoring", waiting_on=("D-11a",))

    def flatten_view(self, decoded: object) -> object:
        """Planes → one provenance-tagged view for people."""
        raise NotBuiltYet("view flattening", waiting_on=("D-05", "P-01"))
