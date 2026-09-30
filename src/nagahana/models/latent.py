"""The shared latent space: what every role exchanges (proposal P-19 names the contract).

Why one space
-------------
"since the latent space is same everywhere, we can deconstruct the imagination space also" [A-13],
and the Advisor works on both caches "ultimately everything is under the same latent space, so it is
possible & practical" [A-20]. For that to hold in code, every role must pass the *same kind* of
latent: produced by CVG-AE, tagged with the version of the space it lives in. Then the Decoder can
check that it is decoding a latent from its own space. KV caches are internal to TSTCT and TAAFT.
They are not this latent (see memory/kvcache.py).

Hybrid, factored (owner's idea ARCH #16; [Q-18])
------------------------------------------------
"Discrete/categorical latents for the combinatorial/structural: topology, entity identity, protocol
class, kill-chain stage … Continuous latents for the metric/quantitative: rates, entropies,
inter-arrival times, behavioral-drift magnitude" (ARCH #16). DreamerV3 found categorical latents
more stable (refs.md#L221).

    z = ( z_cont ∈ ℝ^{Dc},  z_disc ∈ Δ^{C} × … × Δ^{C}  (G groups) )

Variational (D-20)
------------------
CVG-AE is variational [A-11]. Its posterior parameters travel with the sample (`mean`, `logvar`,
`logits`), so the KL terms and uncertainty readouts need no second pass.

Shapes
------
Leading dimensions are free: `[V]` for per-entity latents, `[B, V]` for batches.
- `z_cont`: [..., Dc]
- `z_disc`: [..., G, C] (one-hot samples with straight-through gradients, or relaxed probabilities)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nagahana.core.errors import InvariantViolation


@dataclass(frozen=True)
class LatentState:
    """A point in the shared latent space. See the module docstring."""

    z_cont: torch.Tensor
    z_disc: torch.Tensor
    space: str
    mean: torch.Tensor | None = None
    logvar: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.space:
            raise InvariantViolation("LatentState.space must name the latent-space version")
        if self.z_disc.dim() < 2:
            raise InvariantViolation("z_disc must be [..., G, C]")
        lead_c, lead_d = self.z_cont.shape[:-1], self.z_disc.shape[:-2]
        if lead_c != lead_d:
            raise InvariantViolation(f"leading dims differ: z_cont {tuple(lead_c)} vs z_disc {tuple(lead_d)}")
        for name, t, ref in (("mean", self.mean, self.z_cont), ("logvar", self.logvar, self.z_cont),
                             ("logits", self.logits, self.z_disc)):
            if t is not None and t.shape != ref.shape:
                raise InvariantViolation(f"{name} shape {tuple(t.shape)} ≠ {tuple(ref.shape)}")

    @property
    def leading_shape(self) -> torch.Size:
        """Leading dims shared by both factors (e.g. [V] or [B, V])."""
        return self.z_cont.shape[:-1]

    def flat(self) -> torch.Tensor:
        """Concatenate both factors: [..., Dc + G·C]. For heads that take one vector."""
        return torch.cat([self.z_cont, self.z_disc.flatten(start_dim=-2)], dim=-1)
