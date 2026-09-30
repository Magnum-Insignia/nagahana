"""Parameter budget per component, for ILLUSTRATIVE sizing tiers (not decisions).

The owner asked for "a rough paramter count for each major component of the architecutre, and also
total parameter count? excluding generator (it is separate)" (2026-09-29).

Every width and depth in `conf/` is still `???`, so no single parameter count exists yet. This
module therefore:
1. defines three illustrative tiers matched to the owner's hardware range (edge → enterprise
   workstation → CII server; ARCH #18 "from server gpus to basic rpi");
2. builds each component's *shapes* from real PyTorch modules:
   - the actual reference CVG-AE (models/cvgae);
   - standard transformer encoder/decoder layers for the templated TSTCT, TAAFT and agents;
3. counts parameters exactly for those shapes.

The counts are exact for the stated shapes, and rough for NagaHana, because the shapes are
assumptions.

Design assumptions made visible (each is a sizing choice, not a decision)
-------------------------------------------------------------------------
- **Input layer** (P-22 field embeddings):
  - slot, status and scalar-value embeddings;
  - one hashed categorical table shared by ports, protocols, OT codes and fingerprints.
- **CVG-AE:** P planes × L layers of the reference typed layer, with 7 node kinds and 4 hyperedge
  kinds per plane. Per-kind MLPs make the parameter count scale with (node kinds × planes × layers).
- **TSTCT:** encoder blocks (self-attention across space, time and cause) at width d_t.
- **TAAFT:** decoder-style blocks with *cross-attention*, because it reads the Environment cache
  [A-14]. That gives 16d² per block instead of 12d². There is also one 2-layer MLP per lens (7 lenses).
- **Forecaster:**
  - a latent dynamics transformer for imagination;
  - a policy over ≈ 700 ATT&CK actions (order of magnitude of Enterprise + ICS techniques and
    sub-techniques);
  - value, process-reward, stage and hazard heads.
- **Advisor:** decoder-style blocks that cross-attend to both caches; a policy over ≈ 256 D3FEND
  actions; value and process-reward heads.
- **Verifier:** a small process-reward (step-label) model plus per-output calibration scalars.
- **Decoder:** per-plane field and candidate-edge decoders from the latent z.
- **Memory:** slow-weight read/write projections (the fast memory state is data, not parameters).
- **Generator:** excluded, as requested.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from nagahana.models.cvgae.model import CVGAEEncoder

NODE_KINDS = ("host", "service", "account", "ot_device", "external", "subnet", "application")


@dataclass(frozen=True)
class Tier:
    """One illustrative size. See the module docstring for what each field shapes."""

    name: str
    d_in: int
    cat_rows: int
    field_slots: int
    planes: int
    cvg_dim: int
    cvg_layers: int
    dc: int
    g: int
    c: int
    tst_dim: int
    tst_blocks: int
    tst_heads: int
    taaft_dim: int
    taaft_blocks: int
    taaft_heads: int
    lenses: int
    dyn_blocks: int
    adv_blocks: int
    prm_blocks: int
    attack_actions: int = 700
    defend_actions: int = 256
    stages: int = 14
    horizon_k: int = 32


TIERS: tuple[Tier, ...] = (
    Tier("S (edge / small site)", d_in=32, cat_rows=16_384, field_slots=64, planes=4, cvg_dim=64, cvg_layers=2,
         dc=32, g=4, c=8, tst_dim=256, tst_blocks=4, tst_heads=4, taaft_dim=256, taaft_blocks=6, taaft_heads=4,
         lenses=7, dyn_blocks=2, adv_blocks=1, prm_blocks=1),
    Tier("M (enterprise workstation)", d_in=64, cat_rows=32_768, field_slots=96, planes=4, cvg_dim=128, cvg_layers=3,
         dc=64, g=8, c=16, tst_dim=512, tst_blocks=8, tst_heads=8, taaft_dim=512, taaft_blocks=12, taaft_heads=8,
         lenses=7, dyn_blocks=4, adv_blocks=2, prm_blocks=2),
    Tier("L (CII server)", d_in=128, cat_rows=65_536, field_slots=128, planes=6, cvg_dim=256, cvg_layers=4,
         dc=128, g=16, c=32, tst_dim=1024, tst_blocks=16, tst_heads=16, taaft_dim=1024, taaft_blocks=24, taaft_heads=16,
         lenses=7, dyn_blocks=8, adv_blocks=4, prm_blocks=4),
)


def n_params(m: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def _enc_block(d: int, h: int) -> int:
    return n_params(nn.TransformerEncoderLayer(d, h, dim_feedforward=4 * d, batch_first=True))


def _dec_block(d: int, h: int) -> int:
    return n_params(nn.TransformerDecoderLayer(d, h, dim_feedforward=4 * d, batch_first=True))


def _mlp(i: int, hdn: int, o: int) -> int:
    return n_params(nn.Sequential(nn.Linear(i, hdn), nn.GELU(), nn.Linear(hdn, o)))


def budget(t: Tier) -> dict[str, int]:
    """Parameter count per component for one tier (Generator excluded)."""
    dz = t.dc + t.g * t.c
    out: dict[str, int] = {}
    # Input layer (P-22): slot + status embeddings, scalar value encoder, hashed categorical table.
    out["Input layer (field embeddings)"] = (
        t.field_slots * t.d_in + 5 * t.d_in + n_params(nn.Linear(1, t.d_in)) + t.cat_rows * t.d_in
    )
    # CVG-AE: the actual reference encoder.
    planes = tuple(f"p{i}" for i in range(t.planes))
    enc = CVGAEEncoder(planes=planes, in_dim=t.d_in, dim=t.cvg_dim, layers=t.cvg_layers, node_kinds=NODE_KINDS,
                       edge_kinds={p: ("k0", "k1", "k2", "k3") for p in planes},
                       cont_dim=t.dc, disc_groups=t.g, disc_classes=t.c, latent_space="sizing")
    out["CVG-AE (encoder)"] = n_params(enc)
    # TSTCT: input projection from z, time encoding, encoder blocks, small topology-bias tables.
    out["TSTCT"] = (n_params(nn.Linear(dz, t.tst_dim)) + 16 * t.tst_dim + t.tst_blocks * _enc_block(t.tst_dim, t.tst_heads)
                    + 64 * t.tst_heads)
    # Decoder: per plane, a field decoder (mean and log-scale per slot) and a candidate-edge scorer.
    out["Decoder"] = t.planes * (_mlp(dz, t.cvg_dim, 2 * t.field_slots) + _mlp(2 * dz, t.cvg_dim, 1))
    # Memory: slow-weight read/write projections for Environment and Imagination.
    out["Memory heads (slow weights)"] = 2 * 4 * (max(t.tst_dim, t.taaft_dim) ** 2)
    # TAAFT: projection from TSTCT width, decoder-style blocks (cross-attention to the Environment), lenses.
    out["TAAFT (trunk + lenses)"] = (n_params(nn.Linear(t.tst_dim, t.taaft_dim)) + t.taaft_blocks * _dec_block(t.taaft_dim, t.taaft_heads)
                                     + t.lenses * _mlp(t.taaft_dim, 2 * t.taaft_dim, t.taaft_dim))
    # Forecaster: latent dynamics transformer + heads (policy, value, process reward, stage, hazard, back-projection to z).
    d = t.taaft_dim
    out["Forecaster (dynamics + policy/value)"] = (
        t.dyn_blocks * _enc_block(d, t.taaft_heads) + n_params(nn.Linear(d, t.attack_actions)) + _mlp(d, d, 1) + _mlp(d, d, 1)
        + n_params(nn.Linear(d, t.stages)) + n_params(nn.Linear(d, t.horizon_k)) + n_params(nn.Linear(d, dz))
    )
    # Advisor: decoder-style blocks (cross-attention to both caches) + policy/value/process-reward heads.
    out["Advisor (policy/value)"] = (t.adv_blocks * _dec_block(d, t.taaft_heads) + n_params(nn.Linear(d, t.defend_actions))
                                     + _mlp(d, d, 1) + _mlp(d, d, 1))
    # Verifier: step-label process-reward model + calibration scalars (temperature per output head).
    out["Verifier (step-label PRM + calibration)"] = t.prm_blocks * _enc_block(d, t.taaft_heads) + _mlp(d, d, 1) + 8
    return out


def table() -> str:
    """Markdown table of all tiers (millions of parameters)."""
    budgets = [budget(t) for t in TIERS]
    names = list(budgets[0])
    head = "| Component | " + " | ".join(t.name for t in TIERS) + " |"
    sep = "|---|" + "---:|" * len(TIERS)
    rows = [f"| {n} | " + " | ".join(f"{b[n] / 1e6:,.2f} M" for b in budgets) + " |" for n in names]
    tot = "| **Total (excl. Generator)** | " + " | ".join(f"**{sum(b.values()) / 1e6:,.1f} M**" for b in budgets) + " |"
    return "\n".join([head, sep, *rows, tot])


if __name__ == "__main__":
    print(table())
