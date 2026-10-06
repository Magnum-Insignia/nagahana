"""The Generator's own engineering assumptions (AS-350 ... AS-370, AS-584 ... AS-586) and how code declares their use.

Why this module exists
----------------------
`governance/assumptions.py` holds AS-01 … AS-41 and is outside the Generator's scope (docs/build-agents.md:
"edit only the files your brief assigns to you"). The Generator needs finer-grained assumptions that
all sit *under* AS-27 (Generator families, standing for held D-14) and AS-28 (physics gate, standing for
proposal P-11). Until they are registered centrally (requested change in the engineer's report),
`use()` does the following:

1. If the ID is registered in `governance.assumptions`, call `assume(id)` directly (records the use;
   raises `DecisionHeld` in strict mode).
2. Otherwise call `assume(parent)` for the umbrella assumption it refines (AS-27 or AS-28). Strict mode
   therefore still blocks every Generator assumption, and the central use-report shows the Generator
   relying on the umbrella. The fine-grained use is recorded locally (`local_uses()`).

The human-readable entries (what is assumed, which held item it stands for, reasoning, evidence) are in
`docs/assumptions/generator.md` (AS-350 ... AS-370) and `docs/assumptions/training-pipeline.md`
(AS-584 ... AS-586). The one-line summaries below must match those documents.

Decisions: D-14 stays HELD (governance/decisions.py); P-11 and P-23 stay PROPOSED. Nothing here
changes their status.
"""

from __future__ import annotations

from collections import defaultdict

from nagahana.governance import assumptions as _central

#: ID → (umbrella assumption it refines, one-line summary). Full text: docs/assumptions/generator.md.
GENERATOR_ASSUMPTIONS: dict[str, tuple[str, str]] = {
    "AS-350": ("AS-27", "flow-only export keeps the IPFIX biflow basic profile; other fields → NOT_SUPPLIED"),
    "AS-351": ("AS-27", "dropping packet-level fields marks Level.PACKET cells NOT_OBSERVABLE"),
    "AS-352": ("AS-27", "1-in-n packet sampling: independent Bernoulli(1/n) per packet, inverse-probability rescaling ×n"),
    "AS-353": ("AS-27", "sensor hiding = an untapped segment of internal hosts; intra-segment updates vanish"),
    "AS-354": ("AS-27", "port remap only inside a same-service alias class; ephemeral source ports inside 49152–65535"),
    "AS-355": ("AS-27", "timing jitter = one scale s per update on duration/IAT (s² on variance), |log s| ≤ log(1+ε)"),
    "AS-356": ("AS-27", "rate scaling = time dilation of malicious updates' event times around the first one"),
    "AS-357": ("AS-27", "true event time uniform within ± reorder_uncertainty_s (NaN = certain); re-sorted"),
    "AS-358": ("AS-27", "topology variation touches benign-only relations: dropout, initiator rewiring to same kind"),
    "AS-359": ("AS-27", "projection of learned cells into the hard limits: pass order and which free side moves"),
    "AS-360": ("AS-28", "gate Φ_phys uses w_c = 1 on raw-unit residuals; with τ = 1e-3 any real violation rejects"),
    "AS-361": ("AS-27", "energy acceptance keeps variants within the [1 %, 99 %] quantiles of real energies"),
    "AS-362": ("AS-27", "masked model codes: 256 signed-log1p bins per numeric column, frequency vocab + OOV"),
    "AS-363": ("AS-27", "MaskGIT cosine mask schedule, T = 8 steps, annealed Gumbel choice noise, stage conditioning"),
    "AS-364": ("AS-27", "autoregressive = same network, left-to-right over records by truncation; 50 % of training"),
    "AS-365": ("AS-27", "diffusion: cosine schedule, ε-prediction, posterior variance β̃, standardised signed-log1p"),
    "AS-366": ("AS-27", "learned variants regenerate 30 % of contributing cells, conditioned on the stage label"),
    "AS-367": ("AS-27", "Generator sources are REAL training-split samples only (stands for P-23); zero-shot refused"),
    "AS-368": ("AS-27", "a variant of an attack window must keep at least one malicious update"),
    "AS-369": ("AS-27", "variant tables: raw_hash zero, record −1, origin/derived_from_seq audit columns"),
    "AS-370": ("AS-27", "budget: attack_share to attack windows by largest remainder; an absent class cedes its share"),
    "AS-584": ("AS-27", "variants reference only entities of their source rows; rewiring picks initiators among them"),
    "AS-585": ("AS-27", "tool-fingerprint swap inside (label class, service, protocol) classes built from training data"),
    "AS-586": ("AS-27", "energy-SSL: denoising-score-matched record energy, annealed Langevin (MALA last level)"),
}

_LOCAL_USES: dict[str, set[str]] = defaultdict(set)


def use(key: str, *, by: str) -> str:
    """Declare that code relies on Generator assumption `key`; returns the ID.

    Raises `DecisionHeld` in strict mode (through the central registry), `KeyError` for unknown IDs.
    """
    if key not in GENERATOR_ASSUMPTIONS:
        raise KeyError(f"unknown Generator assumption {key!r}; known: {', '.join(GENERATOR_ASSUMPTIONS)}")
    try:
        _central.get(key)
        registered = True
    except KeyError:
        registered = False
    # Registered centrally → use it directly; otherwise use the umbrella (strict mode still raises).
    _central.assume(key if registered else GENERATOR_ASSUMPTIONS[key][0], by=by)
    _LOCAL_USES[key].add(by)
    return key


def local_uses() -> dict[str, frozenset[str]]:
    """Which modules used which Generator assumptions in this process."""
    return {k: frozenset(v) for k, v in _LOCAL_USES.items()}
