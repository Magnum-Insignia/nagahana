"""Decoder: the parallel output that makes memory visible ([A-13]; D-05, D-11a, P-01, P-20; build-spec §2.4).

Role
----
"the decoder is a parallel output, which also lets us determine/explain & interpret that the
encoder & TSTCT is working correctly while also giving us the direct view into its memory; and
since the latent space is same everywhere, we can deconstruct the imagination space also" [A-13].

1. **Interpretation / training.** p_ψ(x | z, role) scores the fields of the update behind each
   position (stage-3 reconstruction, AS-04) and p_ψ(e ∈ ℰ_p | z_members) scores candidate hyperedges.
2. **Live view.** Decode Environment *and* Imagination latents into one typed multigraph (AS-29),
   every element tagged OBSERVED, BELIEVED or FORECAST (P-01). Beliefs and forecasts never render as
   facts.

Owner sources, decisions and assumptions
----------------------------------------
AS-04 (reconstruction target: contributing cells + candidate hyperedges, standing for held D-11a /
P-20), AS-29 (view: typed multigraph, standing for held D-05), AS-30 (status weights), AS-33
(categorical → service classes), AS-38 (shared field head conditioned on role and planes; one edge head
per plane), D-18/D-37 (physics: hard limits by construction here; the soft term Φ_phys is scored by the
losses on what this module outputs, the Decoder has no physics of its own), P-19 (latent-space guard).
New: AS-106 (bucket tables, `buckets.py`), AS-110 (likelihood details), AS-112 (provenance rules).

Maths
-----
Conditioning (AS-38):  h = MLP([z ; E_role[role] ; planes]) ∈ ℝ^{hidden}.

Per column c (only *contributing* cells enter, weighted by status, AS-30 / `datamodel.status`:
p(o | s, m) = Π_{i∈O} p(x_i | s)^{w(m_i)}):
- CONTINUOUS / COUNT / HISTOGRAM bin: heteroscedastic Gaussian in signed-log1p space,
      x̃ = sign(x) log(1 + |x|),   −log p = ½ log 2π + s_c + ½ ((x̃ − μ̃_c) / e^{s_c})²,
  s_c = clamp(learned head, [s_min, s_max]). (The density is of x̃; the Jacobian of x ↦ x̃ does not
  depend on ψ and is omitted.)
- BITMASK: per-bit Bernoulli, −log p = Σ_{k < max_bits} BCE(ℓ_{c,k}, bit_k(x)).
- CATEGORICAL: softmax over service-class buckets (AS-33, AS-106), −log p = CE(ℓ_c, bucket(x)).

Hard limits by construction (`physics/constraints.py`, D-18), applied to the log-space means μ̃:
- nonnegative columns (COUNT, HISTOGRAM bins, CONTINUOUS columns with a physical unit in the
  catalogue: durations, gaps, variances, TTLs): μ̃ = softplus(raw) ≥ 0, so x = expm1(μ̃) ≥ 0;
- iat_max ≥ iat_mean: μ̃_max = μ̃_mean + softplus(raw_gap) (`ordered_pair` in log space; expm1 is
  increasing, so the raw values are ordered too);
- flag counts ≤ packets: x_flag = (x_pkts_fwd + x_pkts_bwd) · σ(raw_flag), μ̃_flag = log1p(x_flag);
- every μ̃ is clipped to ±`max_log_mean` (a monotone clip, so the limits above survive it).
Decoded values are signed_expm1(μ̃), the model's median of x (the Gaussian in x̃ is symmetric).

Candidate hyperedges (DeepSets, Zaheer et al., "Deep Sets", NeurIPS 2017, arXiv:1703.06114):
    p(e ∈ ℰ_p | z_e) = σ( ρ_p( [Σ_{u∈e} φ_p(z_u) ; log |e|] ) )
Permutation-invariant in the members by construction.

Invariants (tests/test_perception_decoder.py)
---------------------------------------------
- Hard limits hold for any z (random z tested).
- The NLL is exactly zero on non-contributing cells, finite everywhere, and NaN-free even when
  excluded cells hold NaN.
- Provenance: an element decoded from an Imagination latent (belief or forecast) never carries the
  OBSERVED tag; `DecodedView` refuses to exist otherwise. A latent from another space is refused.

Extension points
----------------
- Further exact limits (e.g. iat_max ≤ duration) can join `_log_means` when the owner wants them hard
  rather than soft.
- A view flattening other than the typed multigraph needs D-05 (held).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from nagahana.core.errors import InvariantViolation
from nagahana.datamodel.columnar import STATUS_CODE
from nagahana.datamodel.fields import CATALOGUE, Kind
from nagahana.datamodel.status import ObservationStatus
from nagahana.governance.assumptions import assume
from nagahana.models.config.components import DecoderConfig
from nagahana.models.decoder.buckets import service_class
from nagahana.models.inputs.encoder import CONTRIBUTING_STATUS, float_to_code
from nagahana.models.latent import LatentState
from nagahana.models.vocab import COLUMN_KIND_CODE, HYPEREDGE_KINDS, MASK_STATUS, N_STATUS_CODES, PROVENANCE, ROLES
from nagahana.nn.norms import RMSNorm
from nagahana.nn.numeric import signed_expm1, signed_log1p

_KIND_CONT = COLUMN_KIND_CODE[Kind.CONTINUOUS]
_KIND_COUNT = COLUMN_KIND_CODE[Kind.COUNT]
_KIND_CAT = COLUMN_KIND_CODE[Kind.CATEGORICAL]
_KIND_BIT = COLUMN_KIND_CODE[Kind.BITMASK]
_KIND_HIST = COLUMN_KIND_CODE[Kind.HISTOGRAM]
_NUMERIC = (_KIND_CONT, _KIND_COUNT, _KIND_HIST)
_LOG_2PI = math.log(2 * math.pi)
_FLAGS = ("syn", "ack", "fin", "rst", "psh", "urg")

#: AS-30 status weights for the likelihood: OBSERVED 1.0, STALE 0.5, LOW_RELIABILITY 0.5, others 0.
STATUS_WEIGHT: torch.Tensor = torch.zeros(N_STATUS_CODES)
STATUS_WEIGHT[STATUS_CODE[ObservationStatus.OBSERVED]] = 1.0
STATUS_WEIGHT[STATUS_CODE[ObservationStatus.STALE]] = 0.5
STATUS_WEIGHT[STATUS_CODE[ObservationStatus.LOW_RELIABILITY]] = 0.5


def field_id(column_name: str) -> str:
    """Catalogue ID of a column (histogram bins `"<id>[i]"` → `"<id>"`)."""
    return column_name.split("[", 1)[0]


def is_nonnegative(column_name: str, kind_code: int) -> bool:
    """AS-110: COUNT and HISTOGRAM columns, and CONTINUOUS columns with a physical unit, are ≥ 0."""
    if kind_code in (_KIND_COUNT, _KIND_HIST):
        return True
    spec = CATALOGUE.get(field_id(column_name))
    return kind_code == _KIND_CONT and spec is not None and spec.unit is not None


# ================================================================================ outputs
@dataclass
class DecodedFields:
    """Decoder output for rows [...] (any leading shape).

    log_mean, log_scale: [..., C] Gaussian parameters in signed-log1p space (0 on non-numeric columns).
    values: [..., C] decoded values in raw space, hard limits satisfied: numeric → signed_expm1(μ̃)
        (flag counts exactly (pkts_fwd + pkts_bwd)·σ); bitmask → the mode code (bits with p > ½);
        categorical → the most probable service-class bucket (AS-106).
    bit_logits: [..., C_bit, max_bits]; class_logits: [..., C_cat, n_classes].
    column_names, column_kind: the column layout (kind codes of `vocab.COLUMN_KINDS`).
    """

    log_mean: torch.Tensor
    log_scale: torch.Tensor
    values: torch.Tensor
    bit_logits: torch.Tensor
    class_logits: torch.Tensor
    column_names: tuple[str, ...]
    column_kind: tuple[int, ...]


@dataclass(frozen=True)
class ViewElement:
    """One element of the decoded view (AS-29: a typed multigraph, planes kept as edge types).

    kind: "field" (a decoded or observed field of a row's entity) or "hyperedge".
    plane: plane name for hyperedges, None for fields. entities: the entity ids involved.
    name: column name, or hyperedge kind (`vocab.HYPEREDGE_KINDS`, "candidate" if unknown).
    value: field value, or the hyperedge's decoded probability. provenance: one of `vocab.PROVENANCE`.
    """

    kind: str
    plane: str | None
    entities: tuple[int, ...]
    name: str
    value: float
    provenance: str


SOURCES: tuple[str, ...] = ("environment", "belief", "forecast")


@dataclass(frozen=True)
class DecodedView:
    """A provenance-tagged view. Refuses OBSERVED elements unless decoded from the Environment."""

    source: str
    latent_space: str
    elements: tuple[ViewElement, ...]

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise InvariantViolation(f"unknown view source {self.source!r}; known: {SOURCES}")
        for el in self.elements:
            if el.provenance not in PROVENANCE:
                raise InvariantViolation(f"unknown provenance tag {el.provenance!r}")
            if el.provenance == "observed" and self.source != "environment":
                raise InvariantViolation("an element decoded from Imagination can never carry the OBSERVED tag (P-01)")
            if self.source == "forecast" and el.provenance != "forecast":
                raise InvariantViolation("every element of a forecast view is FORECAST")


def provenance_tag(source: str, backed_by_observation: bool) -> str:
    """AS-112: OBSERVED only for Environment elements backed by an observation; FORECAST for forecasts;
    BELIEVED otherwise (Environment inferences and Imagination beliefs)."""
    if source not in SOURCES:
        raise InvariantViolation(f"unknown view source {source!r}")
    if source == "forecast":
        return "forecast"
    if source == "environment" and backed_by_observation:
        return "observed"
    return "believed"


# ================================================================================ decoder
class Decoder(nn.Module):
    """Field likelihoods, hard-limited decoding, candidate-hyperedge scores and the provenance view.

    Parameters
    ----------
    cfg: `DecoderConfig`.
    latent_dim: dz = Dc + G·C.
    column_kind: kind code per column (`vocab.COLUMN_KINDS`); column_names: names (field IDs, bins).
    n_planes: number of relation planes (edge heads, plane conditioning).
    latent_space: version tag of the latent space this decoder reads (P-19).
    plane_names: plane names (defaults to `vocab.PLANES[:n_planes]`), for the view.
    """

    def __init__(
        self,
        cfg: DecoderConfig,
        *,
        latent_dim: int,
        column_kind: Sequence[int],
        column_names: Sequence[str],
        n_planes: int,
        latent_space: str,
        plane_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        for a in ("AS-04", "AS-29", "AS-30", "AS-33", "AS-38"):
            assume(a, by=__name__)
        if len(column_kind) != len(column_names):
            raise InvariantViolation("column_kind and column_names differ in length")
        if not latent_space:
            raise InvariantViolation("latent_space must name the latent-space version (P-19)")
        from nagahana.models.vocab import PLANES

        self.cfg, self.latent_dim, self.latent_space = cfg, latent_dim, latent_space
        self.column_kind = tuple(int(k) for k in column_kind)
        self.column_names = tuple(column_names)
        self.n_planes = n_planes
        self.plane_names = tuple(plane_names) if plane_names is not None else PLANES[:n_planes]
        kinds = self.column_kind
        # ---- column groups (index lists into the C columns)
        self.num_cols = [i for i, k in enumerate(kinds) if k in _NUMERIC]
        self.bit_cols = [i for i, k in enumerate(kinds) if k == _KIND_BIT]
        self.cat_cols = [i for i, k in enumerate(kinds) if k == _KIND_CAT]
        if len(self.num_cols) + len(self.bit_cols) + len(self.cat_cols) != len(kinds):
            raise InvariantViolation(f"unknown column kind codes in {kinds}")
        self._num_pos = {c: j for j, c in enumerate(self.num_cols)}         # column → position in numeric head
        self.nonneg = [is_nonnegative(self.column_names[c], kinds[c]) for c in self.num_cols]
        names = {field_id(n): i for i, n in enumerate(self.column_names)}
        # Ordered pair iat_mean ≤ iat_max (only when both columns exist and are numeric).
        self.ordered: list[tuple[int, int]] = []
        if "flow.iat_mean" in names and "flow.iat_max" in names:
            lo, hi = names["flow.iat_mean"], names["flow.iat_max"]
            if lo in self._num_pos and hi in self._num_pos:
                self.ordered.append((self._num_pos[lo], self._num_pos[hi]))
        # Flag counts ≤ packets_fwd + packets_bwd (only when the packet columns exist).
        self.flag_cols: list[int] = []
        self.pkt_cols: tuple[int, int] | None = None
        if "flow.packets_fwd" in names and "flow.packets_bwd" in names:
            pf, pb = names["flow.packets_fwd"], names["flow.packets_bwd"]
            if pf in self._num_pos and pb in self._num_pos:
                self.pkt_cols = (self._num_pos[pf], self._num_pos[pb])
                self.flag_cols = [self._num_pos[names[f"flow.flag_count.{f}"]] for f in _FLAGS
                                  if f"flow.flag_count.{f}" in names and names[f"flow.flag_count.{f}"] in self._num_pos]
        # ---- conditioning trunk (AS-38): z, role, plane membership
        self.role = nn.Embedding(len(ROLES), cfg.role_dim)
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + cfg.role_dim + n_planes, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(), RMSNorm(cfg.hidden),
        )
        # ---- per-kind heads
        self.num_head = nn.Linear(cfg.hidden, 2 * len(self.num_cols)) if self.num_cols else None
        self.bit_head = nn.Linear(cfg.hidden, len(self.bit_cols) * cfg.max_bits) if self.bit_cols else None
        self.cat_head = nn.Linear(cfg.hidden, len(self.cat_cols) * cfg.n_service_classes) if self.cat_cols else None
        # ---- candidate-hyperedge heads, one DeepSets pair (φ_p, ρ_p) per plane (AS-38)
        self.edge_phi = nn.ModuleList(
            [nn.Sequential(nn.Linear(latent_dim, cfg.edge_hidden), nn.SiLU(), nn.Linear(cfg.edge_hidden, cfg.edge_hidden))
             for _ in range(n_planes)]
        )
        self.edge_rho = nn.ModuleList(
            [nn.Sequential(nn.Linear(cfg.edge_hidden + 1, cfg.edge_hidden), nn.SiLU(), nn.Linear(cfg.edge_hidden, 1))
             for _ in range(n_planes)]
        )
        self.register_buffer("_contrib", CONTRIBUTING_STATUS.clone(), persistent=False)
        self.register_buffer("_weight", STATUS_WEIGHT.clone(), persistent=False)
        self._contrib: torch.Tensor
        self._weight: torch.Tensor

    # ------------------------------------------------------------------------------ guards
    def _z(self, z: torch.Tensor | LatentState) -> torch.Tensor:
        """The flat latent; a `LatentState` from another latent space is refused (P-19)."""
        if isinstance(z, LatentState):
            if z.space != self.latent_space:
                raise InvariantViolation(f"decoder for space {self.latent_space!r} got a latent from {z.space!r}")
            z = z.flat()
        if z.shape[-1] != self.latent_dim:
            raise InvariantViolation(f"latent width {z.shape[-1]} ≠ decoder latent_dim {self.latent_dim}")
        return z

    # ------------------------------------------------------------------------------ heads
    def _hidden(self, z: torch.Tensor, role: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
        """h = trunk([z ; E_role[role] ; planes]) for rows [...]."""
        x = torch.cat([z.float(), self.role(role.long().clamp(0, len(ROLES) - 1)), planes.float()], dim=-1)
        return self.trunk(x)

    def _log_means(self, raw_mean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Constrained log-space means μ̃ [..., Cn] and the raw flag values [..., F] (hard limits)."""
        cols = [F.softplus(raw_mean[..., j]) if nn_ else raw_mean[..., j] for j, nn_ in enumerate(self.nonneg)]
        for lo, hi in self.ordered:                                          # μ̃_max = μ̃_mean + softplus(gap)
            cols[hi] = cols[lo] + F.softplus(raw_mean[..., hi])
        lim = self.cfg.max_log_mean
        cols = [c.clamp(-lim, lim) for c in cols]                            # monotone clip
        flags_raw: torch.Tensor | None = None
        if self.pkt_cols is not None and self.flag_cols:
            total = torch.expm1(cols[self.pkt_cols[0]]) + torch.expm1(cols[self.pkt_cols[1]])   # raw packets ≥ 0
            fr = [total * torch.sigmoid(raw_mean[..., j]) for j in self.flag_cols]
            for j, x in zip(self.flag_cols, fr, strict=True):
                cols[j] = torch.log1p(x)
            flags_raw = torch.stack(fr, dim=-1)
        return torch.stack(cols, dim=-1), flags_raw

    def _decode(self, z: torch.Tensor, role: torch.Tensor, planes: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """All head outputs for rows [...]."""
        h = self._hidden(z, role, planes)
        out: dict[str, torch.Tensor | None] = {"log_mean": None, "log_scale": None, "flags_raw": None,
                                               "bit_logits": None, "class_logits": None}
        if self.num_head is not None:
            raw = self.num_head(h)
            n = len(self.num_cols)
            mu, flags_raw = self._log_means(raw[..., :n])
            out["log_mean"], out["flags_raw"] = mu, flags_raw
            out["log_scale"] = raw[..., n:].clamp(self.cfg.log_scale_min, self.cfg.log_scale_max)
        if self.bit_head is not None:
            out["bit_logits"] = self.bit_head(h).unflatten(-1, (len(self.bit_cols), self.cfg.max_bits))
        if self.cat_head is not None:
            out["class_logits"] = self.cat_head(h).unflatten(-1, (len(self.cat_cols), self.cfg.n_service_classes))
        return out

    # ------------------------------------------------------------------------------ likelihood
    def cell_weight(self, status: torch.Tensor, true_status: torch.Tensor | None = None, *,
                    mask_weight: float = 1.0) -> torch.Tensor:
        """w per cell [..., C]: AS-30 weight of the true status, × mask_weight where the encoder saw MASK."""
        ts = status if true_status is None else true_status
        w = self._weight[ts.long()]
        return torch.where(status == MASK_STATUS, w * mask_weight, w)

    def field_nll(
        self,
        z: torch.Tensor | LatentState,
        role: torch.Tensor,
        planes: torch.Tensor,
        fields_values: torch.Tensor,
        fields_status: torch.Tensor,
        *,
        true_status: torch.Tensor | None = None,
        mask_weight: float = 1.0,
    ) -> torch.Tensor:
        """Weighted per-cell NLL [..., C]; exactly 0 where the cell does not contribute.

        z [..., dz]; role [...]; planes [..., n_planes] bool; fields_values [..., C] (the *true* values,
        NaN allowed where not contributing); fields_status [..., C] the statuses the encoder saw (MASK
        on masked cells); true_status [..., C] the original statuses (default: fields_status). A cell
        contributes when its true status is OBSERVED / STALE / LOW_RELIABILITY (AS-04): masked cells are
        scored against their true value, up-weighted by `mask_weight` (build-spec §3).
        """
        z_ = self._z(z)
        ts = fields_status if true_status is None else true_status
        if ts.shape != fields_values.shape or fields_status.shape != fields_values.shape:
            raise InvariantViolation("values / status / true_status must share a shape")
        contrib = self._contrib[ts.long()]                                   # [..., C]
        x = torch.where(contrib, fields_values, torch.zeros((), dtype=fields_values.dtype)).float()   # NaN-free
        dec = self._decode(z_, role, planes)
        nll = x.new_zeros(x.shape)
        if self.num_cols:
            idx = torch.tensor(self.num_cols)
            mu, s = dec["log_mean"], dec["log_scale"]
            assert mu is not None and s is not None
            xt = signed_log1p(x[..., idx])
            g = 0.5 * _LOG_2PI + s + 0.5 * ((xt - mu) * torch.exp(-s)) ** 2
            nll = nll.index_copy(-1, idx, g)
        if self.bit_cols:
            idx = torch.tensor(self.bit_cols)
            bl = dec["bit_logits"]
            assert bl is not None
            code = float_to_code(x[..., idx]).clamp_min(0)
            bits = torch.bitwise_and(torch.bitwise_right_shift(code.unsqueeze(-1), torch.arange(self.cfg.max_bits)), 1).float()
            bce = F.binary_cross_entropy_with_logits(bl, bits, reduction="none").sum(-1)
            nll = nll.index_copy(-1, idx, bce)
        if self.cat_cols:
            idx = torch.tensor(self.cat_cols)
            cl = dec["class_logits"]
            assert cl is not None
            code = float_to_code(x[..., idx])
            target = torch.stack([service_class(self.column_names[c], code[..., j], self.cfg.n_service_classes)
                                  for j, c in enumerate(self.cat_cols)], dim=-1)
            ce = -torch.gather(F.log_softmax(cl, dim=-1), -1, target.unsqueeze(-1)).squeeze(-1)
            nll = nll.index_copy(-1, idx, ce)
        w = self.cell_weight(fields_status, true_status, mask_weight=mask_weight)
        return torch.where(contrib, nll * w, torch.zeros((), dtype=nll.dtype))

    # ------------------------------------------------------------------------------ decoding
    def decode_fields(self, z: torch.Tensor | LatentState, role: torch.Tensor, planes: torch.Tensor) -> DecodedFields:
        """Decoded fields for rows [...], with every hard limit satisfied (see the module docstring)."""
        z_ = self._z(z)
        dec = self._decode(z_, role, planes)
        lead = z_.shape[:-1]
        c = len(self.column_names)
        log_mean = z_.new_zeros(*lead, c, dtype=torch.float32)
        log_scale = z_.new_zeros(*lead, c, dtype=torch.float32)
        values = z_.new_zeros(*lead, c, dtype=torch.float32)
        if self.num_cols:
            idx = torch.tensor(self.num_cols)
            mu, s = dec["log_mean"], dec["log_scale"]
            assert mu is not None and s is not None
            raw = signed_expm1(mu)
            if dec["flags_raw"] is not None:                                 # flags exactly total·σ (no round trip)
                raw = raw.index_copy(-1, torch.tensor(self.flag_cols), dec["flags_raw"])
            log_mean = log_mean.index_copy(-1, idx, mu)
            log_scale = log_scale.index_copy(-1, idx, s)
            values = values.index_copy(-1, idx, raw)
        bit_logits = dec["bit_logits"] if dec["bit_logits"] is not None else z_.new_zeros(*lead, 0, self.cfg.max_bits)
        class_logits = dec["class_logits"] if dec["class_logits"] is not None else z_.new_zeros(*lead, 0, self.cfg.n_service_classes)
        if self.bit_cols:
            weights = 2.0 ** torch.arange(self.cfg.max_bits, dtype=torch.float32)
            mode = ((bit_logits > 0).float() * weights).sum(-1)              # bits with p > ½
            values = values.index_copy(-1, torch.tensor(self.bit_cols), mode)
        if self.cat_cols:
            values = values.index_copy(-1, torch.tensor(self.cat_cols), class_logits.argmax(-1).float())
        return DecodedFields(log_mean=log_mean, log_scale=log_scale, values=values, bit_logits=bit_logits,
                             class_logits=class_logits, column_names=self.column_names, column_kind=self.column_kind)

    def physics_inputs(self, decoded: DecodedFields) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """(values, contributing) keyed by catalogue field ID, rows flattened to [N], for `PhysicsTerm`.

        Only numeric, non-histogram columns are exported (residuals read scalar fields). Decoded values
        are model outputs, so every exported field contributes on every row (D-18: the boundary applies
        to everything the model produces).
        """
        vals: dict[str, torch.Tensor] = {}
        contrib: dict[str, torch.Tensor] = {}
        for c in self.num_cols:
            name = self.column_names[c]
            if "[" in name:
                continue
            v = decoded.values[..., c].reshape(-1)
            vals[name] = v
            contrib[name] = torch.ones_like(v, dtype=torch.bool)
        return vals, contrib

    # ------------------------------------------------------------------------------ hyperedges
    def edge_logits(self, plane: int, member_z: torch.Tensor, member_mask: torch.Tensor) -> torch.Tensor:
        """Logit of p(e ∈ ℰ_plane | members) [E] from member latents [E, m, dz] and mask [E, m] (DeepSets)."""
        if not 0 <= plane < self.n_planes:
            raise InvariantViolation(f"plane index {plane} outside [0, {self.n_planes})")
        mz = self._z(member_z)
        mask = member_mask.to(torch.bool)
        phi = self.edge_phi[plane](mz.float())                               # [E, m, H]
        pooled = (phi * mask.unsqueeze(-1).float()).sum(dim=1)               # Σ over real members
        size = torch.log(mask.sum(dim=1).clamp_min(1).float()).unsqueeze(-1)
        return self.edge_rho[plane](torch.cat([pooled, size], dim=-1)).squeeze(-1)

    # ------------------------------------------------------------------------------ view
    def view(
        self,
        latent: LatentState,
        *,
        source: str,
        entity: torch.Tensor,
        role: torch.Tensor,
        planes: torch.Tensor,
        observed_values: torch.Tensor | None = None,
        observed_status: torch.Tensor | None = None,
        candidate_edges: Mapping[str, torch.Tensor] | None = None,
        candidate_kinds: Mapping[str, torch.Tensor] | None = None,
        observed_edges: Mapping[str, torch.Tensor] | None = None,
        edge_threshold: float = 0.5,
    ) -> DecodedView:
        """The provenance-tagged typed multigraph (AS-29, AS-112) for rows [N] of `latent`.

        source: "environment" (TSTCT/CVG-AE latents), "belief" (TAAFT readouts), "forecast" (Forecaster).
        entity [N]: entity id of each row; role [N]; planes [N, n_planes].
        observed_values / observed_status [N, C]: the observation behind each row (Environment only).
        candidate_edges: plane name → long [E, m] row indices into the latent (−1 pad);
        candidate_kinds: plane name → long [E] hyperedge kinds (optional);
        observed_edges: plane name → bool [E], True where the candidate was observed (Environment only).
        Observed edges are always shown; other candidates only when p ≥ edge_threshold.
        """
        if not isinstance(latent, LatentState):
            raise InvariantViolation("view() takes a LatentState so its latent space can be checked (P-19)")
        z = self._z(latent)                                                  # also the space guard
        if source not in SOURCES:
            raise InvariantViolation(f"unknown view source {source!r}; known: {SOURCES}")
        if source != "environment" and (observed_values is not None or observed_status is not None or observed_edges):
            raise InvariantViolation("Imagination latents carry no observations; observed inputs are refused (P-01)")
        if z.dim() != 2:
            raise InvariantViolation("view() expects rows [N, dz]")
        elements: list[ViewElement] = []
        with torch.no_grad():
            dec = self.decode_fields(z, role, planes)
            obs = None
            if observed_values is not None and observed_status is not None:
                obs = self._contrib[observed_status.long()]
            for i in range(z.shape[0]):
                ent = (int(entity[i]),)
                for c, name in enumerate(self.column_names):
                    backed = bool(obs[i, c]) if obs is not None else False
                    value = float(observed_values[i, c]) if (backed and observed_values is not None) else float(dec.values[i, c])
                    elements.append(ViewElement("field", None, ent, name, value, provenance_tag(source, backed)))
            for plane, members in (candidate_edges or {}).items():
                if plane not in self.plane_names:
                    raise InvariantViolation(f"unknown plane {plane!r}")
                pi = self.plane_names.index(plane)
                mask = members >= 0
                logits = self.edge_logits(pi, z[members.clamp_min(0)], mask)
                prob = torch.sigmoid(logits)
                seen = observed_edges.get(plane) if observed_edges else None
                kinds = candidate_kinds.get(plane) if candidate_kinds else None
                for e in range(members.shape[0]):
                    backed = bool(seen[e]) if seen is not None else False
                    if not backed and float(prob[e]) < edge_threshold:
                        continue
                    ents = tuple(int(entity[r]) for r in members[e][mask[e]].tolist())
                    kind = HYPEREDGE_KINDS[int(kinds[e])] if kinds is not None else "candidate"
                    elements.append(ViewElement("hyperedge", plane, ents, kind, float(prob[e]), provenance_tag(source, backed)))
        return DecodedView(source=source, latent_space=self.latent_space, elements=tuple(elements))


#: The template's name, kept so existing references resolve (roles/decoder.py documents it).
ParallelDecoder = Decoder
