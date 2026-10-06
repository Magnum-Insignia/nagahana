"""Tensor contracts between components (build-spec §1–§2). Every component reads and writes these.

Conventions
-----------
- B windows per batch, U updates per window, C columns, P positions (entity states), V entities,
  N nodes of the CVG-AE union graph, M triggers per window.
- **Times are float64 seconds relative to the window origin** (`WindowBatch.origin`, float64 epoch
  seconds). Never float32 epoch seconds (D-49 precision note).
- **Precision (D-54):** weights are stored and served in float32 and the networks compute in float32
  (context, hypotheses, states, latents, K/V are float32). *Outputs* are float64: the Forecaster's
  P_inf, band, median, stage mixture, hazards and route weights; TAAFT's posterior readouts, energies,
  lens shares and energy trace. Each field's dtype is written in its class docstring below.
- Padding: every padded axis has a boolean mask (True = real). Index tensors use −1 for "none".
- Window-local indices: `entities` index rows of the window's entity table [0, V); positions index
  [0, P); updates index [0, U). Flattened indices (b·U + u etc.) are named `flat_*`.
- **No future leakage** is the builder's duty: every structure that a position at time t may read
  is either time-stamped here (so masks can test it) or built as of t (`GraphBatch`).

Ownership
---------
The data pipeline (`data/`) builds `WindowBatch`; the graph builder (`graph/`) builds `GraphBatch`
and the contact matrices; the model fills the outputs (`LatentOut`, `EnvironmentOut`, `TriggerView`,
`AnalysisOut`, `ForecastOut`). Labels (`LabelBatch`) never enter the model's forward pass: labels are
evaluation and training data, not observations (ingest/csv_flows.py, rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nagahana.nn.blocks import KV


# ================================================================================ inputs
@dataclass
class FieldBatch:
    """The value/status matrices of the updates of B windows (from `ColumnarUpdates`).

    values: float32 [B, U, C]. Any content where the cell does not contribute (NaN allowed); the
        FieldEncoder never reads a non-contributing value (D-41).
    status: long [B, U, C]. Codes of `datamodel.columnar.STATUS_ORDER`, or `vocab.MASK_STATUS` for
        cells hidden by self-supervised masking.
    column_kind: long [C]. Codes of `vocab.COLUMN_KINDS`.
    column_slot: long [C]. Row of the field-slot table for each column (stable per field ID, P-22).
    column_names: the column names (for explanations: attributions are reported per column).
    """

    values: torch.Tensor
    status: torch.Tensor
    column_kind: torch.Tensor
    column_slot: torch.Tensor
    column_names: tuple[str, ...]


@dataclass
class WindowBatch:
    """B windows of the update stream, with their entity tables and time-stamped contact structure.

    origin: float64 [B]: epoch seconds of each window's time zero (for clock features, D-50).
    update_time: float64 [B, U]: event time relative to origin.
    update_entities: long [B, U, 3]: (initiator, responder, service) entity indices, −1 = none.
    update_relation: long [B, U]: hyperedge (relation) id of the update in the window, −1 = none.
    update_planes: bool [B, U, n_planes]: planes the update's relation belongs to (AS-01).
    update_mask: bool [B, U].
    reorder_uncertainty: float32 [B, U]: seconds within which the update's order is uncertain
        (`OrderingInfo`); stage-3 may permute inside it (build-spec §4b.5).
    entity_kind: long [B, V] (codes of `vocab.NODE_KINDS`); entity_internal: bool [B, V];
        entity_mask: bool [B, V].
    contact1: float64 [B, V, V]: first time two entities shared a hyperedge (any plane), +inf never.
    contact2: float64 [B, V, V]: first time a 2-hop path existed, min_w max(C¹[u,w], C¹[w,v]).
    contact_planes: float64 [B, V, V, n_planes]: first contact time per plane, +inf never.
    """

    fields: FieldBatch
    origin: torch.Tensor
    update_time: torch.Tensor
    update_entities: torch.Tensor
    update_relation: torch.Tensor
    update_planes: torch.Tensor
    update_mask: torch.Tensor
    reorder_uncertainty: torch.Tensor
    entity_kind: torch.Tensor
    entity_internal: torch.Tensor
    entity_mask: torch.Tensor
    contact1: torch.Tensor
    contact2: torch.Tensor
    contact_planes: torch.Tensor
    positions: PositionBatch
    graph: GraphBatch
    triggers: TriggerBatch


@dataclass
class PositionBatch:
    """Entity states (TSTCT positions): 2 per update, initiator and responder (build-spec §1.1).

    Sorted by (time, update, role) within each window, so position order is a valid time order.
    entity: long [B, P]; time: float64 [B, P]; update: long [B, P]; role: long [B, P] (vocab.ROLES);
    mask: bool [B, P].
    next_index: long [B, P]: the same entity's next position in the window (−1 if none). Target of
        the TSTCT transition prior (the world model's P(S_{t+1} | S_≤t)).
    next_dt: float64 [B, P]: time to that next position (0 where none).
    """

    entity: torch.Tensor
    time: torch.Tensor
    update: torch.Tensor
    role: torch.Tensor
    mask: torch.Tensor
    next_index: torch.Tensor
    next_dt: torch.Tensor


@dataclass
class GraphBatch:
    """Local hypergraphs as of each position's time, as one disjoint union (build-spec §2.2–2.3).

    node_update: long [N]: flat update index (b·U + u) of the node's latest update as of the centre
        time, −1 if the node has none yet (learned "unseen" vector).
    node_role: long [N]: the node's role in that update (vocab.ROLES; 3 = none).
    node_kind: long [N]; node_age: float32 [N]: seconds since that update (0 for none).
    node_hop: long [N]: hop distance from the centre (0 = centre).
    node_owner: long [N]: flat position index (b·P + p) whose subgraph this node belongs to.
    center: long [B·P]: node index of each position's centre node (−1 for padded positions).
    incidence: plane → long [2, nnz] of (node, hyperedge) pairs over the union graph.
    hyperedge_kind: plane → long [E_p] (codes of vocab.HYPEREDGE_KINDS).
    hyperedge_entities: plane → long [E_p, max_members] window entity indices of members (−1 pad);
        used by the Decoder's candidate-edge likelihood.
    rwse: float32 [N, n_planes, rwse_steps] (D-49).
    """

    node_update: torch.Tensor
    node_role: torch.Tensor
    node_kind: torch.Tensor
    node_age: torch.Tensor
    node_hop: torch.Tensor
    node_owner: torch.Tensor
    center: torch.Tensor
    incidence: dict[str, torch.Tensor]
    hyperedge_kind: dict[str, torch.Tensor]
    hyperedge_entities: dict[str, torch.Tensor]
    rwse: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.node_kind.shape[0])


@dataclass
class TriggerBatch:
    """Forecaster triggers inside each window (fixed cadence + capped priority, AS-12).

    time: float64 [B, M] (relative to origin); mask: bool [B, M].
    entity_latest: long [B, M, V]: index of each entity's latest position with time ≤ trigger time
        (−1 = entity not yet seen; such entities are inactive at that trigger).
    """

    time: torch.Tensor
    mask: torch.Tensor
    entity_latest: torch.Tensor


@dataclass
class LabelBatch:
    """Supervision (never a model input). −1 / NaN / +inf mean unknown or never.

    update_malicious: float32 [B, U] in {0, 1} or NaN (unknown).
    update_stage: long [B, U] stage code (vocab.STAGES) or −1.
    update_technique: long [B, U] technique slot or −1.
    entity_infiltrated_at: float64 [B, V]: first time the entity is in an infiltration state
        (AS-18), +inf if never within the window's label horizon.
    entity_malicious_share: float32 [B, V, M]: malicious share of the entity's updates in the window
        trailing each trigger (stage-4 malignity target, build-spec §4b.4); NaN if no updates.
    family: tuple per window of attack-family names ("benign" for none); novelty per window for
        zero-shot evaluation ("known" | "novel" | "").
    label_horizon: float64 [B]: relative time up to which `entity_infiltrated_at` is observed; an
        entity with +inf is right-censored at this time (survival losses need it).
    """

    update_malicious: torch.Tensor
    update_stage: torch.Tensor
    update_technique: torch.Tensor
    entity_infiltrated_at: torch.Tensor
    entity_malicious_share: torch.Tensor
    family: tuple[str, ...] = ()
    novelty: tuple[str, ...] = ()
    label_horizon: torch.Tensor | None = None


# ================================================================================ outputs
@dataclass
class LatentOut:
    """CVG-AE posterior per position (flattened over B·P; padded positions hold zeros).

    z: float [B, P, dz] the sample (training) or mean (inference); mean, logvar: [B, P, Dc];
    logits: [B, P, G, C]; update_vec: [B, U, d_update] (FieldEncoder output, reused by the Decoder
    and explanations); field_weights: [B, U, pool_heads, C] attention-pooling weights (explanations).
    """

    z: torch.Tensor
    mean: torch.Tensor
    logvar: torch.Tensor
    logits: torch.Tensor
    update_vec: torch.Tensor
    field_weights: torch.Tensor | None = None


@dataclass
class LatentPrior:
    """TSTCT's transition prior for each position's *next* state of the same entity."""

    mean: torch.Tensor      # [B, P, Dc]
    logvar: torch.Tensor    # [B, P, Dc]
    logits: torch.Tensor    # [B, P, G, C]


@dataclass
class EnvironmentOut:
    """TSTCT output (build-spec §2.5).

    memory: [B, P, d] memory-stream output; kv: per block (K, V) each [B, H, P, d_h] — the
    Environment cache entries (keys already time-rotated on temporal/causal heads);
    refined: [B, P, d] thinking-stream output after R passes; prior: transition prior;
    causal_gate: [B, H_causal, P, P] gate values g_ij (zeros where not a candidate); passes: R used.
    attention: per block aux dicts when weights were requested (explanations).
    """

    memory: torch.Tensor
    kv: list[KV]
    refined: torch.Tensor
    prior: LatentPrior
    causal_gate: torch.Tensor
    passes: int
    attention: list[dict] = field(default_factory=list)


@dataclass
class AnalysisOut:
    """TAAFT output at the triggers of a window (build-spec §2.7).

    Tokens per trigger: V entity tokens (inactive ones masked) followed by G_adv adversary slots.
    context: [B, M, V + G, d] thinking-stream output; token_mask: [B, M, V + G];
    imagination_kv: per block (K, V) of the memory stream [B, H, M·(V+G), d_h] (written to Imagination);
    y0, y: float32 [B, M, V + G, d_y] initial and refined hypotheses (the descent runs in float32);
    energy_trace: float64 [S+1] mean E_total per step;
    lens_energy: name → float64 [B, M] per-lens energy at ŷ (D-42 decomposition; summed in float64);
    lens_share: name → float64 [B, M] share of the last descent step's displacement attributable to each lens;
    readouts: name → tensor. Float64 (D-54): compromise [B,M,V], stage [B,M,V,15], malignity [B,M,V+G],
        trust [B,M,V], slot_weight [B,M,G], goal [B,M,n_goals], type [B,M,n_types], token_energy [B,M,V+G],
        energy_rel/<lens> [B,M]. Float32: latent_* and next_latent_* mean/logvar/logits [B,M,V,…] (network
        quantities feeding the Decoder and latent likelihoods), lens aux maps, noise features, attention.
    context: float32. passes, descent_steps: run-time budgets used (D-44).
    """

    context: torch.Tensor
    token_mask: torch.Tensor
    imagination_kv: list[KV]
    y0: torch.Tensor
    y: torch.Tensor
    energy_trace: torch.Tensor
    lens_energy: dict[str, torch.Tensor]
    lens_share: dict[str, torch.Tensor]
    readouts: dict[str, torch.Tensor]
    passes: int
    descent_steps: int


@dataclass
class ForecastOut:
    """Forecaster output for each trigger (build-spec §2.8).

    Float64 (D-54): p_inf, p_inf_band, p_inf_median, stage, hazard, route_weight. Float32: step_state,
    step_value, step_reward, step_latent (learned network quantities).

    p_inf: [B, M, K] route-mixture infiltration curve (non-decreasing by construction);
    p_inf_band: [B, M, K, 2] 10 %–90 % band over routes; p_inf_median: [B, M, K];
    stage: [B, M, K, 15] route-mixture stage posterior per step; hazard: [B, M, N, K] per route;
    route_weight: [B, M, N] (duplicates merged: zero weight for merged copies);
    route_actions: long [B, M, N, K, 2] (technique slot, target token index);
    route_distinct: long [B, M] number of distinct routes (≤ N, D-46); mode_route: long [B, M];
    step_state: [B, M, N, K, d] imagined states; step_value, step_reward: [B, M, N, K];
    step_latent: imagined back-projected latent of the target entity [B, M, N, K, dz] (decoded and
    physics-checked by the Decoder); horizon_k, routes_n: budgets used.
    """

    p_inf: torch.Tensor
    p_inf_band: torch.Tensor
    p_inf_median: torch.Tensor
    stage: torch.Tensor
    hazard: torch.Tensor
    route_weight: torch.Tensor
    route_actions: torch.Tensor
    route_distinct: torch.Tensor
    mode_route: torch.Tensor
    step_state: torch.Tensor
    step_value: torch.Tensor
    step_reward: torch.Tensor
    step_latent: torch.Tensor
    horizon_k: int
    routes_n: int
