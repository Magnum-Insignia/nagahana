"""Per-component configuration dataclasses: the single source of configuration (D-57). Defaults are the L preset.

Every field is declared with `core.config.setting(default, source=..., doc=...)`: its default is the
build's value and its metadata names the decision (D-xx) or assumption (AS-xx) behind it. The YAML
files under conf/model/ are generated from these classes (`core.config.generate_conf`) and validated as
overrides against them (`core.config.from_mapping`). The `tiny` preset in `__init__.py` overrides widths
only.

Held decisions are not fields here: their options are configured once, in
`conf/decisions/decisions.yaml` (`governance.decisions.configure`). The integer parameters that an
option needs (for example the number of learned planes under the D-04 option "declared + learned") are
fields of the component they size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from nagahana.core.config import setting
from nagahana.core.errors import InvariantViolation
from nagahana.models.vocab import HYPEREDGE_KINDS, NODE_KINDS, PLANES


@dataclass(frozen=True)
class FieldEncoderConfig:
    """Input layer (build-spec section 2.1)."""

    n_slots: int = setting(128, source="P-22, AS-323", doc="field-slot table size; at least the number of columns")
    d_field: int = setting(128, source="build-spec section 4", doc="width of one field state")
    hash_rows: int = setting(65_536, source="AS-33", doc="hashed categorical rows (ports, protocols, codes)")
    n_frequencies: int = setting(16, source="AS-31", doc="periodic-embedding frequencies per scalar")
    freq_sigma: float = setting(1.0, source="AS-31", doc="initial frequency scale of the periodic embedding (Gorishniy et al., NeurIPS 2022)")
    max_bits: int = setting(16, source="AS-100", doc="bits of a bitmask field that get embeddings (TCP flags use 6 to 9)")
    pool_heads: int = setting(4, source="build-spec section 2.1", doc="attention-pooling heads from field states to the update vector")
    d_update: int = setting(256, source="build-spec section 4", doc="width of the update vector u_i")
    clock_features: bool = setting(False, source="D-50", doc="clock features are built and off when training on lab datasets")
    max_log_magnitude: float = setting(50.0, source="AS-100", doc="|signed_log1p(x)| is clipped here before the periodic embedding (e^50 is about 5e21)")


@dataclass(frozen=True)
class GraphConfig:
    """Graph builder (build-spec section 2.2)."""

    planes: tuple[str, ...] = setting(PLANES, source="AS-01", doc="declared relation planes, in branch order (D-04 held)")
    hyperedge_kinds: tuple[str, ...] = setting(HYPEREDGE_KINDS, source="AS-02", doc="hyperedge kinds of every plane")
    node_kinds: tuple[str, ...] = setting(NODE_KINDS, source="D-47", doc="entity kinds of the data model (vocab.NODE_KINDS)")
    max_nodes: int = setting(32, source="AS-102", doc="local subgraph cap per position (compute profile)")
    rwse_steps: int = setting(16, source="D-49, AS-103", doc="random-walk structural encoding length")
    fan_min_responders: int = setting(8, source="AS-02, AS-101", doc="a fan hyperedge needs at least this many distinct responders")
    fan_window_s: float = setting(60.0, source="AS-101", doc="within this trailing window in seconds; episodes split at larger gaps")
    hyperedge_ttl_s: float = setting(3600.0, source="AS-102", doc="time-based validity of a hyperedge in local subgraphs (never count-based)")
    max_hops: int = setting(2, source="AS-102", doc="local subgraph radius (the TSTCT spatial reach, hop <= 2)")
    learned_planes: int = setting(0, source="D-04, AS-01, AS-720", doc="learned relation planes over the connectivity plane; 0 under the working option 'declared'")

    def __post_init__(self) -> None:
        if not self.planes or len(set(self.planes)) != len(self.planes):
            raise InvariantViolation(f"planes must be non-empty and distinct, got {self.planes}")
        if self.max_nodes < 1 or self.rwse_steps < 1 or self.max_hops < 1:
            raise InvariantViolation("max_nodes, rwse_steps and max_hops must be >= 1")
        if self.fan_min_responders < 2:
            raise InvariantViolation("a fan joins at least two responders (fan_min_responders >= 2)")
        if not (self.fan_window_s > 0 and self.hyperedge_ttl_s > 0):
            raise InvariantViolation("fan_window_s and hyperedge_ttl_s must be positive")
        if self.learned_planes < 0:
            raise InvariantViolation("learned_planes must be >= 0")
        if self.learned_planes > 0 and "connectivity" not in self.planes:
            raise InvariantViolation("learned planes are formed over the connectivity plane, which must be declared")


@dataclass(frozen=True)
class CVGAEConfig:
    """CVG-AE (build-spec section 2.3)."""

    dim: int = setting(256, source="build-spec section 4", doc="branch width")
    layers: int = setting(4, source="A-04, build-spec section 4", doc="horizontal layers per plane branch")
    heads: int = setting(4, source="AS-03", doc="attention heads of the hyperedge-to-node aggregation")
    mlp_mult: int = setting(2, source="AS-108", doc="typed update MLP hidden width = mlp_mult x dim")
    cont_dim: int = setting(128, source="D-20, build-spec section 4", doc="Dc, continuous latent width (rates, entropies, timings)")
    disc_groups: int = setting(16, source="D-20, build-spec section 4", doc="G categorical latent groups (structure, stage)")
    disc_classes: int = setting(32, source="D-20, build-spec section 4", doc="C classes per categorical group")
    unimix: float = setting(0.01, source="AS-05", doc="uniform mixing of categorical posteriors (DreamerV3)")
    max_hops_bias: int = setting(4, source="D-49, AS-108", doc="hop-distance bias buckets")
    size_buckets: int = setting(8, source="AS-108", doc="hyperedge-size bias buckets, bucket floor(log2 |e|)")
    age_buckets: int = setting(32, source="AS-108", doc="learned log-time age buckets (1 ms to about 50 days); no decay imposed")
    age_frequencies: int = setting(8, source="AS-31, AS-108", doc="periodic frequencies of the log(1 + age) encoding")
    logvar_bound: float = setting(10.0, source="AS-108", doc="logvar = b tanh(raw / b), a smooth bound for numerical safety")


@dataclass(frozen=True)
class DecoderConfig:
    """Decoder (build-spec section 2.4)."""

    hidden: int = setting(512, source="build-spec section 4", doc="conditioning trunk width")
    n_service_classes: int = setting(64, source="AS-33, AS-106", doc="service-class buckets of categorical columns")
    edge_hidden: int = setting(512, source="build-spec section 4", doc="width of the candidate-hyperedge heads")
    role_dim: int = setting(32, source="AS-38", doc="role embedding width in the field head's conditioning")
    max_log_mean: float = setting(50.0, source="AS-110", doc="log-space means are clipped at this magnitude")
    log_scale_min: float = setting(-5.0, source="AS-110", doc="lower bound of the learned Gaussian log-scale")
    log_scale_max: float = setting(3.0, source="AS-110", doc="upper bound of the learned Gaussian log-scale")
    max_bits: int = setting(16, source="AS-110", doc="bits decoded per bitmask column (equals FieldEncoderConfig.max_bits)")
    edge_negatives: int = setting(1, source="AS-111", doc="member-swap negatives per observed hyperedge")


@dataclass(frozen=True)
class TSTCTConfig:
    """TSTCT (build-spec section 2.5)."""

    dim: int = setting(1024, source="build-spec section 4", doc="model width")
    blocks: int = setting(16, source="build-spec section 4", doc="pre-norm blocks")
    heads: int = setting(16, source="build-spec section 4", doc="attention heads per block")
    spatial_heads: int = setting(4, source="AS-09", doc="spatial heads")
    temporal_heads: int = setting(8, source="AS-09", doc="temporal heads")
    causal_heads: int = setting(4, source="AS-09", doc="causal heads")
    mlp_hidden: int = setting(2816, source="AS-32", doc="SwiGLU hidden width")
    temporal_window: int = setting(512, source="build-spec section 2.5", doc="last states of the same entity read by temporal heads")
    spatial_keys: int = setting(64, source="build-spec section 2.5", doc="neighbours' latest states read by spatial heads (hop <= 2)")
    causal_keys: int = setting(32, source="AS-10", doc="top-k causal candidates kept by the gate")
    causal_lag_s: float = setting(300.0, source="AS-10", doc="largest cause-to-effect lag considered, seconds")
    rotary_p_min: float = setting(1e-3, source="D-49", doc="shortest rotary period, seconds")
    rotary_p_max: float = setting(604_800.0, source="D-49", doc="longest rotary period, seconds (one week)")
    delta_buckets: int = setting(32, source="D-49", doc="log-delta-t bias buckets")
    train_passes_mean: float = setting(3.0, source="AS-07", doc="R = 1 + Poisson(mean) in training, clipped to max_passes")
    max_passes: int = setting(8, source="AS-07", doc="largest R drawn in training")
    grad_passes: int = setting(2, source="AS-07", doc="truncated backpropagation through the last passes")
    default_passes: int = setting(4, source="AS-08, D-44", doc="run-time default R, recorded with every forecast")
    delta0_s: float = setting(1e-3, source="D-49", doc="first width of the log-delta-t and log-age bias buckets, seconds")
    causal_candidate_cap: int = setting(256, source="AS-151", doc="most recent causal candidates scored by the gate per query")
    gate_hidden: int = setting(64, source="AS-152", doc="hidden width of the causal gate MLP per causal head")
    gate_bias_init: float = setting(2.0, source="AS-152", doc="initial gate logit (sigmoid(2) is about 0.88: gates start almost open)")
    dt_frequencies: int = setting(8, source="AS-153", doc="periodic frequencies of the delta-t encoding (gate and prior)")
    time_tie_s: float = setting(1e-7, source="AS-160", doc="times within 100 ns are simultaneous in as-of and strict-order tests")


@dataclass(frozen=True)
class MemoryConfig:
    """Environment store, long-term memory, Imagination store (build-spec section 2.6)."""

    slots_per_entity: int = setting(512, source="AS-11", doc="working Environment slots per entity")
    bucket_delta0: float = setting(1e-3, source="AS-11", doc="first log-time bucket width, seconds")
    n_buckets: int = setting(32, source="AS-11", doc="log-time buckets (1 ms to about 50 days)")
    longterm_dim: int = setting(1024, source="AS-11, build-spec section 4", doc="Titans-style long-term memory width")
    longterm_hidden: int = setting(2048, source="AS-155", doc="long-term memory MLP hidden width")
    imagination_triggers: int = setting(8, source="AS-13", doc="M_im: past triggers kept in Imagination")
    retention_alpha: float = setting(0.02, source="D-36, AS-11", doc="forgetting fraction alpha_k per trigger, a function of the trigger index only")
    bucket_slots: int = setting(7, source="AS-150", doc="slot quota per dyadic time cell; 7 x (2 x 32 + 2) + 1 = 463 <= 512")
    longterm_momentum: float = setting(0.9, source="AS-155", doc="eta, momentum of the long-term memory surprise")
    longterm_step: float = setting(0.05, source="AS-155", doc="theta, step size on the memory-loss gradient (mean over the trigger's pairs)")


@dataclass(frozen=True)
class TAAFTConfig:
    """TAAFT (build-spec section 2.7)."""

    dim: int = setting(1024, source="build-spec section 4", doc="model width")
    blocks: int = setting(34, source="build-spec section 4", doc="decoder-style blocks")
    heads: int = setting(16, source="build-spec section 4", doc="attention heads per block")
    mlp_hidden: int = setting(2816, source="AS-32", doc="SwiGLU hidden width")
    adversary_slots: int = setting(16, source="D-49, build-spec section 2.7", doc="G_adv adversary-hypothesis slots (learned slot embeddings)")
    d_hyp: int = setting(256, source="AS-14", doc="d_y, hypothesis width")
    own_states: int = setting(32, source="AS-13, AS-37", doc="Environment states of the entity itself read per trigger")
    neighbour_states: int = setting(64, source="AS-13, AS-37", doc="neighbours' latest states read per trigger")
    lens_hidden: int = setting(1024, source="AS-14", doc="hidden width of the lens energy networks")
    descent_steps: int = setting(8, source="D-44", doc="run-time descent steps S, recorded with every forecast")
    suspicion_floor: float = setting(0.01, source="AS-14", doc="assume-breach floor of the compromise readout, strictly in (0, 1)")
    lambda_phys: float = setting(0.1, source="AS-15, D-37", doc="weight of the shared physics term Phi_phys; TAAFT's E_total and every component loss read this one value")
    context_dropout: float = setting(0.1, source="AS-16", doc="probability of the null-context reading per training window (P-09 two readings)")
    n_goals: int = setting(8, source="P-13", doc="adversary goal classes")
    n_types: int = setting(3, source="P-13", doc="adversary types: opportunistic, targeted, insider")
    default_passes: int = setting(4, source="AS-08, D-44", doc="run-time default R, recorded with every forecast")
    lenses: tuple[str, ...] = setting(("belief-trust", "game", "information", "topology", "temporal", "causal"),
                                      source="D-42, AS-14",
                                      doc="lens terms of E_total; physics is always added with lambda_phys; "
                                          "'mechanism-design' and 'noise' are own terms only under their D-24 and D-26 options")
    imagination_triggers: int = setting(8, source="AS-13, AS-217", doc="M_im read by the belief recursion; equals MemoryConfig.imagination_triggers")
    self_rotary_fraction: float = setting(0.5, source="AS-200", doc="share of self-attention heads rotated by trigger time")
    grad_passes: int = setting(2, source="AS-07", doc="truncated backpropagation through the last thinking passes")
    cross_layout: str = setting("gathered", source="AS-13, AS-201", doc="gathered (inference layout) or dense (masked, same function)")
    delta_buckets: int = setting(32, source="D-49", doc="log-delta-t buckets for trigger gaps, key ages and entity ages")
    descent_step_init: float = setting(0.1, source="AS-16", doc="initial descent step alpha, learned through softplus")
    descent_noise: float = setting(0.01, source="AS-212", doc="sigma_0 of the annealed descent noise sigma_i = sigma_0 (1 - i / S)")
    lens_rank_divisor: int = setting(4, source="AS-219", doc="pairwise and game subspaces have rank d_hyp // divisor")
    noise_frequencies: int = setting(64, source="AS-205", doc="periodogram frequency grid size")
    noise_period_min: float = setting(1.0, source="AS-205", doc="shortest period on the grid, seconds")
    noise_period_max: float = setting(3600.0, source="AS-205", doc="longest period on the grid, seconds")
    noise_scales: int = setting(8, source="AS-205", doc="Hurst block sizes 1, 2, 4, ..., 2^(scales - 1)")
    noise_min_events: int = setting(8, source="AS-205", doc="events needed before a periodogram counts as evidence")
    noise_min_blocks: int = setting(4, source="AS-205", doc="blocks needed at a scale before its variance enters the Hurst fit")
    reference_momentum: float = setting(0.99, source="AS-210", doc="EMA momentum of each lens's reference energy")
    memory_probes: int = setting(32, source="AS-220", doc="learned probe queries read from the long-term memory as extra cross keys")


@dataclass(frozen=True)
class ForecasterConfig:
    """Forecaster (build-spec section 2.8)."""

    dim: int = setting(1024, source="build-spec section 4", doc="model width")
    blocks: int = setting(8, source="build-spec section 4", doc="pre-norm blocks")
    heads: int = setting(16, source="build-spec section 4", doc="attention heads per block")
    mlp_hidden: int = setting(2816, source="AS-32", doc="SwiGLU hidden width")
    n_techniques: int = setting(700, source="AS-20", doc="ATT&CK technique action slots")
    context_entities: int = setting(48, source="build-spec section 2.8", doc="top entities by compromise in the context (plus the adversary slots)")
    window_seconds: float = setting(60.0, source="AS-12", doc="one imagined step, and the trigger cadence")
    horizon_k: int = setting(12, source="D-30", doc="default K, set at run time")
    routes_n: int = setting(200, source="D-46", doc="default N, the maximum number of routes")
    mppi_top_b: int = setting(8, source="AS-21", doc="candidate actions re-weighted per step")
    mppi_temperature: float = setting(1.0, source="AS-21", doc="MPPI softmax temperature")
    gamma: float = setting(0.97, source="AS-253", doc="discount of the adversary's return")
    infiltration_bonus: float = setting(1.0, source="AS-17", doc="beta of the adversary reward")
    exposure_cost: float = setting(0.1, source="AS-17", doc="kappa of the adversary reward")
    td_lambda: float = setting(0.95, source="AS-253", doc="lambda of the TD(lambda) value targets")
    route_estimator: str = setting("monte_carlo", source="AS-251", doc="monte_carlo (count / N) or probability (mode-seeking)")
    band_quantiles: tuple[float, float] = setting((0.1, 0.9), source="build-spec section 4b.8", doc="band of the infiltration curve over routes")
    future_stride: int = setting(1, source="AS-12", doc="triggers per imagined step in teacher forcing")
    rotary_p_min: float = setting(1.0, source="D-49", doc="shortest rotary period for imagined-step time t = k x window_seconds")
    rotary_p_max: float = setting(604_800.0, source="D-49", doc="longest rotary period (one week)")
    route_chunk: int = setting(0, source="build-spec section 2.8", doc="routes imagined per chunk (0 = all at once); a memory bound with identical results")


@dataclass(frozen=True)
class AdvisorConfig:
    """Advisor (build-spec section 2.9)."""

    dim: int = setting(1024, source="build-spec section 4", doc="model width")
    blocks: int = setting(4, source="build-spec section 4", doc="decoder-style blocks")
    heads: int = setting(16, source="build-spec section 4", doc="attention heads per block")
    mlp_hidden: int = setting(2816, source="AS-32", doc="SwiGLU hidden width")
    n_actions: int = setting(256, source="build-spec section 2.9", doc="D3FEND action slots")
    beam_width: int = setting(64, source="build-spec section 2.9", doc="beam width W")
    max_steps: int = setting(3, source="build-spec section 2.9", doc="counter-sequence length")
    rollouts: int = setting(50, source="build-spec section 2.9", doc="re-imagined routes per candidate")
    cvar_alpha: float = setting(0.2, source="AS-24", doc="tail share of the CVaR ranking")
    cost_weight: float = setting(0.1, source="AS-23", doc="kappa in -delta P_inf - kappa x cost")
    criticality: tuple[float, ...] = setting((1.0, 1.5, 1.0, 5.0, 0.2, 3.0, 1.5, 2.0), source="AS-23, AS-254",
                                             doc="criticality per entity kind in vocab.NODE_KINDS order; OT highest (D-34)")
    physics_tau: float = setting(1e-3, source="AS-28", doc="feasibility gate Phi_phys <= tau on predicted effects")
    improvement_temperature: float = setting(0.1, source="AS-257", doc="eta of the policy-improvement target q proportional to pi exp(J / eta)")
    train_candidates: int = setting(8, source="AS-257", doc="candidates evaluated per trigger in the policy-improvement loss")


@dataclass(frozen=True)
class VerifierConfig:
    """Verifier (build-spec section 2.10)."""

    dim: int = setting(1024, source="build-spec section 4", doc="model width")
    blocks: int = setting(4, source="build-spec section 4", doc="pre-norm blocks of the process-reward model")
    heads: int = setting(16, source="build-spec section 4", doc="attention heads per block")
    mlp_hidden: int = setting(2816, source="AS-32", doc="SwiGLU hidden width")
    n_output_families: int = setting(4, source="D-45, AS-25", doc="temperatures proposed for P_inf, stage, compromise and advice")
    reliability_bins: int = setting(10, source="AS-25", doc="reliability-diagram bins")
    min_pairs: int = setting(999, source="AS-25, AS-721", doc="scored pairs needed before a calibration is proposed; at least the split-conformal minimum for conformal_alpha")
    drift_window: int = setting(12, source="AS-259", doc="systematic gap over the last resolutions")
    cusum_k: float = setting(0.05, source="AS-25", doc="CUSUM reference value (drift allowance)")
    cusum_h: float = setting(5.0, source="AS-25", doc="CUSUM decision threshold")
    conformal_alpha: float = setting(0.001, source="AS-25", doc="target false-positive rate of split-conformal alert thresholds")
    temperature_min: float = setting(0.25, source="AS-25", doc="lower end of the ML temperature search interval")
    temperature_max: float = setting(4.0, source="AS-25", doc="upper end of the ML temperature search interval")
    ph_delta: float = setting(0.1, source="AS-258", doc="Page-Hinkley allowance on the standardised latent deviation")
    ph_lambda: float = setting(10.0, source="AS-258", doc="Page-Hinkley alarm threshold")
    ph_warmup: int = setting(30, source="AS-258", doc="observations before Welford statistics are trusted for alarms")
    gap_threshold: float = setting(0.2, source="AS-259", doc="systematic-gap alert on |sum(f - y)| / n over the drift window")
    context_entities: int = setting(48, source="build-spec section 2.8", doc="PRM context: top entities by compromise, as for the Forecaster")

    def __post_init__(self) -> None:
        if not 0.0 < self.conformal_alpha < 1.0:
            raise InvariantViolation("conformal_alpha must lie in (0, 1)")
        # The split-conformal threshold is the ceil((n + 1)(1 - alpha))-th smallest score; it is finite only
        # when that rank is <= n (Angelopoulos and Bates, arXiv:2107.07511). The same arithmetic as
        # `models.verifier.calibration.conformal_threshold`, so the check and the computation agree.
        if math.ceil((self.min_pairs + 1) * (1.0 - self.conformal_alpha)) > self.min_pairs:
            need = math.ceil((1.0 - self.conformal_alpha) / self.conformal_alpha)
            raise InvariantViolation(
                f"min_pairs = {self.min_pairs} cannot give a finite split-conformal threshold at conformal_alpha = "
                f"{self.conformal_alpha}; it needs at least {need} pairs"
            )
        if not 0.0 < self.temperature_min < self.temperature_max:
            raise InvariantViolation("the temperature interval must satisfy 0 < temperature_min < temperature_max")


@dataclass(frozen=True)
class GeneratorConfig:
    """Generator, training only (build-spec section 2.11). Not part of the model's parameter count."""

    dim: int = setting(512, source="build-spec section 2.11", doc="width of the masked-generative transformer")
    blocks: int = setting(8, source="build-spec section 2.11", doc="blocks of the masked-generative transformer")
    heads: int = setting(8, source="build-spec section 2.11", doc="attention heads per block")
    mlp_hidden: int = setting(1408, source="AS-32", doc="SwiGLU hidden width")
    diffusion_steps: int = setting(100, source="AS-365", doc="diffusion steps T")
    denoiser_hidden: int = setting(1024, source="AS-365", doc="hidden width of the TabDDPM-style denoiser")
    physics_tau: float = setting(1e-3, source="AS-28", doc="P-11 gate Phi_phys <= tau")
    attack_share: float = setting(0.7, source="AS-370", doc="share of variants drawn from attack events")
    value_bins: int = setting(256, source="AS-362", doc="bins per numeric column in signed-log1p space")
    cat_vocab: int = setting(1024, source="AS-362", doc="classes per categorical or bitmask column, including out-of-vocabulary")
    max_records: int = setting(16, source="AS-364", doc="records per sequence seen by the masked model")
    unmask_steps: int = setting(8, source="AS-363", doc="MaskGIT iterative-decoding steps")
    choice_temperature: float = setting(1.0, source="AS-363", doc="Gumbel noise on confidences, annealed to 0")
    ar_fraction: float = setting(0.5, source="AS-364", doc="share of left-to-right (autoregressive) training examples")
    denoiser_blocks: int = setting(4, source="AS-365", doc="residual SwiGLU blocks of the denoiser")
    cosine_s: float = setting(0.008, source="AS-365", doc="cosine-schedule offset s (Nichol and Dhariwal 2021)")
    regen_fraction: float = setting(0.3, source="AS-366", doc="share of an update's contributing cells regenerated")
    sampling_rates: tuple[int, ...] = setting((8, 32, 128), source="AS-352", doc="1-in-n packet-sampling rates drawn from")
    sensor_hide_fraction: float = setting(0.2, source="AS-353", doc="share of internal hosts in the hidden segment")
    jitter_rel: float = setting(0.1, source="AS-355", doc="timing scale s in [1 / (1 + eps), 1 + eps]")
    rate_scale: tuple[float, float] = setting((0.5, 2.0), source="AS-356", doc="time-dilation factor range of attack activity")
    edge_dropout: float = setting(0.1, source="AS-358", doc="share of benign-only relations dropped")
    rewire_rate: float = setting(0.1, source="AS-358", doc="share of benign-only relations rewired")
    physics_weight: float = setting(1.0, source="AS-360", doc="w_c of every residual in the gate's Phi_phys")
    energy_quantiles: tuple[float, float] = setting((0.01, 0.99), source="AS-361", doc="accepted quantile range of real-data energies")
    mtu: float | None = setting(None, source="physics/residuals.py MTUBound", doc="site MTU in bytes; required where a wire-capture size bound is used, never defaulted")
    link_bps: float | None = setting(None, source="physics/residuals.py LinkCapacityBound", doc="site line rate in bit/s; null means the link-capacity bound is not checked")
    max_attempts: int = setting(3, source="AS-27", doc="attempts per requested variant (rejections are reported)")


@dataclass(frozen=True)
class TrainingConfig:
    """Data windows and optimisation (build-spec section 3)."""

    window_updates: int = setting(1024, source="AS-317, build-spec section 4", doc="updates per training window (2,048 positions)")
    batch_windows: int = setting(8, source="build-spec section 3", doc="windows per micro-batch")
    max_entities: int = setting(4096, source="AS-317", doc="entities per window (working context)")
    mask_ratio: float = setting(0.15, source="AS-113", doc="stage-3 field masking probability")
    free_bits: float = setting(1.0, source="AS-05", doc="free bits of the KL terms, nats")
    beta_dyn: float = setting(0.5, source="AS-05", doc="dynamics weight of the balanced KL")
    beta_rep: float = setting(0.1, source="AS-05", doc="representation weight of the balanced KL")
    lambda_edge: float = setting(0.5, source="build-spec section 3", doc="weight of the candidate-hyperedge loss")
    lambda_gate: float = setting(1e-3, source="AS-10", doc="L1 weight on the causal gates")
    lr: float = setting(3e-4, source="AS-406", doc="AdamW learning rate after warm-up")
    weight_decay: float = setting(0.1, source="AS-406", doc="AdamW weight decay on matrices")
    warmup_steps: int = setting(2000, source="AS-406", doc="linear warm-up steps")
    grad_clip: float = setting(1.0, source="AS-406", doc="global gradient-norm clip")
    precision: str = setting("bf16", source="AS-39", doc="matrix compute precision with fp32 master weights")
    splits_full: tuple[float, float, float] = setting((0.6, 0.2, 0.2), source="AS-326", doc="train / test / validation shares of full training")
    splits_pretrain: tuple[float, float] = setting((0.7, 0.3), source="AS-326", doc="train / validation shares of pretraining")
    lora_rank: int = setting(16, source="AS-26", doc="rank of the site calibration adapters")
    lora_alpha: float = setting(32.0, source="AS-26", doc="scale of the site calibration adapters")


@dataclass(frozen=True)
class SiteConfig:
    """The monitored site, which sizes the working memory of the one L model (D-63; `lab/sizing.py`).

    The defaults are the critical-infrastructure point of the published compute profile (`lab/compute.py`), so
    the published memory figures are reproduced by the budget of the default site.
    """

    monitored_hosts: int = setting(4096, source="D-63, AS-727", doc="internal machines the site monitors")
    entity_ratio: float = setting(1.0, source="AS-727", doc="entities in the working context per monitored host")
    state_rate: float = setting(18_400.0, source="D-63, AS-727", doc="state updates per second at the site (sustained)")
    retention_s: float = setting(2_419_200.0, source="D-15 working option (AS-11), AS-727",
                                 doc="seconds of event log the site keeps (four weeks); the Environment is rebuilt from it")
    desktop_accelerator_bytes: float = setting(24e9, source="AS-727", doc="memory of one desktop accelerator")
    workstation_accelerator_bytes: float = setting(48e9, source="AS-727", doc="memory of one workstation accelerator")
    accelerator_bytes: float = setting(80e9, source="D-54, AS-727", doc="memory of one data-centre accelerator")

    def __post_init__(self) -> None:
        if self.monitored_hosts < 1:
            raise InvariantViolation("a site monitors at least one host")
        if not (self.entity_ratio > 0 and math.isfinite(self.entity_ratio)):
            raise InvariantViolation("entity_ratio must be positive and finite")
        if not (self.state_rate >= 0 and math.isfinite(self.state_rate)):
            raise InvariantViolation("state_rate must be >= 0 and finite")
        if not (self.retention_s > 0 and math.isfinite(self.retention_s)):
            raise InvariantViolation("retention_s must be positive and finite")
        sizes = (self.desktop_accelerator_bytes, self.workstation_accelerator_bytes, self.accelerator_bytes)
        if not all(s > 0 for s in sizes) or list(sizes) != sorted(sizes):
            raise InvariantViolation("accelerator memories must be positive and ordered desktop <= workstation <= data centre")
