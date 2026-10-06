"""NagaHana: the composed world model (build-spec §1, §2; integration, engineer G).

Purpose
-------
One `nn.Module` that owns every learned component of NagaHana except the training-only Generator
(D-40), wires them in the one-way order of the processing story (build-spec §1, D-35, D-43) and
exposes the stage forwards the trainers and the inference engine call:

    perceive:  FieldBatch ─FieldEncoder→ u ─CVG-AE(as-of local graphs)→ z ─TSTCT(+carry)→ Environment
    analyse:   Environment + long-term memory keys + carried Imagination ─TAAFT(R, S)→ Imagination, ŷ
    forecast:  ŷ, context ─Forecaster(K steps, N routes)→ P_inf(k), stages, routes
    (Advisor on demand, Verifier over forecasts, Decoder over any latent)

Owner sources, decisions, assumptions
-------------------------------------
- [A-10] … [A-21] (component roles), D-19 (names), D-35 (memory access: beliefs never flow back
  into the Environment; TAAFT only reads it), D-36 (retention per trigger), D-40 (Generator excluded),
  D-43/D-44 (run-time R, S, K, N), D-49 (time, never index), D-51 (training across windows).
- AS-05 (posterior sample in training, mean at inference), AS-11 (long-term memory form, α),
  AS-13 (TAAFT reads TSTCT K/V), AS-15 (λ_phys on normalised residuals), AS-22 (STAGED coupling).
- AS-220 … AS-224 (TAAFT: long-term memory as extra cross keys, carried Imagination across calls).
- New (docs/assumptions/integration.md): AS-401 (what the long-term memory stores), AS-402 (physics
  term of training: residual set and weights), AS-403 (TAAFT reads only the current window's causal
  gates). AS-400 (the earlier token-initialisation read) is superseded by AS-220.

Maths of the integration pieces (everything else lives in the components)
--------------------------------------------------------------------------
Long-term memory (Titans form, `memory/longterm.py`), written once per trigger τ_k with the
memory-stream states of the entities active at τ_k (observations, never beliefs; AS-401):

    x_{k,v} = m_{latest_k(v)} ∈ ℝ^{d_TSTCT},     M_k = (1 − α_k) M_{k−1} + S_k   (α_k = α, AS-11)

and read by TAAFT as extra cross-attention keys (AS-220): TAAFT's X learned probes query the memory,
r_x = M(normalise(W_Q p_x)), and K_mem, V_mem = maps of r enter every token's cross-attention. The
state read at trigger τ_m holds only writes of triggers strictly before τ_m (AS-222): `analyse`
accepts one state for the call or one state per trigger (`longterm=[M_{<τ_0}, M_{<τ_1}, …]`).

Imagination across calls (AS-223): `analyse(past=…)` passes the carried memory-stream K/V and ŷ of
earlier triggers (`models/taaft/imagination.py`), so the belief recursion spans windows (training)
and live triggers (inference) exactly as one long call would.

Physics (D-37, AS-15, AS-402): Φ_phys over the unconditional flow residuals, each normalised by its
bound (`physics/normalise.RelativeResidual`), equal weights w_c = 1, with the site MTU required from
`GeneratorConfig.mtu` (never defaulted: a wrong MTU teaches a false boundary).

Parameter count: `count_parameters(cfg)` builds the model on the meta device (no memory) and reports
every component and the total (build-spec §4, §5 "the L preset is built on the meta device").

Invariants (tests/test_integration_model.py)
--------------------------------------------
- The L preset builds on the meta device; its total is ≥ 1.0e9 (TAAFT at 34 blocks, owner's scaling).
- TAAFT's M_im equals the Imagination store's (AS-217): the constructor refuses a mismatch.
- `taaft_view` never changes the Environment it reads (a new `EnvironmentOut` is returned).
- Gradients reach every component trained by a stage (tests per stage).

Extension points
----------------
- `taaft_view` is the one place where TAAFT's input view of the Environment is formed (carry
  slicing); a TAAFT cause lens that reads carried gates natively replaces it.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence

import torch
from torch import nn

from nagahana.core.errors import ConfigMissing, InvariantViolation
from nagahana.governance.assumptions import assume
from nagahana.memory.longterm import LongTermMemory, NeuralMemoryState
from nagahana.memory.retention import ConstantRetention
from nagahana.models.advisor.model import Advisor
from nagahana.models.batch import AnalysisOut, EnvironmentOut, FieldBatch, ForecastOut, LatentOut, WindowBatch
from nagahana.models.config import NagaHanaConfig
from nagahana.models.cvgae.attn import CVGAE
from nagahana.models.decoder.model import Decoder
from nagahana.models.forecaster.exposure import exposure_from_taaft
from nagahana.models.forecaster.model import Forecaster, ForecastIntervention, MarginalEnergy
from nagahana.models.inputs.encoder import FieldEncoder
from nagahana.models.taaft.imagination import PastImagination
from nagahana.models.taaft.model import TAAFT
from nagahana.models.taaft.structure import gather_rows
from nagahana.models.tstct.model import TSTCT, CarriedEnvironment
from nagahana.models.verifier.model import VerifierNet
from nagahana.models.vocab import N_STAGES
from nagahana.nn.blocks import KV
from nagahana.physics.normalise import RelativeResidual
from nagahana.physics.residuals import (
    CountWithinPackets,
    FlagCountBound,
    IATMaxBound,
    IATMeanBound,
    IATVarianceBound,
    MTUBound,
    Residual,
)
from nagahana.physics.term import PhysicsTerm, Target

#: Components counted by `count_parameters`, in processing order (the Generator is excluded, D-40).
COMPONENTS: tuple[str, ...] = ("inputs", "cvgae", "decoder", "tstct", "longterm", "taaft", "forecaster", "advisor",
                               "verifier")

#: AS-402: weight w_c of every (relative) physics residual in the training term.
PHYSICS_RESIDUAL_WEIGHT = 1.0


# ===================================================================================== physics
def physics_term(cfg: NagaHanaConfig, *, weight: float = PHYSICS_RESIDUAL_WEIGHT) -> PhysicsTerm:
    """Φ_phys for model outputs (D-37, AS-15, AS-402) with the site MTU from config (required).

    Residuals: flag counts ≤ packets (6 flags), bytes ≤ packets·MTU per direction, iat_max ≤ duration,
    iat_mean ≤ duration, iat_var ≤ iat_max²/2, DF/MF/retransmitted counts ≤ packets — each relative to
    its bound (`RelativeResidual`). `MinHeaderBound` is left out: its site value has no config field.
    """
    assume("AS-15", by=__name__)
    assume("AS-402", by=__name__)
    if cfg.generator.mtu is None:
        raise ConfigMissing("GeneratorConfig.mtu (the site MTU) is required for the physics term; it is never defaulted.")
    base: list[Residual] = [FlagCountBound(f) for f in ("syn", "ack", "fin", "rst", "psh", "urg")]
    base += [MTUBound("fwd", float(cfg.generator.mtu)), MTUBound("bwd", float(cfg.generator.mtu))]
    base += [IATMaxBound(), IATMeanBound(), IATVarianceBound()]
    base += [CountWithinPackets(f) for f in ("pkt.ip_df_count", "pkt.ip_mf_count", "pkt.retransmissions")]
    rel: list[Residual] = [RelativeResidual(r) for r in base]
    return PhysicsTerm(rel, {r.name: float(weight) for r in rel}, target=Target.MODEL_OUTPUT)


# ===================================================================================== identity
def latent_space_id(cfg: NagaHanaConfig) -> str:
    """The latent-space version tag plus its shape: '<latent_space>|Dc=…|G=…|C=…|unimix=…' (P-19)."""
    c = cfg.cvgae
    return f"{cfg.latent_space}|Dc={c.cont_dim}|G={c.disc_groups}|C={c.disc_classes}|unimix={c.unimix:g}"


def latent_space_hash(cfg: NagaHanaConfig) -> str:
    """SHA-256 (first 16 hex digits) of `latent_space_id`: caches and checkpoints carry it (P-18, P-19)."""
    return hashlib.sha256(latent_space_id(cfg).encode("utf-8")).hexdigest()[:16]


def model_hash(module: nn.Module) -> str:
    """SHA-256 over the module's state dict (names, dtypes, shapes, bytes): the cache key of P-18."""
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        x = t.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(x.dtype).encode())
        h.update(str(tuple(x.shape)).encode())
        if x.numel():
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ===================================================================================== the model
class NagaHana(nn.Module):
    """The composed model. See the module docstring.

    Parameters
    ----------
    cfg: `NagaHanaConfig` (`preset("L")` is the model; `preset("tiny")` a test fixture).
    column_names, column_kind: the Decoder's column layout (names and `vocab.COLUMN_KINDS` codes).
        None = the canonical layout of the data pipeline (`data.windows.COLUMN_SLOTS`, P-22 / AS-323),
        which is what real windows carry. Synthetic test batches pass their own layout.
    """

    def __init__(self, cfg: NagaHanaConfig, *, column_names: Sequence[str] | None = None,
                 column_kind: Sequence[int] | None = None) -> None:
        super().__init__()
        assume("AS-13", by=__name__)
        if cfg.taaft.imagination_triggers != cfg.memory.imagination_triggers:
            raise InvariantViolation("TAAFTConfig.imagination_triggers must equal MemoryConfig.imagination_triggers (AS-217)")
        if (column_names is None) != (column_kind is None):
            raise InvariantViolation("pass both column_names and column_kind, or neither")
        if column_names is None:
            from nagahana.data.windows import CANONICAL_KIND_CODES, COLUMN_SLOTS

            column_names, column_kind = COLUMN_SLOTS, CANONICAL_KIND_CODES
        assert column_kind is not None
        self.cfg = cfg
        n_planes = len(cfg.graph.planes)
        dz = cfg.latent_dim
        # ---- Simulator: input layer, CVG-AE, TSTCT (perceptors, [A-19]) and the Decoder (parallel output)
        self.inputs = FieldEncoder(cfg.inputs)
        self.cvgae = CVGAE(cfg.cvgae, cfg.graph, cfg.inputs.d_update)
        self.decoder = Decoder(cfg.decoder, latent_dim=dz, column_kind=column_kind, column_names=column_names,
                               n_planes=n_planes, latent_space=cfg.latent_space, plane_names=cfg.graph.planes)
        self.tstct = TSTCT.from_config(cfg)
        # ---- long-term memory (slow weights; the fast memory is data, `NeuralMemoryState`) + its read map
        self.longterm = LongTermMemory(cfg.memory, input_dim=cfg.tstct.dim)
        # ---- Forecaster role: TAAFT (Imagination) and the adversary's policy/value world model
        self.taaft = TAAFT(cfg.taaft, tstct=cfg.tstct, latent_dim=dz, n_planes=n_planes, n_stages=N_STAGES,
                           latent_split=(cfg.cvgae.cont_dim, cfg.cvgae.disc_groups, cfg.cvgae.disc_classes),
                           memory_input_dim=cfg.tstct.dim, memory_dim=cfg.memory.longterm_dim)
        self.forecaster = Forecaster(cfg.forecaster, d_context=cfg.taaft.dim, d_hyp=cfg.taaft.d_hyp, latent_dim=dz,
                                     n_stages=N_STAGES)
        # ---- Advisor (memory-less, advisory only) and Verifier (learned parts; human-gated, D-21)
        self.advisor = Advisor(cfg.advisor, d_context=cfg.taaft.dim, latent_dim=dz, n_stages=N_STAGES)
        self.verifier = VerifierNet(cfg.verifier, d_context=cfg.taaft.dim, d_hyp=cfg.taaft.d_hyp,
                                    d_state=cfg.forecaster.dim, n_techniques=cfg.forecaster.n_techniques,
                                    n_stages=N_STAGES, window_seconds=cfg.forecaster.window_seconds)
        #: Retention of the long-term memory: α_k = α for every trigger (D-36 form, value AS-11).
        self.retention = ConstantRetention(cfg.memory.retention_alpha)

    # ------------------------------------------------------------------ groups of modules
    def perceptors(self) -> list[nn.Module]:
        """CVG-AE, Decoder, TSTCT and the input layer: trained in stage 3, frozen in stage 4 (D-22)."""
        return [self.inputs, self.cvgae, self.decoder, self.tstct]

    def analyser(self) -> list[nn.Module]:
        """TAAFT and the long-term memory it reads (AS-220): trained from stage 4 on."""
        return [self.taaft, self.longterm]

    def component(self, name: str) -> list[nn.Module]:
        """Modules of a component name of `COMPONENTS` (TAAFT's memory probes and maps count under taaft)."""
        if name not in COMPONENTS:
            raise KeyError(f"unknown component {name!r}; known: {COMPONENTS}")
        return [getattr(self, name)]

    # ------------------------------------------------------------------ stage forwards
    def encode_updates(self, fields: FieldBatch, window: WindowBatch, *,
                       states: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """FieldEncoder: (u [B, U, d_update], field weights [B, U, H_pool, C]); `states` for attributions."""
        return self.inputs(fields, origin=window.origin, update_time=window.update_time, states=states)

    def latents(self, window: WindowBatch, update_vec: torch.Tensor, *, sample: bool,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """CVG-AE on the as-of local graphs: (z, mean, logvar, logits), each [B, P, …] (logits raw)."""
        b, u, d = update_vec.shape
        p = window.positions.entity.shape[1]
        z, mean, logvar, logits = self.cvgae(window.graph, update_vec.reshape(b * u, d), b * p, sample=sample,
                                             generator=generator)
        g, c = self.cfg.cvgae.disc_groups, self.cfg.cvgae.disc_classes
        return z.view(b, p, -1), mean.view(b, p, -1), logvar.view(b, p, -1), logits.view(b, p, g, c)

    def perceive(
        self,
        window: WindowBatch,
        *,
        sample: bool,
        passes: int,
        carry: CarriedEnvironment | None = None,
        grad_passes: int | None = None,
        generator: torch.Generator | None = None,
        fields: FieldBatch | None = None,
        need_weights: bool = False,
    ) -> tuple[LatentOut, EnvironmentOut]:
        """Perception: (LatentOut, EnvironmentOut) for B windows (build-spec §2.1–§2.5).

        sample: posterior sample (training) or posterior mean (inference, AS-05). passes: TSTCT R.
        carry: the earlier windows' Environment, aligned to `window` (D-51). fields: a perturbed copy
        of `window.fields` (stage-3 masking); the window's structure is unchanged.
        """
        f = window.fields if fields is None else fields
        u, w = self.encode_updates(f, window)                                    # [B, U, d_u], [B, U, H, C]
        z, mean, logvar, logits = self.latents(window, u, sample=sample, generator=generator)
        lat = LatentOut(z=z, mean=mean, logvar=logvar, logits=logits, update_vec=u, field_weights=w)
        env = self.tstct(z, window, passes=passes, grad_passes=grad_passes, need_weights=need_weights, carry=carry)
        return lat, env

    def taaft_view(self, env: EnvironmentOut, window: WindowBatch) -> EnvironmentOut:
        """TAAFT's input view of the Environment (a new object; `env` is untouched).

        With a carry, TSTCT returns causal gates [B, H_c, P, C + P] (carried candidates first); TAAFT's
        cause lens aggregates gates over window positions, so only the current part [..., −P:] is
        passed (AS-403: carried causes are not aggregated by TAAFT's cause lens).
        """
        p = window.positions.entity.shape[1]
        return dataclasses.replace(env, causal_gate=env.causal_gate[..., -p:])

    def memory_kv(self, longterm: NeuralMemoryState | Sequence[NeuralMemoryState]) -> KV:
        """TAAFT's long-term memory keys (AS-220): one state for the call → [B, H, X, d_h]; one state per
        trigger → [B, M, H, X, d_h]. Each state must hold only writes of earlier triggers (AS-222)."""
        if isinstance(longterm, NeuralMemoryState):
            return self.taaft.read_longterm(self.longterm, longterm)
        return self.taaft.read_longterm_per_trigger(self.longterm, list(longterm))

    def analyse(
        self,
        env: EnvironmentOut,
        window: WindowBatch,
        *,
        passes: int,
        descent_steps: int,
        longterm: NeuralMemoryState | Sequence[NeuralMemoryState] | None = None,
        past: PastImagination | None = None,
        physics: PhysicsTerm | None = None,
        create_graph: bool = False,
        need_weights: bool = False,
        generator: torch.Generator | None = None,
        drop_context: bool = False,
        hidden_entities: torch.Tensor | None = None,
    ) -> AnalysisOut:
        """TAAFT at the window's triggers (build-spec §2.7).

        longterm: the long-term memory to read (AS-220, AS-222), one state or one per trigger; None = no read.
        past: carried Imagination of earlier calls (AS-223), aligned to this call's tokens; None = none.
        physics: Φ_phys on decoded beliefs (D-37).
        """
        view = self.taaft_view(env, window)
        mem = self.memory_kv(longterm) if longterm is not None else None
        extra = past.kwargs() if past is not None else {}
        return self.taaft(view, window, passes=passes, descent_steps=descent_steps, physics=physics,
                          decoder=self.decoder if physics is not None else None, create_graph=create_graph,
                          need_weights=need_weights, generator=generator, drop_context=drop_context,
                          hidden_entities=hidden_entities, memory_kv=mem, **extra)

    def exposure(self) -> MarginalEnergy:
        """E(∅, ŷ) of one imagined hypothesis (AS-17, AS-252): the Forecaster's exposure callable."""
        return exposure_from_taaft(self.taaft)

    def forecast(self, analysis: AnalysisOut, *, horizon_k: int, routes_n: int, generator: torch.Generator | None = None,
                 with_exposure: bool = True, intervention: ForecastIntervention | None = None,
                 temperature: float | None = None) -> ForecastOut:
        """Imagine `routes_n` routes of `horizon_k` steps at every trigger (build-spec §2.8; no gradient)."""
        return self.forecaster.imagine(analysis, horizon_k=horizon_k, routes_n=routes_n, generator=generator,
                                       exposure=self.exposure() if with_exposure else None, intervention=intervention,
                                       temperature=temperature)

    # ------------------------------------------------------------------ long-term memory (AS-220, AS-222, AS-401)
    def longterm_inputs(self, env: EnvironmentOut, window: WindowBatch, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """What trigger m writes: memory-stream states at each active entity's latest position ≤ τ_m.

        Returns (x [B, V, d_TSTCT], mask [B, V]). Observations only (AS-401, D-35): beliefs never enter.
        """
        latest = window.triggers.entity_latest[:, m]                              # [B, V]
        mask = (latest >= 0) & window.entity_mask & window.triggers.mask[:, m, None]
        x = gather_rows(env.memory, latest)                                       # [B, V, d]
        return x * mask[..., None].to(x.dtype), mask

    def longterm_init(self, batch: int = 1) -> NeuralMemoryState:
        """A fresh long-term memory M₀ (one per stream or site)."""
        return self.longterm.init_state(batch)

    def longterm_write(self, state: NeuralMemoryState, x: torch.Tensor, mask: torch.Tensor) -> NeuralMemoryState:
        """One trigger's write (D-36: α depends only on the trigger index)."""
        assume("AS-401", by=__name__)
        return self.longterm.write(state, x, self.retention, mask=mask)

    def longterm_per_trigger(self, start: NeuralMemoryState, env: EnvironmentOut, window: WindowBatch
                             ) -> list[NeuralMemoryState]:
        """States read at each trigger of a window (AS-222): entry m = `start` + the writes of triggers < m.

        Rows whose trigger m is padding are not written (the state is kept, not forgotten): a padded
        trigger must not move the memory.
        """
        states = [start]
        cur = start
        for m in range(window.triggers.time.shape[1] - 1):
            x, mask = self.longterm_inputs(env, window, m)
            nxt = self.longterm_write(cur, x, mask)
            row = window.triggers.mask[:, m].view(-1, 1, 1)

            def keep(a: torch.Tensor, b: torch.Tensor, row: torch.Tensor = row) -> torch.Tensor:
                return torch.where(row, a, b)

            cur = NeuralMemoryState(w1=keep(nxt.w1, cur.w1), w2=keep(nxt.w2, cur.w2), s1=keep(nxt.s1, cur.s1),
                                    s2=keep(nxt.s2, cur.s2), trigger=nxt.trigger)
            states.append(cur)
        return states


# ===================================================================================== parameter count
def count_parameters(cfg: NagaHanaConfig) -> dict[str, int]:
    """Parameters per component and in total, from a meta-device build (no memory is allocated).

    Keys: `COMPONENTS` plus "total". The Generator is not part of the model (D-40); its own count is
    `models.generator.model.generator_parameter_count`.
    """
    with torch.device("meta"):
        model = NagaHana(cfg)
    out: dict[str, int] = {}
    for name in COMPONENTS:
        out[name] = sum(p.numel() for m in model.component(name) for p in m.parameters())
    out["total"] = sum(p.numel() for p in model.parameters())
    if out["total"] != sum(out[n] for n in COMPONENTS):
        raise InvariantViolation("component counts do not add up to the total (a module is unassigned)")
    return out


__all__ = ["COMPONENTS", "PHYSICS_RESIDUAL_WEIGHT", "NagaHana", "count_parameters", "latent_space_hash",
           "latent_space_id", "model_hash", "physics_term"]
