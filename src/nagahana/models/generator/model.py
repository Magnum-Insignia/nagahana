"""Generator: training-only, physics-bounded event variants (D-40; [A-17], [Q-37], [Q-06]).

The owner's design
------------------
- Purpose: "for a particular event we can slide the observability, signatures, attack traces, etc as
  similar samples for it to have clarity of the partial observable concept" [Q-37]. It is "variants of
  the same event with varying signatures, traces, impacts as observability sliding" (ai-mod-arch,
  component 4), varying "features & relationships that the encoder is capturing".
- Methods: "even the generator uses the same concept of physics informed concept to not hallucinate
  impossible ones, and further uses energy based modeling via ssl, further using Joint Energy Models,
  generative training (where we make the model generate the missing parts etc to make it our expert
  network data generator), autoregressive & diffusion methods" [A-17].
- World-model framing: "the data is just guidance and the generative sampling with self supervision
  is it (or us modeling) to fill the gaps with awareness of boundaries of capabilities" [Q-06].

Contract
--------
    (x~, G~) = G_phi(x, G, c, eps),        y(x~, G~) = y(x, G)      (label-preserving)

The controls c are observability (drop packet fields, hide a sensor, sample 1-in-n), signature (ports,
timing jitter, tool fingerprints), trace (re-ordering, rate) and topology (benign relations).

Families (D-14 held; the option in force resolves through `governance.decisions.require("D-14")`)
-----------------------------------------------------------------------------------------------
Working option "all families" (AS-27). `active_families()` maps the option in force to producer
families (`config.families_for_option`); the deterministic families run under every option.

    registry name           builds                                              module
    observability-sliding   [DropFields, FlowOnlyExport, PacketSampling,        transforms.py
                             SensorHiding]
    signature-variation     [PortRemap, TimingJitter, RateScaling, Reorder]     transforms.py
    topology-variation      [HyperedgeDropout, RewireBenign]                    transforms.py
    tool-fingerprint        ToolFingerprintSwap (needs the tool-class table)    transforms.py
    masked-generative       MaskedFieldTransformer (MaskGIT order)              masked.py
    autoregressive          MaskedFieldTransformer (left-to-right reading)      masked.py
    diffusion               TabularDiffusion (TabDDPM-style)                    diffusion.py
    energy-ssl              EnergySSL (denoising-score-matched energy, SGLD)    energy.py
    jem                     EnergyAcceptance (JEM-style acceptance)             acceptance.py

The pipeline (`pipeline.VariantPipeline`) draws variants from the producers and accepts them through
`acceptance.AcceptanceGate`.

Guards
------
- Training only (D-40): `PhysicsBoundedSampler.sample`, `VariantPipeline.generate` and the fit
  functions require RunMode.TRAIN.
- Physics-bounded ([A-17], decided): hard limits checked on every variant (limits.py); Phi_phys
  reported per variant; the hard threshold Phi_phys <= tau is proposal P-11, enabled per run
  (`enabled_proposals`; `hard_gate_enabled`).
- No leakage (P-23, assumed in AS-367): sources and training windows must be REAL training-split
  samples; zero-shot samples are always refused (D-23).

Parameter count: `generator_parameter_count(cfg, columns)` builds the learned families on the meta
device. It is separate from the model's count (the Generator is training-only augmentation).

Family lineage: JEM (Grathwohl et al., ICLR 2020, arXiv:1912.03263); denoising diffusion (Ho et al.,
NeurIPS 2020, arXiv:2006.11239) in the TabDDPM form (Kotelnikov et al., ICML 2023, arXiv:2209.15421);
masked generative modelling (Chang et al., MaskGIT, CVPR 2022, arXiv:2202.04200); noise-conditional
energies sampled by annealed Langevin dynamics (Song and Ermon, NeurIPS 2019, arXiv:1907.05600).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from typing import Any

import numpy as np
import torch

from nagahana.core.errors import ConfigMissing, ProposalNotEnabled
from nagahana.core.modes import RunMode, require_mode
from nagahana.core.registry import Registry
from nagahana.datamodel.columnar import Column
from nagahana.governance import decisions
from nagahana.governance.assumptions import assume
from nagahana.models.config.components import GeneratorConfig
from nagahana.models.generator.acceptance import AcceptanceGate, EnergyAcceptance, MarginalEnergy, physics_gate_active
from nagahana.models.generator.codec import DISCRETE_KINDS, NUMERIC_KINDS, DiscreteSpec, FieldCodec, NumericSpec
from nagahana.models.generator.config import GeneratorPolicy, families_for_option
from nagahana.models.generator.diffusion import TabularDiffusion
from nagahana.models.generator.energy import EnergySSL
from nagahana.models.generator.masked import build_masked_model
from nagahana.models.generator.pipeline import Producer, VariantBatch, VariantPipeline
from nagahana.models.generator.transforms import (
    FingerprintClasses,
    ToolFingerprintSwap,
    observability_transforms,
    signature_transforms,
    topology_transforms,
)

GENERATORS: Registry[Any] = Registry("generator family")


def _family(name: str, what: str, build: Callable[..., Any]) -> None:
    # Every family records the use of AS-27 (the working option of held D-14); strict mode raises there.
    def factory(*args: Any, **kwargs: Any) -> Any:
        assume("AS-27", by=__name__)
        return build(*args, **kwargs)

    GENERATORS.register(name, summary=f"{what} (D-14 held; working option AS-27)")(factory)


_family("jem", "Joint Energy Model reading: acceptance by TAAFT's marginal energy",
        lambda energy_fn, cfg: EnergyAcceptance(energy_fn, quantiles=cfg.energy_quantiles))
_family("energy-ssl", "energy-based self-supervised generation: denoising-score-matched energy, annealed Langevin",
        lambda cfg, codec, policy=None: EnergySSL(cfg, codec, policy if policy is not None else GeneratorPolicy()))
_family("masked-generative", "generate the missing parts (MaskGIT-style masked modelling)",
        lambda cfg, codec: build_masked_model(cfg, codec))
_family("autoregressive", "record-by-record generation (left-to-right reading of the masked model)",
        lambda cfg, codec: build_masked_model(cfg, codec))
_family("diffusion", "TabDDPM-style Gaussian diffusion of numeric fields", lambda cfg, codec: TabularDiffusion(cfg, codec))
_family("observability-sliding", "drop packet fields, flow-only export, 1-in-n sampling, sensor hiding",
        lambda cfg, policy=None: observability_transforms(cfg, policy))
_family("signature-variation", "port remap, timing jitter, rate scaling, re-ordering",
        lambda cfg, policy=None: signature_transforms(cfg, policy))
_family("topology-variation", "hyperedge dropout and benign rewiring", lambda cfg, policy=None: topology_transforms(cfg))
_family("tool-fingerprint", "client tool fingerprints swapped inside their tool class",
        lambda table: [ToolFingerprintSwap(table=table)])


def active_families(*, configured: str | None = None) -> tuple[tuple[str, ...], bool]:
    """(producer families, JEM acceptance on) for the D-14 option in force (module docstring)."""
    option = decisions.require("D-14", configured, by=__name__).value
    assert option is not None
    return families_for_option(option)


class PhysicsBoundedSampler:
    """Wraps families with the decided guards.

    Parameters
    ----------
    family: one producer or a sequence of producers (`pipeline.Producer`).
    enabled_proposals: proposals enabled in this run (the hard Phi gate is P-11).
    cfg: Generator config (needed to sample: MTU, tau, attempts). energy: optional JEM-style acceptance.
    """

    def __init__(self, family: Any, *, enabled_proposals: Collection[str] = (), cfg: GeneratorConfig | None = None,
                 energy: EnergyAcceptance | None = None, seed: int = 0) -> None:
        self.family = family
        self.enabled = frozenset(enabled_proposals)
        self.cfg, self.energy, self.seed = cfg, energy, seed

    def sample(self, event: Any, controls: Any) -> VariantBatch:
        """Draw variants of one real training event. Training only.

        event: (ColumnarUpdates window, UpdateLabels, Sample). controls: number of variants (int).
        """
        require_mode(RunMode.TRAIN, component="Generator")
        if self.cfg is None:
            raise ConfigMissing("PhysicsBoundedSampler needs a GeneratorConfig to sample (MTU, tau, attempts).")
        producers: Sequence[Producer] = list(self.family) if isinstance(self.family, list | tuple) else [self.family]
        gate = AcceptanceGate.from_config(self.cfg, energy=self.energy, enabled_proposals=self.enabled)
        window, labels, source = event
        return VariantPipeline(self.cfg, producers, gate, seed=self.seed).generate(window, labels, source, n=int(controls))

    def hard_gate_enabled(self) -> bool:
        """True only if proposal P-11 (accept iff Phi_phys <= tau) is enabled in this run."""
        try:
            decisions.require_proposal("generator-physics-gate", self.enabled)
        except ProposalNotEnabled:
            return False
        return True

    def gate_active(self) -> bool:
        """True if the Phi_phys gate rejects in this run (P-11 enabled, or decided)."""
        return physics_gate_active(self.enabled)


def nominal_codec(columns: Sequence[Column], cfg: GeneratorConfig) -> FieldCodec:
    """A codec with full-size class tables for every column (for parameter counting only; not fitted)."""
    numeric = {j: NumericSpec(0.0, 1.0, 0.0, 1.0, seen=True) for j, c in enumerate(columns) if c.kind in NUMERIC_KINDS}
    codes = np.arange(cfg.cat_vocab - 1, dtype=np.int64)
    discrete = {j: DiscreteSpec(codes, np.array([cfg.cat_vocab], dtype=np.int64), np.ones(1), seen=True)
                for j, c in enumerate(columns) if c.kind in DISCRETE_KINDS}
    return FieldCodec(columns, numeric, discrete, value_bins=cfg.value_bins, cat_vocab=cfg.cat_vocab, fitted_on=())


def generator_parameter_count(cfg: GeneratorConfig, columns: Sequence[Column],
                              policy: GeneratorPolicy | None = None) -> dict[str, int]:
    """Parameters of the learned families for a column layout, built on the meta device (no memory)."""
    codec = nominal_codec(columns, cfg)
    pol = policy if policy is not None else GeneratorPolicy()
    with torch.device("meta"):
        masked = build_masked_model(cfg, codec)
        diffusion = TabularDiffusion(cfg, codec)
        energy = EnergySSL(cfg, codec, pol)
    counts = {
        "masked-generative (= autoregressive, shared network)": sum(p.numel() for p in masked.parameters()),
        "diffusion": sum(p.numel() for p in diffusion.parameters()),
        "energy-ssl": sum(p.numel() for p in energy.parameters()),
    }
    counts["total"] = sum(counts.values())
    return counts


__all__ = ["FingerprintClasses", "GENERATORS", "MarginalEnergy", "PhysicsBoundedSampler", "active_families",
           "generator_parameter_count", "nominal_codec"]
