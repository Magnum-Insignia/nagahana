"""Model configuration: one frozen dataclass per component, assembled into `NagaHanaConfig`.

Presets
-------
- `"L"`: the full model, about 1 B parameters without the Generator (build-spec §4; owner,
  2026-10-02: "go for the full model i mean 1b one"). The exact count is whatever the built modules
  hold; `models.nagahana.count_parameters(preset("L"))` reports it from a meta-device build.
- `"tiny"`: a test fixture (the same code at small widths, for unit tests and the CPU smoke run).
  It is not a model: NagaHana is only L (owner, 2026-10-02).

Each component's dataclass lives in its own module so that the engineer who owns a component edits
only that file. Every field has a comment naming the decision (D-) or assumption (AS-) behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from nagahana.models.config.components import (
    AdvisorConfig,
    CVGAEConfig,
    DecoderConfig,
    FieldEncoderConfig,
    ForecasterConfig,
    GeneratorConfig,
    GraphConfig,
    MemoryConfig,
    TAAFTConfig,
    TrainingConfig,
    TSTCTConfig,
    VerifierConfig,
)


@dataclass(frozen=True)
class NagaHanaConfig:
    """The whole model's configuration."""

    name: str
    inputs: FieldEncoderConfig = field(default_factory=FieldEncoderConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    cvgae: CVGAEConfig = field(default_factory=CVGAEConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    tstct: TSTCTConfig = field(default_factory=TSTCTConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    taaft: TAAFTConfig = field(default_factory=TAAFTConfig)
    forecaster: ForecasterConfig = field(default_factory=ForecasterConfig)
    advisor: AdvisorConfig = field(default_factory=AdvisorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    #: Version tag of the shared latent space (P-19): caches and decoders refuse other spaces.
    latent_space: str = "nagahana-z-v1"

    @property
    def latent_dim(self) -> int:
        """dz = Dc + G·C."""
        return self.cvgae.cont_dim + self.cvgae.disc_groups * self.cvgae.disc_classes


def preset(name: str) -> NagaHanaConfig:
    """`"L"` (the NagaHana model, target 1.0–1.1 B) or `"tiny"` (a test fixture, not a model)."""
    if name == "L":
        return NagaHanaConfig(name="L")
    if name == "tiny":
        return NagaHanaConfig(
            name="tiny",
            inputs=FieldEncoderConfig(d_field=16, hash_rows=512, n_frequencies=4, pool_heads=2, d_update=32),
            graph=GraphConfig(max_nodes=8, rwse_steps=4),
            cvgae=CVGAEConfig(dim=32, layers=2, heads=2, cont_dim=8, disc_groups=2, disc_classes=4),
            decoder=DecoderConfig(hidden=32, n_service_classes=8, edge_hidden=32),
            tstct=TSTCTConfig(dim=64, blocks=2, heads=4, spatial_heads=1, temporal_heads=2, causal_heads=1,
                              mlp_hidden=128, temporal_window=16, spatial_keys=8, causal_keys=4,
                              gate_hidden=8, causal_candidate_cap=16),
            memory=MemoryConfig(slots_per_entity=96, n_buckets=20, bucket_slots=2, longterm_dim=64,
                                longterm_hidden=64, imagination_triggers=3),   # 2·(2·20+2)+1 = 85 ≤ 96 (AS-150)
            taaft=TAAFTConfig(dim=64, blocks=3, heads=4, mlp_hidden=128, adversary_slots=4, d_hyp=16, imagination_triggers=3,
                              own_states=4, neighbour_states=4, lens_hidden=32, descent_steps=3, n_goals=4),
            forecaster=ForecasterConfig(dim=64, blocks=2, heads=4, mlp_hidden=128, n_techniques=32,
                                        context_entities=6, horizon_k=4, routes_n=6, mppi_top_b=3),
            advisor=AdvisorConfig(dim=64, blocks=2, heads=4, mlp_hidden=128, n_actions=16, beam_width=4,
                                  max_steps=2, rollouts=3),
            verifier=VerifierConfig(dim=64, blocks=2, heads=4, mlp_hidden=128, context_entities=6),
            generator=GeneratorConfig(dim=32, blocks=2, heads=2, mlp_hidden=64, diffusion_steps=10, denoiser_hidden=32,
                                      value_bins=32, cat_vocab=16, max_records=8, denoiser_blocks=2),
            training=TrainingConfig(window_updates=64, batch_windows=2, max_entities=32),
        )
    raise KeyError(f"unknown preset {name!r}; known: 'L', 'tiny'")


__all__ = [
    "AdvisorConfig", "CVGAEConfig", "DecoderConfig", "FieldEncoderConfig", "ForecasterConfig", "GeneratorConfig",
    "GraphConfig", "MemoryConfig", "NagaHanaConfig", "TAAFTConfig", "TSTCTConfig", "TrainingConfig",
    "VerifierConfig", "preset", "replace",
]
