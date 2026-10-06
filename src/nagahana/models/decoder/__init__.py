"""Decoder: parallel output from latents to a readable, provenance-tagged graph (build-spec §2.4).

- `model`: `Decoder` (alias `ParallelDecoder`), `DecodedFields`, `DecodedView`, `ViewElement`.
- `buckets`: service-class buckets of categorical columns (AS-33, AS-106).
- `losses`: stage-3 pieces: reconstruction, candidate hyperedges, physics, masking, reordering.
"""

from nagahana.models.decoder.model import DecodedFields, DecodedView, Decoder, ParallelDecoder, ViewElement

__all__ = ["DecodedFields", "DecodedView", "Decoder", "ParallelDecoder", "ViewElement"]
