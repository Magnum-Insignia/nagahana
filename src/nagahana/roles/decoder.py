"""The Decoder role: the live, provenance-tagged view of memory ([A-13]; D-05, P-01).

It reads the Environment and Imagination (memory/access.py), decodes latents with
`models.decoder.ParallelDecoder`, flattens planes into one view (method held, D-05), and tags every
element observed / believed / forecast (`roles.contracts.ProvenanceTag`). Reading the Monitor so
deviations show in the view is proposal P-12.
"""

from __future__ import annotations

from nagahana.core.errors import NotBuiltYet


class DecoderView:
    """Role orchestrator (template)."""

    def render(self, *args: object, **kwargs: object) -> object:
        """Produce the provenance-tagged network view."""
        raise NotBuiltYet("Decoder view", waiting_on=("D-05", "D-11a", "P-01"))
