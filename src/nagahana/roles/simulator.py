"""The Simulator: builds and maintains the Environment (D-35; [A-03]–[A-05], [A-11], [A-12]).

"the first phase of the work is done by simulator always which does the work of taking input and
generating/updating/managing the environment" (ai-mod-arch).

Per state update (event-driven, D-30):
1. fold the update into the hypergraph (graph/builder.py; D-04 held);
2. encode with CVG-AE (models/cvgae) into the shared latent space;
3. run TSTCT, which appends to the Environment KV cache (memory/kvcache.py; D-15 held);
4. score reconstructions with the shared physics term during training (physics/term.py).

The Simulator is the only writer of the Environment (memory/access.py). Beliefs never enter it.
"""

from __future__ import annotations

from nagahana.core.errors import NotBuiltYet
from nagahana.datamodel.records import StateUpdate


class Simulator:
    """Role orchestrator (template)."""

    def ingest(self, update: StateUpdate) -> None:
        """Process one state update into the Environment."""
        raise NotBuiltYet("Simulator ingest (builder → CVG-AE → TSTCT → Environment)", waiting_on=("D-04", "D-15"))
