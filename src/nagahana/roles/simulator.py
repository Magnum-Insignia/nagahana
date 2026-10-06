"""The Simulator role: builds and maintains the Environment (D-35; [A-03], [A-05], [A-11], [A-12]).

The Simulator takes input and generates, updates and manages the Environment. Per state update (event-driven,
D-30), the inference engine (`inference/engine.Engine.ingest`) runs its steps:

1. the update joins the event log (`inference/buffer.EventLog`, the durable record under the working option of
   D-15, AS-11);
2. the open window is built as training builds it (`data.windows.build_window`, whose structure is
   `graph.window`'s hypergraph as of every position, AS-01, AS-02, D-52);
3. CVG-AE encodes the new positions (posterior mean, AS-05);
4. TSTCT's step appends their keys and values to the Environment store (`memory/environment.py`); the Simulator
   is the only writer of the Environment, and beliefs never enter it (D-35, D-43).

The physics term scores the reconstructions during training (`physics/term.py`); the Decoder has none of its
own. `Simulator` is the role's handle on that computation: it converts state-update objects to the engine's
columnar form and returns the Forecaster triggers that fell due while the updates were processed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from nagahana.datamodel.columnar import Column
from nagahana.datamodel.records import StateUpdate

if TYPE_CHECKING:
    from nagahana.inference.engine import Engine, TriggerResult
    from nagahana.memory.environment import EnvironmentStore


class Simulator:
    """The Simulator role over the inference engine (module docstring).

    Parameters
    ----------
    engine: the inference engine of the site.
    columns: the column layout of the adapter's state updates (`datamodel.columnar.Column`).
    """

    def __init__(self, engine: Engine, columns: Sequence[Column]) -> None:
        self.engine = engine
        self.columns = tuple(columns)

    def ingest(self, update: StateUpdate) -> list[TriggerResult]:
        """Process one state update into the Environment; returns the triggers that fell due meanwhile."""
        return self.ingest_many([update])

    def ingest_many(self, updates: Sequence[StateUpdate]) -> list[TriggerResult]:
        """Process state updates (in event-time order) into the Environment; returns the triggers that fired."""
        if not updates:
            return []
        return self.engine.ingest_states(updates, self.columns)

    @property
    def environment(self) -> EnvironmentStore | None:
        """The Environment store the engine maintains (None before the first update)."""
        return self.engine.store
