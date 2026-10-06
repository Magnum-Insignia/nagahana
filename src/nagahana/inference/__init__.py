"""Inference: the live engine and forensic replay (architecture §1 "two settings of one engine").

- `engine.Engine`: incremental ingestion → Environment (TSTCT.step into the store) → triggers (cadence +
  capped priority, AS-12) → TAAFT → Forecaster → Verifier readings → role contracts.
- `explain`: driving features by Expected Gradients over field states and energy-lens shares.
- `forensic.replay`: the engine over a capture → `ForensicReport`.
- `buffer`: the engine's event log with stable entity rows.

Runs only in RunMode.INFER_LIVE / RunMode.FORENSIC_REPLAY (`core/modes.py`).
"""

from nagahana.inference.engine import Budgets, Engine, EngineSettings, TriggerResult
from nagahana.inference.forensic import build_report, replay

__all__ = ["Budgets", "Engine", "EngineSettings", "TriggerResult", "build_report", "replay"]
