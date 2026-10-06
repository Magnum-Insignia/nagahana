"""End-to-end smoke run of NagaHana (tiny preset) on the CSE-CIC-IDS2018 infiltration sample slice.

    python scripts/smoke_e2e.py [--capture PATH] [--workdir DIR]

Stages 3, 4 and 5 train for a few steps in stream order with the TSTCT carry (D-51), each checked on one
batch with fixed random draws (loss finite and lower after the steps); then the inference engine replays
a 20 s cut (RunMode.FORENSIC_REPLAY) and prints the forecasts and the forensic report. Timings are
measured and printed. The tiny preset is a test fixture: nothing here is an accuracy result.

Facts about the capture: nagahana-app/sample-data/FACTS.md (backdoor 14:45:40 UTC, scan from 14:46:22).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nagahana.training.smoke import run_smoke, utc  # noqa: E402

DEFAULT = ROOT.parent / "nagahana-app" / "sample-data" / "cic2018-infiltration-slice.pcap"
DAY = (2018, 2, 28)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--capture", type=Path, default=DEFAULT)
    p.add_argument("--workdir", type=Path, default=None, help="where the cut captures go (default: a temporary directory)")
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    workdir = args.workdir if args.workdir is not None else Path(tempfile.mkdtemp(prefix="nagahana-smoke-"))
    t0 = time.perf_counter()
    rep = run_smoke(args.capture, workdir, train_cut=(utc(DAY, 14, 44, 30), utc(DAY, 14, 47, 0)),
                    replay_cut=(utc(DAY, 14, 44, 50), utc(DAY, 14, 46, 1)), labeller="cic2018-infiltration-slice",
                    mtu=1500.0, seed=0, steps=args.steps, windows_before=3, attribution_samples=2, routes_n=6)
    print(f"train cut: {rep.updates} state updates, {rep.windows} windows")
    for stage in (3, 4, 5):
        before, after = rep.same_batch[stage]
        print(f"stage {stage}: stream losses {[round(x, 3) for x in rep.stream_losses[stage]]}; "
              f"same batch {before:.4f} -> {after:.4f}")
    for r in rep.results:
        f = r.forecast
        print(f"trigger {r.time:.0f} ({r.kind}): P_inf = {[round(x, 3) for x in f.p_inf]}, trust {r.trust:.2f}, "
              f"F(mode route) {r.explanation.f_value:.3f}, EG gap {r.explanation.completeness_gap:.2e}")
        print("  driving:", [(d.feature, round(d.contribution, 4)) for d in f.driving_features[:6]])
    assert rep.report is not None
    for _t, line in rep.report.timeline:
        print("timeline:", line)
    for line in (*rep.report.counterfactuals, *rep.report.observability_gaps, *rep.report.tamper_signs):
        print("-", line)
    for k, v in rep.timings.items():
        print(f"time {k:<22}{v:8.2f} s")
    print(f"total {time.perf_counter() - t0:.2f} s; notes: {rep.notes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
