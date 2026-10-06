"""Parameter count of L and the site-sized working memory of the one model (D-63, D-64).

Parameter count
---------------
`count_parameters(preset("L"))` builds every component on the meta device (no memory is allocated) and counts
its parameters. The count follows the data model (D-64): the Decoder's field heads grow with its columns, so
every published copy of the count is regenerated from this build. The Generator is training-only and not part
of the model (D-40); `generator_counts` reports its own count for the canonical layout of the data windows and
for the PCAP adapter's layout.

Working memory sized by the site (D-63)
---------------------------------------
Every site runs the same L model; what changes with the site is the working memory. For a site with H
monitored hosts, an entity ratio rho, a sustained rate of lambda state updates per second and a retention of
T seconds (`SiteConfig`), with b = 4 bytes per stored value (fp32 weights and caches, D-54):

    V      = ceil(rho H)                         entities of the working context
    n      = 2 lambda T / V                      states one entity writes over T under uniform activity
                                                 (two positions per update, one per endpoint, AS-41)
    s(n)   = min(S, ceil(n) + 1), 0 if n = 0     Environment rows reserved per entity: its states and the
                                                 latest-state register (AS-154), at most S = slots_per_entity

    Environment   = V s(n) 2 L_tstct d_tstct b
    Imagination   = (V + G) M_im 2 L_taaft d_taaft b              (AS-13, AS-157)
    TAAFT view    = (V + G) (n_own + n_nb) 2 d_taaft b            one block's gathered cross keys
    TAAFT scores  = 3 H_taaft (V + G) (V + G + M_im + 1) b        one block's self-attention scores
    Route caches  = (1 + B) N_c (T_ctx + 1 + K) 2 L_fc d_fc b     N_c routes in flight (N, or the route chunk)
    Long-term     = 4 D_mem D_hid b                               W_1, W_2 and their momenta (AS-155)
    Weights       = P b                                           P parameters

G adversary slots, M_im kept triggers, n_own and n_nb Environment keys per entity, B lookahead candidates, K
imagined steps, T_ctx the Forecaster's context positions. The memory formulas are those of the compute
profile (`lab/compute.py`), which evaluates them at the critical-infrastructure point, so the default
`SiteConfig` reproduces every published memory figure; the long-term state (32 MiB at L) is the one addition.
The sum is pooled over one server's accelerators (the Environment shards by entity). The plan is the smallest
of one desktop accelerator, one workstation accelerator, or ceil(sum / M) data-centre accelerators of memory M.
The event log the Environment is rebuilt from (D-15 working option, AS-11) is on disk: 24 bytes per update
over the retention window.

Why the formula bounds what the stores hold
-------------------------------------------
- An entity never holds more Environment slots than its states plus the register, and the store keeps at most
  m (2 n_buckets + 2) + 1 slots per entity (property P4 of `memory/environment.py`; 463 at L), a bound it
  refuses to configure above S. Live rows per entity are therefore at most min(ceil(n) + 1, m (2 n_buckets
  + 2) + 1) <= s(n) (`live_rows_per_entity`).
- f(k) = min(S, k + 1) for k >= 1, f(0) = 0, has non-increasing increments (2, 1, ..., 1, 0, ...), so it is
  concave on the integers, and for any split of the same states over the V entities the sum of f is at most
  V f(ceil(mean)) = V s(n) (Jensen's inequality): uniform activity is the worst case.
- The Imagination store keeps exactly the last M_im triggers (AS-157), so after M_im triggers of V + G
  positions it holds exactly the formula's bytes; the long-term state is exactly four D_mem x D_hid matrices.
`tests/test_lab_sizing.py` checks each of these against the stores at the test fixture's widths.

Not counted: host-side indices of the stores, the inference engine's per-trigger bookkeeping (the believed
states, (V + G) M_im d_y values: 34 MB at the published point) and transient activations of one block.

Site classes (AS-727)
---------------------
The classes differ in monitored hosts; every class runs at the published point's rate per host (18,400 updates
per second over 4,096 hosts) and the default four-week retention. `site_table()` gives the full table and
`capacity_table()` the most hosts each accelerator configuration holds:

    small office                25 hosts    one 24 GB desktop accelerator
    branch office              250 hosts    one 48 GB workstation accelerator
    enterprise                 800 hosts    one 80 GB data-centre accelerator
    large enterprise         2,000 hosts    two 80 GB data-centre accelerators
    critical infrastructure  4,096 hosts    four 80 GB data-centre accelerators (the published point)

With every host at full Environment depth the Environment is 64 MiB per entity (512 rows of 128 KiB), so the
host count, not the rate, sets the plan; the rate and the retention set the event log and the compute. A route
chunk (`ForecasterConfig.route_chunk`, identical results) lowers the route caches where a site needs the room.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from nagahana.lab import compute as C
from nagahana.memory.environment import max_cells
from nagahana.models.config import NagaHanaConfig, preset
from nagahana.models.config.components import SiteConfig
from nagahana.models.nagahana import COMPONENTS, count_parameters

#: Display names of the counted components, in processing order (`models.nagahana.COMPONENTS`).
LABELS: dict[str, str] = {
    "inputs": "Input layer (FieldEncoder)",
    "cvgae": "CVG-AE",
    "decoder": "Decoder",
    "tstct": "TSTCT",
    "longterm": "Long-term memory (slow weights)",
    "taaft": "TAAFT (blocks, lenses, readouts, memory probes)",
    "forecaster": "Forecaster",
    "advisor": "Advisor",
    "verifier": "Verifier (process reward, trust and calibration heads)",
}

DAY_S = 86_400.0

#: Accelerator tiers of a plan, smallest first.
TIERS: tuple[str, ...] = ("desktop", "workstation", "data-centre")


def built_counts() -> dict[str, int]:
    """Parameters of L per component and in total ("total"), from the meta-device build."""
    return count_parameters(preset("L"))


def generator_counts() -> dict[str, int]:
    """The Generator's parameters (outside the model) for the canonical and the PCAP column layouts."""
    from nagahana.data.windows import CANONICAL_COLUMNS
    from nagahana.ingest.pcap import COLUMNS
    from nagahana.models.generator.model import generator_parameter_count

    g = preset("L").generator
    return {"canonical": generator_parameter_count(g, CANONICAL_COLUMNS)["total"],
            "pcap": generator_parameter_count(g, COLUMNS)["total"],
            "canonical_columns": len(CANONICAL_COLUMNS), "pcap_columns": len(COLUMNS)}


def table() -> str:
    """Markdown table of the built L model's parameters (exact counts and millions)."""
    b = built_counts()
    total = b["total"]
    rows = [f"| {LABELS[n]} | {b[n]:,} | {b[n] / 1e6:,.2f} M | {100 * b[n] / total:.1f} % |" for n in COMPONENTS]
    tot = f"| **Total (Generator excluded)** | **{total:,}** | **{total / 1e6:,.1f} M** | 100 % |"
    return "\n".join(["| Component | Parameters | Millions | Share |", "|---|---:|---:|---:|", *rows, tot])


@lru_cache(maxsize=8)
def _counted_total(cfg: NagaHanaConfig) -> int:
    return count_parameters(cfg)["total"]


def parameter_total(cfg: NagaHanaConfig) -> int:
    """Parameters of the model built from `cfg` (a preset shares the compute profile's cached build)."""
    try:
        if cfg == preset(cfg.name):
            return C.parameters(cfg.name)["total"]
    except KeyError:
        pass
    return _counted_total(cfg)


def _ceil(x: float) -> int:
    # ceil that ignores floating-point noise below 1e-9 (2 * 1.0 * 3.0 / 3 must give 2, not 3)
    return math.ceil(round(x, 9))


def site_entities(site: SiteConfig) -> int:
    """V = ceil(entity_ratio x monitored_hosts): the entities of the site's working context."""
    return _ceil(site.entity_ratio * site.monitored_hosts)


def states_per_entity(site: SiteConfig, entities: int | None = None) -> float:
    """n = 2 lambda T / V: the states one entity writes over the retention window under uniform activity."""
    v = site_entities(site) if entities is None else entities
    return C.OP.positions_per_update * site.state_rate * site.retention_s / v


def environment_rows_per_entity(cfg: NagaHanaConfig, n: float) -> int:
    """s(n) = min(S, ceil(n) + 1), 0 without states: Environment rows reserved per entity."""
    if n <= 0:
        return 0
    return min(cfg.memory.slots_per_entity, _ceil(n) + 1)


def live_rows_bound(cfg: NagaHanaConfig, n: float) -> int:
    """min(ceil(n) + 1, m (2 n_buckets + 2) + 1), 0 without states: the most rows the store holds per entity."""
    if n <= 0:
        return 0
    mem = cfg.memory
    return min(_ceil(n) + 1, mem.bucket_slots * max_cells(mem.n_buckets) + 1)


def routes_in_flight(cfg: NagaHanaConfig) -> int:
    """Routes the Forecaster imagines at once: N, or the route chunk when one is set."""
    fc = cfg.forecaster
    return fc.routes_n if fc.route_chunk <= 0 else min(fc.route_chunk, fc.routes_n)


def event_log_bytes(site: SiteConfig) -> int:
    """The event log over the retention window on disk: 24 bytes per state update (full telemetry)."""
    return round(C.OP.retained_bytes_per_update * site.state_rate * site.retention_s)


@dataclass(frozen=True)
class AcceleratorPlan:
    """The accelerators that hold one site's working memory: `count` devices of the `tier` class."""

    tier: str
    count: int
    device_bytes: float

    @property
    def capacity_bytes(self) -> float:
        """Memory of the plan in all."""
        return self.count * self.device_bytes

    def describe(self) -> str:
        size = f"{self.device_bytes / 1e9:,.0f} GB"
        if self.count == 1:
            return f"1 x {size} ({self.tier})"
        return f"{self.count} x {size} ({self.tier})"


def accelerator_plan(total_bytes: float, site: SiteConfig) -> AcceleratorPlan:
    """The smallest plan that holds `total_bytes`: one desktop, one workstation, or n data-centre accelerators."""
    if total_bytes <= site.desktop_accelerator_bytes:
        return AcceleratorPlan("desktop", 1, site.desktop_accelerator_bytes)
    if total_bytes <= site.workstation_accelerator_bytes:
        return AcceleratorPlan("workstation", 1, site.workstation_accelerator_bytes)
    return AcceleratorPlan("data-centre", max(1, math.ceil(total_bytes / site.accelerator_bytes)),
                           site.accelerator_bytes)


@dataclass(frozen=True)
class MemoryBudget:
    """The working memory of the one L model at one site, in bytes (fp32 weights and caches, D-54).

    Accelerator memory: `weights`, `environment`, `imagination`, `taaft_view`, `taaft_scores`, `route_cache` and
    `longterm`, whose sum is `total`. Disk: `event_log`. `profile_total` is the sum the compute profile reports
    as its inference memory (`lab/compute.py`): everything but the long-term state.
    """

    site: SiteConfig
    entities: int                   # V
    states_per_entity: float        # n = 2 lambda T / V
    rows_per_entity: int            # s(n), reserved
    live_rows_per_entity: int       # the store's bound on live rows per entity (property P4)
    weights: int
    environment: int
    imagination: int
    taaft_view: int
    taaft_scores: int
    route_cache: int
    longterm: int
    event_log: int

    @property
    def parts(self) -> dict[str, int]:
        """Accelerator memory by part."""
        return {"weights": self.weights, "environment": self.environment, "imagination": self.imagination,
                "taaft_view": self.taaft_view, "taaft_scores": self.taaft_scores, "route_cache": self.route_cache,
                "longterm": self.longterm}

    @property
    def total(self) -> int:
        """Accelerator memory in all."""
        return sum(self.parts.values())

    @property
    def profile_total(self) -> int:
        """The compute profile's inference memory: the total without the long-term state."""
        return self.total - self.longterm

    @property
    def plan(self) -> AcceleratorPlan:
        """The smallest accelerator plan that holds `total`."""
        return accelerator_plan(self.total, self.site)


def memory_budget(site: SiteConfig | None = None, cfg: NagaHanaConfig | None = None) -> MemoryBudget:
    """The working memory of `cfg` (default L) at `site` (default: the published critical-infrastructure point)."""
    site = SiteConfig() if site is None else site
    cfg = C.CFG if cfg is None else cfg
    v = site_entities(site)
    n = states_per_entity(site, v)
    rows = environment_rows_per_entity(cfg, n)
    fc = cfg.forecaster
    return MemoryBudget(
        site=site, entities=v, states_per_entity=n, rows_per_entity=rows,
        live_rows_per_entity=live_rows_bound(cfg, n),
        weights=C.FP32 * parameter_total(cfg),
        environment=C.environment_bytes(cfg, v * rows),
        imagination=C.imagination_store_bytes(cfg, v, cfg.memory.imagination_triggers),
        taaft_view=C.taaft_view_bytes(cfg, v),
        taaft_scores=C.taaft_scores_bytes(cfg, v),
        route_cache=C.route_cache_bytes(cfg, routes_in_flight(cfg), fc.horizon_k, fc.mppi_top_b),
        longterm=C.longterm_state_bytes(cfg),
        event_log=event_log_bytes(site),
    )


def sustained_flops(site: SiteConfig | None = None, cfg: NagaHanaConfig | None = None) -> float:
    """Compute the site needs: lambda state updates per second plus one Forecaster trigger per window over its
    V entities, at the run-time budgets of the compute profile (FLOP per second)."""
    site = SiteConfig() if site is None else site
    cfg = C.CFG if cfg is None else cfg
    op = dataclasses.replace(C.OP, entities=site_entities(site))
    per_trigger = sum(C.trigger_parts(cfg, op).values())
    return C.update_flops(cfg, op) * site.state_rate + per_trigger / cfg.forecaster.window_seconds


def scaled_site(hosts: int, base: SiteConfig | None = None) -> SiteConfig:
    """`base` (default: the published point) with `hosts` monitored hosts at the same rate per host."""
    base = SiteConfig() if base is None else base
    per_host = base.state_rate / base.monitored_hosts
    return dataclasses.replace(base, monitored_hosts=hosts, state_rate=per_host * hosts)


#: Site classes (AS-727): monitored hosts at the published point's rate per host and retention.
SITE_CLASSES: Mapping[str, SiteConfig] = MappingProxyType({
    "small office": scaled_site(25),
    "branch office": scaled_site(250),
    "enterprise": scaled_site(800),
    "large enterprise": scaled_site(2_000),
    "critical infrastructure": SiteConfig(),
})


def max_hosts(capacity_bytes: float, cfg: NagaHanaConfig | None = None, *, base: SiteConfig | None = None) -> int:
    """The most monitored hosts whose working memory fits `capacity_bytes` (0 if one host does not fit).

    Hosts scale from `base` at its rate per host (`scaled_site`); the total grows with the host count, so the
    answer is found by doubling and bisection on the exact budget."""
    def fits(h: int) -> bool:
        return memory_budget(scaled_site(h, base), cfg).total <= capacity_bytes

    if not fits(1):
        return 0
    lo, hi = 1, 2
    while fits(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 1:                     # fits(lo) and not fits(hi)
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _gb(x: float) -> str:
    return f"{x / 1e12:,.2f} TB" if x >= 1e12 else f"{x / 1e9:,.2f} GB"


def site_table(cfg: NagaHanaConfig | None = None, classes: Mapping[str, SiteConfig] | None = None) -> str:
    """Markdown table of the memory budget, the plan, the event log and the compute of every site class."""
    classes = SITE_CLASSES if classes is None else classes
    head = ("| Site class | Hosts | Updates/s | Environment | Imagination | TAAFT working set | Route caches | Weights "
            "| Long-term | Total | Accelerators | Event log | Compute |")
    out = [head, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for name, site in classes.items():
        b = memory_budget(site, cfg)
        days = site.retention_s / DAY_S
        out.append(
            f"| {name} | {site.monitored_hosts:,} | {site.state_rate:,.0f} | {_gb(b.environment)} | {_gb(b.imagination)} "
            f"| {_gb(b.taaft_view + b.taaft_scores)} | {_gb(b.route_cache)} | {_gb(b.weights)} | {_gb(b.longterm)} "
            f"| {_gb(b.total)} | {b.plan.describe()} | {_gb(b.event_log)} ({days:g} d) "
            f"| {sustained_flops(site, cfg) / 1e12:,.2f} TFLOP/s |")
    return "\n".join(out)


def capacity_table(cfg: NagaHanaConfig | None = None, *, base: SiteConfig | None = None,
                   counts: Sequence[int] = (1, 2, 3, 4)) -> str:
    """Markdown table: the most hosts each accelerator configuration holds (the published rate per host)."""
    base = SiteConfig() if base is None else base
    plans = [AcceleratorPlan("desktop", 1, base.desktop_accelerator_bytes),
             AcceleratorPlan("workstation", 1, base.workstation_accelerator_bytes),
             *[AcceleratorPlan("data-centre", n, base.accelerator_bytes) for n in counts]]
    out = ["| Accelerators | Memory | Most monitored hosts |", "|---|---:|---:|"]
    for plan in plans:
        out.append(f"| {plan.describe()} | {plan.capacity_bytes / 1e9:,.0f} GB "
                   f"| {max_hosts(plan.capacity_bytes, cfg, base=base):,} |")
    return "\n".join(out)


def describe_budget(b: MemoryBudget) -> str:
    """One site's budget as text lines (bytes and decimal units)."""
    lines = [f"entities {b.entities:,}; states per entity over the retention {b.states_per_entity:,.1f}; "
             f"Environment rows per entity {b.rows_per_entity} reserved, at most {b.live_rows_per_entity} live"]
    lines += [f"{name:<13} {value:>16,} B  {_gb(value)}" for name, value in b.parts.items()]
    lines.append(f"{'total':<13} {b.total:>16,} B  {_gb(b.total)}  ->  {b.plan.describe()}")
    lines.append(f"{'event log':<13} {b.event_log:>16,} B  {_gb(b.event_log)} on disk")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import io
    import sys

    from nagahana.core.config import site_config

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Parameter count of L and the site-sized working memory.")
    parser.add_argument("--site", help="a site YAML (conf/site/site.yaml layout): print that site's budget only")
    args = parser.parse_args()
    if args.site:
        print(describe_budget(memory_budget(site_config(args.site))))
    else:
        print(table())
        gc = generator_counts()
        print(f"\nGenerator (outside the count): {gc['canonical']:,} at the canonical {gc['canonical_columns']}-column "
              f"layout; {gc['pcap']:,} at the PCAP adapter's {gc['pcap_columns']} columns.\n")
        print(site_table())
        print()
        print(capacity_table())
