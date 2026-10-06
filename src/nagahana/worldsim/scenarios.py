"""The named scenario library and its YAML serialization (P-14).

The configuration dataclasses in `config.py` are the single source of truth. A scenario is defined in
code as a `ScenarioConfig`; the YAML files under conf/worldsim are generated from these objects by
`write_library`, and `load_scenario` reads a YAML file back into a `ScenarioConfig` and validates it
through the dataclasses' own `__post_init__`. `to_dict` and `from_dict` round-trip exactly (tested),
so an edited YAML is still a fully validated configuration and never carries an undecided `???`.

The library spans the campaign styles the brief names: slow reconnaissance, fast ransomware lateral
movement, data exfiltration, denial of service and OT manipulation, plus a benign-only baseline and a
small fixture for tests. Enterprise scenarios use the IT topology; OT scenarios add the Purdue levels.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from nagahana.worldsim.config import (
    FLAT_7,
    FLAT_24,
    AttackConfig,
    BenignConfig,
    DefenderConfig,
    ObservationConfig,
    ScenarioConfig,
    SensorConfig,
    SimulationConfig,
    TopologyConfig,
)

#: A full enterprise IT estate: two user segments, servers, identity, a DMZ and egress.
_ENTERPRISE = TopologyConfig(counts=(
    ("workstation", 24), ("admin_workstation", 3), ("file_server", 2), ("app_server", 2),
    ("database", 1), ("domain_controller", 1), ("dmz_web", 2), ("dmz_mail", 1), ("gateway", 1),
    ("cloud_egress", 1),
), enterprise_segments=3)

#: A plant with the Purdue levels and a small enterprise front office that reaches the OT DMZ.
_OT_PLANT = TopologyConfig(counts=(
    ("workstation", 8), ("admin_workstation", 1), ("file_server", 1), ("domain_controller", 1),
    ("dmz_web", 1), ("gateway", 1), ("ot_dmz", 1), ("historian", 1), ("engineering_workstation", 2),
    ("scada_server", 1), ("hmi", 2), ("plc", 4), ("rtu", 2), ("field_device", 4),
), enterprise_segments=2, ot_from_enterprise=True)

#: A sensor fabric that mixes full packet tap coverage, a NetFlow exporter and a Zeek logger with an
#: IDS; the NetFlow exporter samples and the tap has a little loss and clock skew, so the model sees a
#: realistic partial, multi-fidelity view (P-03 statuses, D-41).
_MIXED_SENSORS = ObservationConfig(sensors=(
    SensorConfig("tap", granularity="flow-state", packet_loss=0.02, clock_skew_s=0.0),
    SensorConfig("netflow", sampling_n=4, active_timeout_s=60.0, inactive_timeout_s=15.0),
    SensorConfig("zeek"),
    SensorConfig("ids", detection_prob=0.6, false_alarm_per_hour=2.0),
    SensorConfig("authlog", coverage=("enterprise",)),
))


def _scenario(name: str, description: str, network: str, topology: TopologyConfig, campaign: str,
              *, novelty: str = "", benign: BenignConfig | None = None,
              attack: AttackConfig | None = None, defender: DefenderConfig | None = None,
              observation: ObservationConfig | None = None,
              simulation: SimulationConfig | None = None) -> ScenarioConfig:
    return ScenarioConfig(
        name=name, description=description, network=network, topology=topology,
        benign=benign or BenignConfig(),
        attack=attack or AttackConfig(campaign=campaign),
        defender=defender or DefenderConfig(enabled=True),
        observation=observation or _MIXED_SENSORS,
        simulation=simulation or SimulationConfig(horizon_s=7200.0, event_slots=16384),
        novelty=novelty,
    )


def _library() -> dict[str, ScenarioConfig]:
    tiny = ScenarioConfig(
        name="tiny", description="Small fixture for tests (not a model or a realistic site).",
        network="tiny-net",
        topology=TopologyConfig(counts=(("workstation", 3), ("file_server", 1), ("domain_controller", 1),
                                        ("dmz_web", 1), ("gateway", 1))),
        attack=AttackConfig(campaign="apt_full", start_min_s=30.0, start_max_s=90.0, max_events=32,
                            vuln_density=0.5),
        defender=DefenderConfig(enabled=False),
        benign=BenignConfig(diurnal=FLAT_24, weekly=FLAT_7, session_rate_per_host_hour=60.0),
        observation=ObservationConfig(sensors=(SensorConfig("tap"),)),
        simulation=SimulationConfig(horizon_s=600.0, event_slots=512),
    )
    items = [
        tiny,
        _scenario("enterprise_apt", "Enterprise APT: recon, exploit, lateral movement, C2, exfiltration.",
                  "ent-apt", _ENTERPRISE, "apt_full"),
        _scenario("enterprise_ransomware", "Fast ransomware: rapid lateral movement then encryption impact.",
                  "ent-ransom", _ENTERPRISE, "fast_ransomware",
                  attack=AttackConfig(campaign="fast_ransomware", start_min_s=300.0, start_max_s=1200.0,
                                      max_events=400, vuln_density=0.3)),
        _scenario("enterprise_exfiltration", "Data exfiltration over a command-and-control channel.",
                  "ent-exfil", _ENTERPRISE, "data_exfiltration"),
        _scenario("enterprise_dos", "Denial of service against public-facing services.",
                  "ent-dos", _ENTERPRISE, "denial_of_service",
                  attack=AttackConfig(campaign="denial_of_service", start_min_s=300.0, start_max_s=900.0,
                                      max_events=600)),
        _scenario("slow_recon", "Low-and-slow reconnaissance that stays below flow-based thresholds.",
                  "ent-recon", _ENTERPRISE, "slow_recon",
                  simulation=SimulationConfig(horizon_s=14400.0, event_slots=20000)),
        _scenario("ot_manipulation", "OT manipulation: enterprise entry, pivot through the OT DMZ, manipulate a PLC.",
                  "ot-plant", _OT_PLANT, "ot_manipulation",
                  attack=AttackConfig(campaign="ot_manipulation", start_min_s=600.0, start_max_s=2400.0,
                                      max_events=300, vuln_density=0.6, local_vuln_density=0.6,
                                      credential_reuse=0.6)),
        _scenario("ot_benign_baseline", "OT plant with no attacker: benign polling and IT traffic only.",
                  "ot-benign", _OT_PLANT, "apt_full",
                  attack=AttackConfig(campaign="apt_full", enabled=False), defender=DefenderConfig(enabled=False)),
        _scenario("enterprise_benign_baseline", "Enterprise estate with no attacker (benign baseline).",
                  "ent-benign", _ENTERPRISE, "apt_full",
                  attack=AttackConfig(campaign="apt_full", enabled=False), defender=DefenderConfig(enabled=False)),
        _scenario("enterprise_apt_novel", "Enterprise APT held out as a novel family for zero-shot evaluation.",
                  "ent-apt-novel", _ENTERPRISE, "apt_full", novelty="novel"),
    ]
    return {s.name: s for s in items}


SCENARIOS: dict[str, ScenarioConfig] = _library()


def get_scenario(name: str) -> ScenarioConfig:
    """A named scenario from the library."""
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(sorted(SCENARIOS))}") from None


def list_scenarios() -> tuple[tuple[str, str], ...]:
    """(name, description) of every library scenario, in name order."""
    return tuple((n, SCENARIOS[n].description) for n in sorted(SCENARIOS))


def to_dict(scenario: ScenarioConfig) -> dict[str, Any]:
    """A plain, YAML-friendly dict of a scenario (tuples become lists; nested dataclasses expand)."""
    return dataclasses.asdict(scenario)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(tuple(x) if isinstance(x, list) else x for x in value)


def from_dict(data: dict[str, Any]) -> ScenarioConfig:
    """Rebuild a `ScenarioConfig` from a dict (lists become tuples), validated by the dataclasses."""
    topo = data["topology"]
    topology = TopologyConfig(
        counts=tuple((str(n), int(c)) for n, c in topo["counts"]),
        enterprise_segments=int(topo.get("enterprise_segments", 2)),
        allow_cross_segment=bool(topo.get("allow_cross_segment", True)),
        ot_from_enterprise=bool(topo.get("ot_from_enterprise", True)),
        ipv4_base=str(topo.get("ipv4_base", "10.0.0.0")),
    )
    b = data.get("benign", {})
    benign = BenignConfig(
        session_rate_per_host_hour=float(b.get("session_rate_per_host_hour", 30.0)),
        diurnal=_as_tuple(b.get("diurnal", FLAT_24)), weekly=_as_tuple(b.get("weekly", FLAT_7)),
        ot_poll_interval_s=float(b.get("ot_poll_interval_s", 2.0)),
        ot_poll_jitter_s=float(b.get("ot_poll_jitter_s", 0.2)),
        heavy_tail_alpha=float(b.get("heavy_tail_alpha", 1.6)),
    )
    a = data.get("attack", {})
    attack = AttackConfig(
        campaign=str(a.get("campaign", "apt_full")), enabled=bool(a.get("enabled", True)),
        start_min_s=float(a.get("start_min_s", 300.0)), start_max_s=float(a.get("start_max_s", 1800.0)),
        vuln_density=float(a.get("vuln_density", 0.25)), local_vuln_density=float(a.get("local_vuln_density", 0.5)),
        credential_reuse=float(a.get("credential_reuse", 0.3)), max_events=int(a.get("max_events", 256)),
    )
    d = data.get("defender", {})
    defender = DefenderConfig(
        enabled=bool(d.get("enabled", False)), response_rate_per_hour=float(d.get("response_rate_per_hour", 4.0)),
        suspicion_threshold=float(d.get("suspicion_threshold", 3.0)), isolate=int(d.get("isolate", 2)),
        block=int(d.get("block", 1)), patch=int(d.get("patch", 1)),
    )
    sensors = tuple(
        SensorConfig(
            kind=str(s["kind"]), coverage=_as_tuple(s.get("coverage", ("enterprise", "ot"))),
            granularity=str(s.get("granularity", "flow-state")), sampling_n=int(s.get("sampling_n", 1)),
            active_timeout_s=float(s.get("active_timeout_s", 60.0)),
            inactive_timeout_s=float(s.get("inactive_timeout_s", 15.0)),
            packet_loss=float(s.get("packet_loss", 0.0)), clock_skew_s=float(s.get("clock_skew_s", 0.0)),
            clock_drift_ppm=float(s.get("clock_drift_ppm", 0.0)), detection_prob=float(s.get("detection_prob", 0.6)),
            false_alarm_per_hour=float(s.get("false_alarm_per_hour", 1.0)), reliability=float(s.get("reliability", 0.0)),
        )
        for s in data.get("observation", {}).get("sensors", [{"kind": "tap"}])
    )
    observation = ObservationConfig(sensors=sensors)
    s = data.get("simulation", {})
    simulation = SimulationConfig(
        horizon_s=float(s.get("horizon_s", 3600.0)), event_slots=int(s.get("event_slots", 4096)),
        start_epoch_s=float(s.get("start_epoch_s", 1_700_000_000.0)),
        start_jitter_s=float(s.get("start_jitter_s", 604800.0)),
    )
    return ScenarioConfig(
        name=str(data["name"]), description=str(data.get("description", "")), network=str(data["network"]),
        topology=topology, benign=benign, attack=attack, defender=defender, observation=observation,
        simulation=simulation, novelty=str(data.get("novelty", "")),
    )


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Read a scenario YAML into a validated `ScenarioConfig`."""
    import yaml

    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return from_dict(data)


def write_library(directory: str | Path) -> list[Path]:
    """Write every library scenario to conf/worldsim/<name>.yaml, generated from the dataclasses."""
    import yaml

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in sorted(SCENARIOS):
        payload = to_dict(SCENARIOS[name])
        text = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False, allow_unicode=False)
        header = (f"# Scenario {name} for the ground-truth world simulator (P-14).\n"
                  "# Generated from nagahana.worldsim.config dataclasses; edit and it is re-validated on load.\n")
        p = out / f"{name}.yaml"
        p.write_text(header + text, encoding="utf-8")
        written.append(p)
    return written


__all__ = [
    "SCENARIOS", "from_dict", "get_scenario", "list_scenarios", "load_scenario", "to_dict",
    "write_library",
]
