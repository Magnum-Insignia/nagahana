# The ground-truth world simulator

Decision P-14. The world simulator builds simulated IT and OT networks whose hidden state is known and
whose observation is modelled explicitly, and emits records in the NagaHana data model. It exists
because public datasets carry label noise and no hidden-state truth (Engelen, Rimmer and Joosen, IEEE
SPW 2021), so belief accuracy, trust estimation, forecast skill and the Bayes-optimal information
ceiling of the audit (P-15) can be measured exactly only against worlds whose ground truth is known.

The package is `nagahana.worldsim`; the lab entry point is `nagahana.lab.world_sim.simulate`. The lab
stays isolated from the production core and imports JAX lazily.

## What a world is

A scenario (`worldsim.config.ScenarioConfig`) fixes the structure of a network and the rules of its
benign traffic, its attacker, its defender and its sensors. A world is one random instance of a
scenario, drawn from a world index under a run seed. Every world of a scenario shares the structure
and differs only in per-world attributes and the stochastic dynamics (AS-800), which is what lets a
batch of worlds run as one vectorised computation.

## Backends and reproducibility

The hidden dynamics run on two backends:

- JAX with Equinox is primary: a per-world `jax.lax.scan` over event slots, mapped over the world axis
  and compiled with `equinox.filter_jit`. The default world mapping is `jax.lax.map`; an explicit
  `jax.vmap` path (`run_jax_vmap`) is the accelerator route and the one the equivalence test uses. On
  the XLA CPU backend a `vmap` of a `scan` compiles poorly beyond two worlds, so `lax.map` is the
  default there (AS-820).
- NumPy is the reference and fallback: a Python loop over the same per-slot transition.

All randomness is a counter-based generator (Threefry-2x32-20; Salmon et al., SC 2011), addressed by a
key hierarchy seed -> world -> stream -> draw (AS-801). Because every draw is a pure function of its
address, a world is a bit-for-bit function of its seed, identical across the two backends and
independent of how many worlds run together. Every discrete decision is an integer comparison and
every event time is an integer number of microseconds, so nothing structural depends on a floating
transform that could differ between NumPy and XLA (AS-802). The JAX backend runs under
`jax.enable_x64` so its 64-bit arithmetic matches NumPy (AS-819).

## Topology

`worldsim.topology.build_topology` places entities, the services they expose and a reachability
matrix. Enterprise estates have user segments, a server segment, an identity segment, a DMZ and an
egress; OT plants follow the Purdue reference model (ISA-95 / IEC 62264): engineering workstations,
historians and an OT DMZ at level 3; HMIs and SCADA servers at level 2; PLCs and RTUs at level 1;
field devices at level 0. Reachability encodes the segmentation and the IEC 62443 zone-and-conduit
policy: the internet reaches only the DMZ, the enterprise reaches OT only through the OT DMZ, and
inside OT a level reaches its own and adjacent levels. Addresses are deterministic and unique; a
multicast or broadcast destination is a group entity (D-47). OT protocol ports follow the IANA
registry (Modbus 502, DNP3 20000, IEC 60870-5-104 2404, S7comm 102, EtherNet/IP 44818).

## Hidden state and the attacker

The hidden state (`worldsim.state.DynState`) holds, per entity, the attacker's control level, its
persistence, its knowledge, the credentials it holds, and per-entity impact (collection, C2,
exfiltration, denial of service, OT manipulation, ransom), plus the suspicion the defender reacts to
(AS-808). The attacker is a stochastic policy over a logical attack graph in the style of MulVAL (Ou,
Govindavajhala, Appel, USENIX Security 2005): each technique reads facts (reachability, an exposed
vulnerable service, a stolen credential, a foothold, a privilege) and, when it fires, writes facts
that enable further techniques (AS-806). Techniques carry their native ATT&CK Enterprise or ICS
identifier and tactic; the emitted stage is the model class (`models.vocab.STAGES`), with ICS-only
tactics mapped to the nearest class, "impact", while the native identifiers are kept in the ground
truth (AS-807). Campaign styles (`worldsim.vocab.CAMPAIGNS`) set the pace and loudness: slow
reconnaissance, fast ransomware, data exfiltration, denial of service, OT manipulation and a full
APT. An optional defender isolates, blocks or patches in response to suspicion (AS-813). Every
attacker event records a causal parent (the event that established the fact it used), so the ground
truth is an explicit attack DAG.

## Dynamics

The hidden process is a marked point process advanced by Ogata's modified thinning (Ogata 1981; Lewis
and Shedler 1979): a constant intensity bound gives candidate event times at an exponential spacing,
and each candidate is accepted into a channel (a benign session, an attacker technique, a defender
action) in proportion to that channel's instantaneous integer intensity, the rest being thinned
(AS-803). Benign sessions arrive as an inhomogeneous Poisson process whose intensity follows an
integer diurnal and weekly profile, with heavy-tailed session sizes (AS-804); OT polling is a
deterministic periodic channel (AS-805); self-similarity arises from the superposition of
heavy-tailed sessions (Willinger et al., IEEE/ACM ToN 1997; AS-814). The loop is a fixed number of
candidate slots, so it is fixed-shape for `scan`; a world that fills its slots before the horizon is
flagged saturated (AS-817).

## Observation

`worldsim.observe` turns hidden sessions into telemetry. The sensor fabric is a packet tap, a
NetFlow/IPFIX exporter, a Zeek-style logger, host authentication logs and an IDS (AS-809). A session
is seen only by sensors whose coverage includes one of its endpoints; of those, the most capable
not-dropped sensor supplies the record, and host logs fill identity-session gaps. A NetFlow exporter
supplies flow-level fields and marks packet-level fields NOT_SUPPLIED; sampling keeps one in n
sessions; packet loss and lack of coverage leave no record at all. Every supplied field carries its
observation status (P-03), and absence is a status, never a zero (D-41). Records are emitted at
flow-state granularity (D-51) with the end-reason and unanswered fields (D-53), or per packet
(AS-810). The IDS raises alerts with a per-technique detection probability and a Poisson false-alarm
rate, reported as a ground-truth table since an alert is not a field of the data model.

## Outputs

`worldsim.simulate.simulate_world` returns a `WorldOutput`:

- `updates`: a `datamodel.columnar.ColumnarUpdates`, built through `to_columnar`, so it validates
  against the data model by construction.
- `labels`: a label table aligned row for row with the records, in the columns `data.labels` uses.
- `episodes`: one row per compromised internal entity, with the start of malicious activity and the
  time it reached an infiltration stage (AS-18, AS-811), usable as an `evaluation.predictions.EpisodeTable`.
- `entity_truth`: the per-entity hidden state at the end of the run.
- `event_truth`: every ground-truth event with its causal parent (the attack DAG).
- `alerts`: the IDS alerts (true and false).
- `meta`: scenario, campaign, family, seed, world index, saturation and counts.

`WorldOutput.source_data()` wraps the records and labels as a `data.windows.SourceData`, so a simulated
world flows straight into the windowing pipeline and the model.

## Scenarios and the CLI

`worldsim.scenarios` is the named library; the YAML files under `conf/worldsim` are generated from the
configuration dataclasses by `write_library` and reload to equal objects, with no undecided
placeholder. The CLI (wired in through `register_cli`) offers:

- `nagahana worldsim list` -- the scenario library.
- `nagahana worldsim generate --scenario NAME --worlds N --out DIR [--seed S] [--backend numpy|jax]`
  -- simulate a scenario's worlds and write their records, labels, episodes and ground-truth tables.
- `nagahana worldsim write-library --out conf/worldsim` -- regenerate the scenario YAML.

## Invariants (tests/test_worldsim_*.py)

Determinism under seed; the JAX and NumPy backends produce identical worlds; a `jax.vmap` batch equals
the sequential worlds; event time is monotone; no technique fires before its precondition event and
the cause links form an earlier-in-time DAG; stage labels match the technique catalogue; no event
involves an entity outside the topology; NetFlow records mark packet-level fields NOT_SUPPLIED and
sampling keeps about one in n; IDS detection and false-alarm behave as configured; the emitted records
validate against the data model and flow into the windowing pipeline; the scenario YAML round-trips
through the dataclasses.
