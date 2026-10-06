# Ground-truth world simulator assumptions (AS-800 ... AS-829)

Engineer, world simulator (decision P-14), 2026-10-06. Each entry states what is assumed, which held
decision or unspecified detail it stands for, the reasoning, and the evidence. "code run: <test>,
<result>" cites a check in `tests/test_worldsim_*.py`. These IDs are not yet in
`governance/assumptions.py` (its registration is outside this work package); the build report lists
the requested entries. Until they are registered, the code cites them in docstrings and comments only.

The simulator exists because public datasets have label noise and no hidden-state truth, so belief,
trust and forecast quality (and the information-ceiling audit, P-15) cannot be measured on them alone
(Engelen, Rimmer and Joosen, "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case
Study", IEEE SPW 2021, DOI 10.1109/SPW53761.2021.00009).

---

### AS-800: Fixed structure per scenario, per-world variation in attributes and dynamics
- **Assumed:** every world of a scenario shares the topology structure (counts, segments, Purdue
  levels, reachability); only per-world attributes (vulnerability placement, credential reuse,
  attacker entry, benign intensity and timing) and the stochastic dynamics vary.
- **Stands for:** detail of P-14 (how a batch of worlds is defined).
- **Why:** fixing the structure makes a batch of worlds a stack of equal-shaped arrays, which is what
  `jax.vmap` and `jax.lax.scan` require; the variation that matters for generalisation
  (which host is weak, where the attacker starts, how it paces) is retained.
- **Evidence:** code run: `test_worldsim_dynamics.py::test_different_worlds_differ`, worlds 0 and 1 of
  one scenario differ; `::test_determinism_same_seed`, a world is a pure function of its seed.

### AS-801: One counter-based generator shared by both backends
- **Assumed:** all randomness is Threefry-2x32 with 20 rounds, addressed by a key hierarchy
  (seed -> world -> stream -> draw), implemented once against an array namespace so NumPy and JAX run
  the same source.
- **Stands for:** detail of P-14 ("reproducible bit-for-bit across backends for the same seed").
- **Why:** a counter-based bijection makes every draw a pure function of its address, independent of
  consumption order, so a vectorised run and a loop agree and the two backends agree.
- **Evidence:** Salmon, Moraes, Dror and Shaw, "Parallel random numbers: as easy as 1, 2, 3", SC 2011,
  DOI 10.1145/2063384.2063405. code run: `test_worldsim_rng.py::test_known_answer_vectors` (Random123
  vectors) and `::test_numpy_jax_bits_and_uniform_identical`.

### AS-802: Discrete decisions and event times are integers
- **Assumed:** every discrete decision (which channel, technique, target, session) is an integer
  comparison on the generator's words, and event times are quantised to microseconds; floating
  transforms appear only in continuous magnitudes.
- **Stands for:** detail of P-14 (bit-for-bit cross-backend worlds despite libm differences).
- **Why:** integer comparisons are identical on every backend; a floating `log`/`exp` can differ in
  the last bit between NumPy and XLA, so nothing structural (ordering, which host, which technique)
  is allowed to depend on one.
- **Evidence:** code run: `test_worldsim_dynamics.py::test_jax_equals_numpy_bit_for_bit`, the full
  integer event timeline is identical across backends for four worlds.

### AS-803: Next-event simulation by thinning with a fixed slot budget
- **Assumed:** the marked point process is advanced by Ogata's modified thinning with a constant
  intensity bound a_max, over a fixed number of candidate slots (`SimulationConfig.event_slots`); a
  world that fills its slots before the horizon is flagged saturated.
- **Stands for:** detail of P-14 (continuous-time, next-event, "in the style of Gillespie") made
  fixed-shape for `scan`.
- **Why:** thinning turns a time-varying-rate process into constant-rate candidates with an
  accept/reject, which is exactly a fixed-length `scan`; the saturation flag keeps truncation honest
  rather than silent.
- **Evidence:** Ogata, "On Lewis' simulation method for point processes", IEEE Trans. Information
  Theory 27(1), 1981, DOI 10.1109/TIT.1981.1056305; Lewis and Shedler, "Simulation of nonhomogeneous
  Poisson processes by thinning", Naval Research Logistics 26(3), 1979. code run:
  `test_worldsim_dynamics.py::test_monotone_time`.

### AS-804: Benign arrivals are an inhomogeneous Poisson process with integer diurnal and weekly profiles
- **Assumed:** benign sessions arrive with an intensity that is a base per-host rate modulated by a
  piecewise-constant integer profile over the hour of day (24) and day of week (7); session sizes are
  heavy-tailed (bounded Pareto / lognormal).
- **Stands for:** detail of P-14 (benign traffic with diurnal and weekly cycles and heavy-tailed flow
  sizes).
- **Why:** integer profiles keep the arrival intensity exact and bit-identical (AS-802) and model the
  business-hours rhythm directly; heavy-tailed sizes reproduce the measured distribution of flow
  volumes.
- **Evidence:** Leland, Taqqu, Willinger and Wilson, "On the self-similar nature of Ethernet traffic",
  IEEE/ACM Trans. Networking 2(1), 1994, DOI 10.1109/90.282603.

### AS-805: OT polling is a deterministic periodic channel
- **Assumed:** OT polling (a historian or SCADA server reading its controllers) is generated as
  scheduled sessions at a fixed interval with bounded jitter, outside the stochastic jump process.
- **Stands for:** detail of P-14 (Modbus polling as a benign OT traffic model).
- **Why:** polling is periodic by design, not a Poisson arrival; modelling it deterministically is
  both more faithful and avoids forcing a near-deterministic timer into the thinning loop.
- **Evidence:** the Modbus application protocol is a request/response master-slave poll (Modbus
  Organization, "Modbus Application Protocol Specification V1.1b3", 2012).

### AS-806: The attacker is a stochastic policy over a logical attack graph with a seeded entry
- **Assumed:** the attacker's enabled moves are the techniques whose preconditions hold in the hidden
  state (reachability, an exposed vulnerable service, a stolen credential, a foothold, a privilege);
  firing a technique writes facts that may enable others. Each internet-reachable internal host that
  exposes a web service carries the matching web exploit, so initial access is always possible.
- **Stands for:** held D-11c (adversary objective) and detail of P-14 (attacker over an attack graph).
- **Why:** this is forward simulation of a MulVAL-style logical attack graph; the seeded entry makes
  the campaign's intended way in reliable while the rest of the chain stays stochastic.
- **Evidence:** Ou, Govindavajhala and Appel, "MulVAL: A Logic-based Network Security Analyzer",
  USENIX Security 2005. code run: `test_worldsim_dynamics.py::test_no_lateral_move_before_source_foothold`
  and `::test_cause_links_form_earlier_dag`.

### AS-807: ICS tactics map to the nearest model stage; native identifiers are preserved
- **Assumed:** a technique keeps its native ATT&CK identifier and tactic (Enterprise or ICS); the
  emitted stage label is the model class (`models.vocab.STAGES`), and ICS-only tactics (Inhibit
  Response Function, Impair Process Control) map to "impact", the nearest model class.
- **Stands for:** detail of P-14 ("aligned with models/vocab.py") for ICS coverage.
- **Why:** the stage head has the 14 Enterprise tactics plus "none"; ICS tactics have no class there,
  so the stage is the nearest class while the ground truth keeps the exact ICS technique and tactic.
- **Evidence:** ATT&CK Enterprise (https://attack.mitre.org/techniques/enterprise/) and ICS
  (https://attack.mitre.org/techniques/ics/). code run:
  `test_worldsim_dynamics.py::test_stage_labels_consistent_with_catalogue`.

### AS-808: Hidden-state encoding
- **Assumed:** per entity the hidden state is a control level (none/user/admin), persistence,
  attacker knowledge, held credentials, and per-entity impact flags (collected, C2, exfiltrated,
  denial-of-service, OT manipulation, ransom), plus an integer suspicion the defender reacts to.
- **Stands for:** detail of P-14 (per-entity compromise state, privileges, persistence, attacker
  knowledge; ground-truth technique and stage at every time).
- **Why:** these are exactly the preconditions and effects the attack graph reads and writes, and the
  belief and forecast outputs the evaluation scores against.
- **Evidence:** the kill-chain tactics of ATT&CK (as above). code run:
  `test_worldsim_emit.py::test_entity_truth_marks_compromise`.

### AS-809: Sensor models and their parameters
- **Assumed:** the sensor fabric is tap, NetFlow/IPFIX, Zeek, authentication log and IDS, each with
  coverage, granularity, sampling, export timeouts, packet loss, clock skew and (for IDS) a detection
  probability and false-alarm rate; a session is recorded by the most capable covering sensor, host
  logs fill identity-session gaps, and uncovered or dropped sessions leave no record.
- **Stands for:** P-03 (observation statuses) and detail of P-14 (explicit, configurable observation
  models).
- **Why:** this reproduces the partial, multi-fidelity, imperfect view a real monitoring estate gives,
  which is what belief under partial observability must be tested against.
- **Evidence:** NetFlow export on flow expiry or timeout (Claise, "Cisco Systems NetFlow Services
  Export Version 9", RFC 3954, 2004; Trammell and Boschi, IPFIX, RFC 7011, 2013). code run:
  `test_worldsim_observe.py::test_netflow_marks_packet_fields_not_supplied`,
  `::test_sampling_keeps_about_one_in_n`, `::test_ids_detection_probability_bounds`,
  `::test_coverage_excludes_uncovered_domain`.

### AS-810: Flow-state and packet granularity; end reason and unanswered flag
- **Assumed:** the tap emits at flow-state granularity by default (D-51) or per packet; every flow
  carries `flow.end_reason` and `flow.unanswered` (D-53), with scans reported as unanswered idle-ended
  SYNs.
- **Stands for:** D-51 (flow-state updates) and D-53 (end-reason fields) applied to synthetic traffic.
- **Why:** the simulator's records must match the PCAP adapter's emission contract so a model trained
  on real captures sees the same record semantics on simulated worlds.
- **Evidence:** D-51, D-53 in `governance/decisions.py`. code run:
  `test_worldsim_emit.py::test_records_validate_and_round_trip`.

### AS-811: Infiltration and episode definition
- **Assumed:** an entity is infiltrated when it is the internal actor of a malicious action whose stage
  is a post-initial-access tactic; an episode is one compromised internal entity, its start the first
  malicious activity on it and its completion the time it reached an infiltration stage.
- **Stands for:** AS-18 (infiltration state) and the `EpisodeTable` contract of
  `evaluation/predictions.py`.
- **Why:** this matches the evaluation's episode semantics (start and completion times) and the
  problem statement's "before compromise is completed".
- **Evidence:** `evaluation/predictions.py` `EpisodeTable`. code run:
  `test_worldsim_emit.py::test_episode_table_is_valid`.

### AS-812: Synthetic addressing and group entities
- **Assumed:** entities get deterministic, unique IPv4 addresses by segment; a multicast or broadcast
  destination is a `multicast` entity (D-47); a service is a `service` entity keyed
  `<address>:<port>/<proto>`.
- **Stands for:** D-47 (group addresses) and the entity model of `datamodel.records`.
- **Why:** the data model keys entities by address and kind; unique addresses keep the entity graph
  well formed, and a group destination is a node of its own, not a machine.
- **Evidence:** D-47 in `governance/decisions.py`. code run:
  `test_worldsim_dynamics.py::test_no_traffic_from_nonexistent_hosts`.

### AS-813: Defender actions and their effects
- **Assumed:** the optional defender reacts to accumulated suspicion by isolating a host (removing its
  reachability), blocking its exposed services, or patching its vulnerabilities; the attacker's
  reachability, service exposure and vulnerability are the effective ones after these actions.
- **Stands for:** detail of P-14 (optional defender actions: block, isolate, patch).
- **Why:** defences change the attack graph the attacker plays on, which the forward simulation must
  honour; coupling them through effective reachability keeps the ground truth consistent.
- **Evidence:** D3FEND defensive techniques (https://d3fend.mitre.org/). code run:
  `test_worldsim_dynamics.py::test_jax_equals_numpy_bit_for_bit` exercises the defender in the batch.

### AS-814: Self-similarity through heavy tails and Poisson arrivals
- **Assumed:** aggregate traffic self-similarity is produced by heavy-tailed session sizes and
  durations superposed as a Poisson arrival process, not fitted to a target Hurst exponent.
- **Stands for:** detail of P-14 (self-similar aggregate traffic).
- **Why:** the superposition of many heavy-tailed on/off sources is the established generative
  mechanism for self-similarity, so it arises by construction rather than being imposed.
- **Evidence:** Willinger, Taqqu, Sherman and Wilson, "Self-Similarity Through High-Variability", IEEE/ACM
  Trans. Networking 5(1), 1997, DOI 10.1109/90.554723.

### AS-815: Synthetic features obey the physics boundary
- **Assumed:** session features are generated so the unconditional flow-accounting limits hold: bytes
  per direction are at least packets times the minimum header and at most packets times the MTU, a
  TCP flag count does not exceed the packet count, and the maximum inter-arrival gap does not exceed
  the duration.
- **Stands for:** detail of P-14 ("simulated data obey the same boundary the model learns", `physics/`).
- **Why:** records that violated the physics residuals would be impossible states the model is taught
  cannot occur, so the simulator must stay inside the same boundary (D-18).
- **Evidence:** the residual catalogue in `physics/residuals.py` (MTUBound, FlagCountBound,
  IATMaxBound, MinHeaderBound).

### AS-816: Credential reuse and domain trust
- **Assumed:** a credential valid on an internal host may be stored on another host (a holder archetype)
  with probability `credential_reuse`; a domain controller stores credentials valid on every internal
  host. Credential validity is independent of reachability, which is checked when the credential is
  used.
- **Stands for:** detail of P-14 (credentials and trust relationships) and held D-11c (adversary
  surface).
- **Why:** credential reuse and domain trust are the dominant lateral-movement surface in enterprise
  intrusions; separating validity from reachability lets the attack graph decide usability at move time.
- **Evidence:** ATT&CK T1078 Valid Accounts, T1003 OS Credential Dumping
  (https://attack.mitre.org/techniques/enterprise/). code run:
  `test_worldsim_dynamics.py::test_no_lateral_move_before_source_foothold`.

### AS-817: Time horizon, event budget and record caps
- **Assumed:** a scenario sets the simulated horizon and the number of candidate slots; the attacker
  has a maximum number of events; scan and denial-of-service events fan out to a bounded number of
  sessions.
- **Stands for:** detail of P-14 (bounding a fixed-shape simulation).
- **Why:** fixed budgets keep the simulation fixed-shape for `scan` and bound the record count; the
  saturation flag records when a budget bound; the fan caps keep a sweep or flood finite.
- **Evidence:** code run: `test_worldsim_emit.py::test_simulate_worlds_batch` runs a batch within the
  budgets.

### AS-818: The world index enters only through the generator key
- **Assumed:** the only per-world input to the simulation is the world index, consumed solely to derive
  the world's generator key; everything else is shared or derived from that key.
- **Stands for:** detail of P-14 (vmap over worlds equals a loop over worlds).
- **Why:** if the world index affected anything but the key, a vectorised batch and a loop could
  diverge; keeping it to the key makes them identical by construction.
- **Evidence:** code run: `test_worldsim_dynamics.py::test_vmap_equals_sequential`.

### AS-819: The JAX backend runs under 64-bit types
- **Assumed:** the JAX simulation runs inside `jax.enable_x64(True)` so int64 and float64 match the
  NumPy computation.
- **Stands for:** detail of P-14 (cross-backend identity) and consistency with D-54 (fp64 where it
  matters).
- **Why:** without 64-bit types JAX would compute the clock and the counter arithmetic at 32 bits and
  diverge from NumPy.
- **Evidence:** code run: `test_worldsim_emit.py::test_jax_backend_emits_identical_records`.

### AS-820: lax.map is the default world mapping; vmap is the accelerator path
- **Assumed:** the default batch runner maps worlds with `jax.lax.map` over a per-world `jax.lax.scan`;
  an explicit `jax.vmap` runner is provided for accelerators and the equivalence test.
- **Stands for:** detail of P-14 (jax.vmap over many worlds) under the XLA CPU backend.
- **Why:** on the XLA CPU backend the compile of `vmap` over a `scan` is pathological for more than two
  worlds (it does not terminate within minutes), whereas `lax.map` over the single-world scan compiles
  in a second or two and gives the same results; `vmap` remains correct and is used on accelerators and
  to prove equivalence.
- **Evidence:** code run: single-world jit scan compiles in about 4 s and `lax.map` over 8 worlds in
  about 1.5 s, while `jit(vmap(scan))` over 3 worlds did not compile within 100 s on this host;
  `test_worldsim_dynamics.py::test_vmap_equals_sequential` confirms the `vmap` result equals the
  sequential worlds at two worlds.
