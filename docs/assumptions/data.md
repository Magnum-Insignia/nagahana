# Data pipeline assumptions (AS-300 … AS-338)

Engineer E (data pipeline), 2026-10-02. Each entry: what is assumed, what held decision or open detail
it stands for, the reasoning, and the evidence. "(unverified)" marks a fact written from memory of a
source that must be checked against the files in stage 1 (deep data analysis, Q-02). A finding from
a code run is cited as "code run: <test or script>, <number>".

These IDs are not yet in `governance/assumptions.py` (outside this engineer's scope); the requested
registry entries are in the build report. Until they are registered, code cites them in docstrings
and comments only; code that relies on registered assumptions calls `assume()` (AS-11, AS-12, AS-18,
AS-20, AS-34, AS-35, AS-41).

---

## Ingest (CSV adapters, `ingest/csv_flows.py`)

### AS-300 Event time of a flow record is the flow's end
- **Assumed:** `event_time = flow_start + duration` when the duration is a valid measurement, else
  `flow_start` (recorded in the audit column `time_basis`: 1 end, 0 start, −1 none).
- **Stands for:** detail of D-30 (one record = one state update) for flow sources.
- **Why:** a flow record carries totals (bytes, packets, duration, IAT statistics) that exist only at
  the end of the flow; an exporter emits it then. Placing it at the start would show the model, at
  t_start, information from up to `duration` seconds later: future leakage, the failure mode the
  owner's "no future leakage" invariant forbids.
- **Evidence:** NetFlow / IPFIX export flow records on expiry (RFC 7011 §1, "Flow Record" exported
  after the flow ends or times out; to verify the exact wording). code run:
  `tests/test_data_csv.py::test_cic2018_column_map_and_units` (event time = start + 2 s).

### AS-301 Timestamp resolution and time zone
- **Assumed:** the reorder uncertainty of a CSV record is at least its timestamp resolution: 1 s for
  CSE-CIC-IDS2018 (`dd/mm/YYYY HH:MM:SS`), 60 s for CIC-IDS2017 files written to the minute, 1 µs
  for CTU-13. CSV times carry no zone: `utc_offset_hours` converts local time to UTC; without it the
  text is read as UTC and `clock_quality` says "timezone-unverified".
- **Stands for:** detail (ordering resolved before the model, Q-20).
- **Why:** the order of two records inside the same second is not known from the file; saying so is
  the ARCH §9.2 rule "tagged, not hidden". Absolute clock time is not a training input (D-50), so
  the zone matters only for aligning a CSV with captures or with the publisher's attack tables.
- **Evidence:** the sample slice's FACTS.md measured the CSE-CIC-IDS2018 publisher table as local time
  UTC−4 against capture timestamps in UTC. Timestamp formats per dataset (unverified, to check per
  file in stage 1).

### AS-302 Values that are not measurements become NOT_SUPPLIED
- **Assumed:** NaN, ±inf, negative durations / IATs, CICFlowMeter's −1 "no initial window" marker,
  protocol 0 in CICFlowMeter output, IAT statistics of flows with too few packets, and window
  averages offered as counts or categories (CIC-IoT-2023) are NOT_SUPPLIED, and every refusal is
  counted in `stats["refused_values"]`.
- **Stands for:** D-41 applied to values a source *writes* but that cannot be observations.
- **Why:** a zero or −1 written for "no value" is absence encoded as a number; keeping it would teach
  the model a fake magnitude (D-41). Counting keeps the refusal auditable.
- **Evidence:** CIC-IDS2017/2018 contain infinite `Flow Byts/s` values and negative durations
  (Engelen, Rimmer & Joosen, IEEE SPW 2021, and Liu et al., IEEE CNS 2022 describe flow-construction
  errors; the specific value patterns are unverified). That CICFlowMeter writes protocol 0 for an
  unknown protocol is unverified.

### AS-303 Byte counts as the source defines them (revised 2026-10-02)
- **Assumed:** each byte definition has its own field (catalogue fields added with the lead's
  approval): CICFlowMeter `TotLen Fwd/Bwd Pkts` → `flow.payload_bytes_fwd/bwd` (CICFlowMeter sums
  payload bytes: unverified, from memory of `BasicFlow.java`, which adds `getPayloadBytes()` to the
  length statistics); `flow.bytes_*` (IP layer) and `flow.bidir_ratio` (a ratio of IP-layer bytes)
  are NOT_SUPPLIED for CICFlowMeter files; `Tot Fwd + Tot Bwd Pkts` → `flow.packets_total`. Argus
  `SrcBytes` / `TotBytes − SrcBytes` stay in `flow.bytes_*` (Argus counts whole packets; definition
  unverified) and `TotPkts` → `flow.packets_total`. The PCAP adapter fills all three new fields
  (payload bytes = transport payload per direction, any IP protocol).
- **Stands for:** the catalogue's "source-specific definitions are recorded per adapter".
- **Why:** one slot, one definition: a model trained on CIC payload bytes and run on PCAP IP-layer
  bytes would otherwise see a systematic header offset in the same slot.
- **Evidence:** code run: `test_cic2018_column_map_and_units` (payload fields observed, IP-layer
  fields not supplied); `test_new_payload_and_packet_fields` (PCAP: 2 × 18 forward and 200 backward
  payload bytes, 7 packets). To verify in stage 1 by recomputing one CSV row from the matching PCAP.

### AS-304 IAT statistics need enough packets; CICFlowMeter's std is a sample std
- **Assumed:** `flow.iat_mean` / `iat_max` need ≥ 2 packets and `flow.iat_var` ≥ 3 (as the PCAP
  adapter: a variance before the second inter-arrival does not exist). `Flow IAT Std` is converted
  from the sample standard deviation (n − 1) to the population variance (n) the PCAP adapter uses:
  `var = std² · (n − 1)/n`, n = packets − 1.
- **Stands for:** detail (one field, one definition across adapters).
- **Why:** CICFlowMeter writes 0 for the IAT of one-packet flows (absence as zero) and uses Apache
  Commons Math `SummaryStatistics`, whose `getStandardDeviation` is bias-corrected (unverified).
- **Evidence:** code run: `test_cic2018_column_map_and_units` (0.1 s sample std over 4 IATs →
  0.0075 s²).

### AS-305 Port-access evidence on flow records
- **Assumed:** `derived.portscan_sequential` / `_random` are computed for CSV sources with addresses
  using the PCAP adapter's definition (per source–destination pair, last 32 distinct destination
  ports, ≥ 4 needed), over records in event-time (arrival) order, so a record only uses records that
  arrived no later than itself.
- **Stands for:** the problem statement's "port scan signatures"; the catalogue leaves the
  definition to stage 1.
- **Why:** the same definition across sources keeps the field's meaning; arrival order keeps it
  causal (flow start order would use records that arrive later).
- **Evidence:** code run: `test_port_access_evidence_matches_pcap_definition` (ports 20–25 → 1.0).

### AS-306 Synthetic entities for rows without addresses
- **Assumed:** a row without both addresses gets two entities of its own (kind `host`, key
  `synthetic:<source>:<record>:initiator|responder`, `synthetic = True`), `flow.src_ip/dst_ip`
  NOT_SUPPLIED; their `internal` flag must be stated by the caller (`synthetic_internal`, no default;
  the adapter raises otherwise).
- **Stands for:** detail of AS-41 (2 positions per update) when identity is absent.
- **Why:** the 2-positions layout needs an entity per role; inventing identity (e.g. keying by port)
  would fabricate shared structure. Synthetic entities never recur, so they add no history, no
  transition target and no hyperedge shared with another update.
- **Evidence:** the CSE-CIC-IDS2018 "TrafficForML" CSVs carry only `Dst Port` except the 20-02-2018
  file, which adds `Flow ID, Src IP, Src Port, Dst IP` (unverified; to check per file). code run:
  `test_cic2018_without_addresses_needs_stated_synthetic_flag`.
- **Consequence for the owner:** CSE-CIC-IDS2018 temporal and topological learning must come from
  the dataset's PCAPs (as the sample slice does), not from its CSVs.

### AS-307 CIC-IoT-2023 CSVs: what is mapped, and what can be learned
- **Assumed:** the public CSVs carry 46 features computed over windows of packets (averages), a label,
  and no addresses, ports or timestamps (unverified; Neto et al., Sensors 23(13):5941, 2023). Only
  `Protocol Type` (when integral) → `flow.protocol`, `Duration` (the paper's TTL, unverified) →
  `pkt.ttl_mean`, and integral `*_count` → `flow.flag_count.*` are mapped. A `ts` column is used if
  present; otherwise event time is NaN and the window builder gives every update time 0 with origin
  0 (no time is invented; D-49 forbids index-based positions).
- **Stands for:** detail (§4b.7 names CIC-IoT-2023 as a dataset).
- **What the model can learn from these rows:** per-update field statistics and the FieldEncoder's
  embeddings of the mapped fields; CVG-AE/Decoder reconstruction of single updates; per-update
  malignity and stage readouts (supervised); cross-domain (IoT) variation of protocol/TTL/flag mixes.
- **What it cannot learn from them:** transitions (every `next_index` is −1, so no KL dynamics term),
  topology (no shared hyperedges), inter-update timing, entity history, triggers beyond the single one
  at time 0, and therefore no forecasting signal. The dataset's PCAPs (to verify that they are
  published) through `ingest/pcap.py` are the path for temporal learning.
- **Evidence:** code run: `test_timeless_source_has_one_trigger_and_no_transitions`.

### AS-308 Monitored networks per dataset
- **Assumed:** CIC-IDS2017 `192.168.10.0/24` (the firewall's `172.16.0.1`, which NATs the external
  attacker, is external); CSE-CIC-IDS2018 `172.31.0.0/16`; CTU-13 `147.32.0.0/16`. Default when none
  is given: RFC 1918, RFC 4193 ULA, link-local, loopback.
- **Stands for:** detail of AS-18 (infiltration needs "internal").
- **Why:** CTU-13's internal hosts have public addresses; CIC-IDS2017's attack flows show the NAT
  address as source, which an RFC 1918 rule would make internal and therefore "infiltrated" by a DoS.
- **Evidence:** dataset pages and the CTU-13 paper (Garcia et al., Computers & Security 45, 2014) name
  hosts in these ranges (unverified per range); FACTS.md of the sample slice (27 internal addresses
  in 172.31.0.0/16).

### AS-309 Initiator of a flow record
- **Assumed:** CICFlowMeter's source (direction of the flow's first packet) and Argus's reported
  source are the initiator; Argus `Dir` is kept as an audit column.
- **Stands for:** detail (flow adapters' convention: entity_0 initiator).
- **Evidence:** CICFlowMeter README on bidirectional flows (unverified wording). Argus direction
  semantics for `<-`, `<?>` are unverified.

---

## Labels (`data/labels.py`)

### AS-310 Mapping principles
- **Assumed:** a dataset label maps to the ATT&CK tactic of the dominant activity its authors
  describe; techniques only when unambiguous; benign → stage 0; labels the authors did not assign
  (CTU-13 "Background") → unknown (NaN, stage −1); an unknown label string raises.
- **Stands for:** AS-34 (the mapping exists, is explicit and reviewable).
- **Why:** unknown is not benign (D-41's principle for labels); silently defaulting an unseen label
  would corrupt supervision.
- **Evidence:** ATT&CK Enterprise matrix (https://attack.mitre.org/); every entry with an
  "approximate" note is a judgement for review. code run: `test_published_label_sets_are_covered`.

### AS-311 Technique slot table
- **Assumed:** `KNOWN_TECHNIQUES` (24 IDs, append-only) take slots 0…23; any other ID hashes by
  SHA-256 into [24, n_techniques).
- **Stands for:** AS-20 detail (the table is not specified).
- **Why:** stable across processes and checkpoints; fits the tiny preset (32 slots).
- **Evidence:** code run: `test_technique_slots`. Must be reconciled with the Forecaster owner's table.

### AS-312 CIC-IoT-2023 scans are reconnaissance
- **Assumed:** Recon-* and VulnerabilityScan → `reconnaissance` (T1595 family), not `discovery`.
- **Why:** the label does not say whether the scanning device was itself compromised inside the
  network; `discovery` would make it an infiltrated entity (AS-18). The conservative reading asserts
  less.

### AS-313 CTU-13 label tokens
- **Assumed:** rules on the label's tokens (first match wins): CC → C2 (T1071.001 with HTTP, else
  T1071); SPAM, click fraud → impact (T1496, approximate); DDoS → impact (T1498); scan → discovery
  (T1046); P2P, DNS → C2 (approximate); other botnet → C2 without technique; `To-Botnet` flows have
  the responder as actor; Normal → benign; Background → unknown.
- **Evidence:** Garcia et al. 2014 labelling scheme (label grammar from memory, unverified per
  scenario). code run: `test_ctu13_rules`.

### AS-314 Actor role
- **Assumed:** each malicious label names the entity that performs the stage: the initiator by
  default, the responder for CTU-13 `To-Botnet` flows and for the slice's attacker-initiated C2
  packets.
- **Stands for:** detail of AS-18.
- **Why:** infiltration is a property of the *acting* internal entity: a DoS by an external attacker
  must not mark its victim as infiltrated, a bot's beacon must mark the bot.

### AS-315 CSE-CIC-IDS2018 "Infilteration" is discovery
- **Assumed:** stage `discovery`, T1046, actor initiator.
- **Why:** the publisher's scenario is phishing → backdoor → Nmap sweep from the victim, and the
  flow-level label covers the victim's flows in the attack window, dominated by the scan. Label
  noise is documented (Liu et al., IEEE CNS 2022). The PCAP slice labeller separates C2 and scan.

### AS-316 Sample-slice labeller
- **Assumed:** rules from FACTS.md: victim ↔ attacker from 14:45:40 → C2 (T1571); victim-initiated
  TCP to 172.31.69.1…23 from each target's first SYN → discovery (T1046); the victim's traffic with
  131.202.242.193 → benign (FACTS: "treat as benign background"); any other victim traffic from
  14:45:40 → unknown; everything else → benign.
- **Evidence:** code run: `test_slice_labeller_on_real_packets`; on 14:45:30–14:46:40 of the slice
  (4,406 packets): 62 backdoor-C2, 1,500 scan, 345 victim-unannotated, 2,499 benign packets.

---

## Windows (`data/windows.py`, `data/collate.py`)

### AS-317 Window plan
- **Assumed:** non-overlapping windows over the time-sorted log, greedy: at most `window_updates`
  updates and `max_entities` entities (initiator, responder, service); the plan depends only on
  order and entity counts.
- **Stands for:** detail of TrainingConfig (window_updates, max_entities).
- **Finding:** code run (script on the 14:45:30–14:46:40 cut of the slice, L `window_updates` =
  1024, packet updates): the five windows span 2.6–23.8 s, and four of five hold no trigger at the
  60 s cadence. **Resolved by the owner's D-51** (flow-state updates + consecutive windows with a
  carried Environment); the measurement under D-51 is in AS-332.

### AS-318 Triggers on the epoch grid
- **Assumed:** triggers at k·c (c = `window_seconds`) epoch seconds inside [t_first, t_last]; none
  if the window is shorter than the gap to the next grid point. Priority triggers (AS-12) are added
  at inference.
- **Stands for:** AS-12 detail.
- **Why:** windows of one stream share trigger instants, like a deployed wall-clock cadence; no input
  volume can move a trigger.

### AS-319 Infiltration label horizon
- **Assumed:** `entity_infiltrated_at` = the first time over the whole source at which the entity is
  the internal actor of a malicious update in an infiltration stage, kept if ≤ t_last + K·c
  (K = `horizon_k`), else +inf; may be negative.
- **Stands for:** AS-18 detail ("within the window's label horizon").
- **Why:** P_inf(k) forecasts K cadence steps past a trigger; targets must reach that far. Labels are
  not inputs, so looking ahead is not leakage.
- **Open:** the censoring time (horizon end) is not in `LabelBatch`; requested field
  `label_horizon: float64 [B]`.

### AS-320 Out-of-order augmentation
- **Assumed:** t'_i = t_i + U(−r_i, r_i) with r_i the recorded reorder uncertainty, then a stable
  re-sort; all structure is rebuilt from t'. Training windows only, seeded per (seed, epoch, index).
- **Stands for:** §4b.5 detail ("permutes updates within their recorded reorder_uncertainty_s").
- **Why:** positions are ordered by time, not index (D-49), so permuting indices without moving times
  would change nothing; moving times within the uncertainty is the effective permutation.
- **Evidence:** code run: `test_out_of_order_augmentation_within_uncertainty` (|t' − t| ≤ 8 s).

### AS-321 Malicious share
- **Assumed:** over the window's updates with t ≤ τ_m in which the entity is initiator or responder,
  with a known label: share = mean of malicious; NaN if none. "Trailing window" = from the window
  start to the trigger, as `testing/synthetic.py`.
- **Stands for:** §4b.4 detail.

### AS-322 Internal flag
- **Assumed:** the adapter's `internal` column when present; else host ⇒ internal, external ⇒ not;
  else the address in the key by the default internal rule (AS-308).

### AS-323 Column slots
- **Assumed:** `COLUMN_SLOTS` (49 columns, catalogue order of 2026-10-02, histogram bins expanded) is
  append-only; a column's slot is its index; `FieldIndex` reserves the rest of `n_slots`.
- **Stands for:** P-22 (stable slots) detail.
- **Evidence:** code run: `test_field_slots_are_stable_and_fit` (49 ≤ 128 for tiny and L).

### AS-324 Batch shapes
- **Assumed:** U = `window_updates` fixed, P = 2U, V and M = the batch maximum (M ≥ 1).
- **Why:** fixed U keeps shapes stable; V at `max_entities` would make the float64 contact tensors
  [B, V, V, planes] needlessly large (at V = 4,096: 0.8 GB per window for `contact_planes`).

---

## Sampling and splits (`data/sampling.py`)

### AS-325 Class balance by dominant family, inverse frequency
- **Assumed:** w_i = 1/(F·n_f(i)) over the dominant family of each window; `power` < 1 interpolates
  to uniform.
- **Stands for:** §4b.6 detail.
- **Evidence:** code run: `test_sampler_frequencies_are_balanced_and_seeded` (each family within
  2 points of 1/F over 20,000 draws). Cui et al., CVPR 2019 (arXiv:1901.05555) for the
  effective-number alternative.

### AS-326 Chronological stratified split
- **Assumed:** within each (network, family) stratum, earliest windows → train, then test, then val
  (60/20/20; pretraining 70/30).
- **Why:** temporal snooping (Arp et al., USENIX Security 2022) — a random split puts test windows
  between training windows of the same sessions.

### AS-327 Purge at the cuts
- **Assumed:** one window after each cut inside a stratum is excluded.
- **Why:** adjacent windows share sessions and entity histories; one window of separation is the
  smallest purge (López de Prado, *Advances in Financial Machine Learning*, Wiley 2018, ch. 7).

### AS-328 Zero-shot by any family
- **Assumed:** a window is zero-shot if its network is held out or *any* family in it is novel.
- **Why:** a window with one novel-family update would otherwise put that family in training.
- **Evidence:** code run: `test_full_split_ratios_chronology_and_zero_shot`.

### AS-329 Test split validated as validation
- **Assumed:** `pipeline.splits.Split` has no TEST; test rows are checked as VAL rows (real; families
  seen) plus "test is real" here. Requested change: add `Split.TEST`.

### AS-330 Generated windows pair with real windows by position
- **Assumed:** a generated source is a record-for-record variant of a real source (`derived_from` =
  the real source id); its window starting at update k derives from the real window starting at k.
  A variant joins train only if that real window is in train and it holds no novel family;
  otherwise it is excluded.
- **Why:** §4b.7 (generated only in train) and P-23 (no leakage), checked by
  `pipeline.splits.validate` with the generator-no-leakage rule on.

### AS-331 Window family
- **Assumed:** the most frequent family among the window's malicious updates; "benign" if it has only
  benign updates; "unknown" otherwise.

---

## Stream order and flow-state emission (D-51)

### AS-332 Active timeout of the flow-state mode: 1.0 s
- **Assumed:** `ACTIVE_TIMEOUT_S = 1.0`: an open flow emits an update on the first packet at least
  1 s after its previous update; the idle timeout stays 120 s (the adapter's flow split).
- **Stands for:** a parameter of D-51 (the owner named the trigger, not its value).
- **Why a value far below NetFlow / IPFIX practice:** exporters use active timeouts of minutes
  (Cisco's NetFlow default is 30 minutes, to verify) to bound export volume for collectors. Here the
  update *is* the model's observation: a long flow's state (bytes, IAT, periodicity of a C2 channel)
  reaches TSTCT and the triggers only through updates, so the timeout bounds how stale a flow's
  state can be at a trigger. 1 s is 1/60 of a cadence step and 1/300 of the causal lag Λ_lag
  (300 s), so staleness is negligible against every time scale the model reasons at, and the
  measured cost of going that low is small (below).
- **Evidence:** code run (scripts `measure_slice.py` / `measure_stream.py` on the whole sample slice,
  270,596 packets, 50 min, L window of 1,024 updates):

  | emission | updates | windows | span median / p10 / p90 (s) | baseline rate (/s) | scan rate (/s) |
  |---|---:|---:|---|---:|---:|
  | packet | 270,596 | 265 | 6.2 / 0.3 / 32.2 | 72.8 | 110.4 |
  | flow-state, active 0.25 s | 161,438 | 158 | 9.8 / 1.6 / 50.4 | – | – |
  | flow-state, active 1 s | 153,935 | 151 | 9.7 / 1.5 / 60.2 | 15.2 | 91.6 |
  | flow-state, active 60 s | 136,230 | 134 | 9.2 / 1.5 / 64.0 | 10.5 | 84.4 |

  Going from 60 s to 1 s costs +13 % updates; from 1 s to 0.25 s, +5 %. The backdoor C2 session gets
  37 updates at 1 s against 25 at 60 s (median gap 48 s vs 71 s).
- **What the measurement also shows:** flow-state emission cuts the baseline rate about 4.8× but the
  internal scan only 1.2× (each probed port is its own 1–2-packet flow: start, then end or
  idle-end), so during an attack a 1,024-update window still spans only seconds. What makes triggers
  reachable is part (2) of D-51: in stream order every one of the slice's 49 cadence grid points
  belongs to exactly one window (AS-335), and the carried Environment gives that window the earlier
  context. Windows holding at least one trigger: 18 % (packet), 30 % (flow-state, 1 s).
- **Idle-end updates (resolved by D-53):** 40,618 of the 153,935 updates (26 %) are idle-end updates
  of flows with no packet since their previous update (mostly unanswered scan SYNs, reported 120 s
  later). Since D-53 they carry `flow.end_reason` = idle and `flow.unanswered` (AS-337, AS-338).

### AS-333 Segments of at least K·c
- **Assumed:** consecutive windows are grouped into segments spanning ≥ `segment_seconds`, default
  K·c (`horizon_k · window_seconds`; 720 s at L). Segments are the unit of splitting and of
  class-balanced sampling; the carry resets at each segment start.
- **Why:** carry must not cross split boundaries (that would feed test traffic into training
  memory); a segment of at least K·c holds at least K triggers, and with a purge of one segment the
  K-step label look-ahead stays inside the purge gap.
- **Evidence:** on the slice at L (flow-state, 1 s): 4 segments of 10, 13, 71 and 57 windows. A
  50-minute capture gives few segments; full days give hundreds. code run:
  `test_segments_split_and_validate`.

### AS-334 Label limit
- **Assumed:** a record's label look-ahead (AS-319) stops at the start of the next record of the same
  source in another split; generated records take the limit of the real record they vary.
  `LabelBatch.label_horizon` = min(last update or trigger + K·c, limit) − origin, and infiltration
  times beyond it are +inf (right-censored at the horizon).
- **Why:** labels of a test period must never be training targets, even as look-ahead.
- **Evidence:** code run: `test_label_limits_stop_at_another_split`, `test_label_limit_censors_infiltration`.

### AS-335 Stream trigger ranges and carry bookkeeping
- **Assumed:** window k's triggers are the grid points in [t_first(k), t_first(k+1)); entity keys are
  the source's entity rows (stable across windows); `carried_keys` are the entities seen earlier in
  the segment; `origin_shift` converts carried relative times.
- **Why:** each grid point is computed once, by the window whose updates precede it, so no trigger
  sees a later update and none is lost between windows.
- **Caveat:** `entity_latest` covers the window's own entities. An entity seen only in earlier
  windows is not a TAAFT token at this window's triggers unless the model adds carried entities
  (TSTCT/TAAFT side, engineer B).
- **Evidence:** code run: `test_trigger_ranges_partition_the_grid`, `test_lane_loader_carry_bookkeeping_and_contract`.

### AS-336 Flow-state emission details
- **Assumed:** flag change = a flag bit (including ECE, CWR) new *in the packet's direction*; the
  active timeout is checked on packet arrival and only while the flow is open; a closed flow emits
  idle-end only if packets arrived after its flow-end update; capture-end comes at the time of the
  capture's last record, for flows with packets not yet in an update; `app.payload` of an update
  aggregates its packets (OBSERVED > NOT_OBSERVABLE > NOT_SUPPLIED; digest of the last readable
  payload); provenance = the triggering packet's raw record (idle-end and capture-end: the flow's
  last packet).
- **Stands for:** details of D-51.
- **Evidence:** code run: `tests/test_ingest_pcap_flowstate.py` (each trigger; fields equal to the
  packet mode's at the same packet, cell for cell; identity tables equal in both modes).

## Flow end as evidence (D-53)

### AS-337 `flow.end_reason`: codes, and "still open" as NOT_SUPPLIED
- **Assumed:** CATEGORICAL codes 1 fin (FIN from both sides), 2 rst, 3 idle, 4 capture_end; 0 is never
  written. While a flow is open the field is NOT_SUPPLIED, not a "none" code. The first cause is
  kept (a flow reset after one FIN is rst; packets after a close keep the close's reason). In packet
  mode only fin and rst can appear, from the closing packet on (idle and capture ends have no packet,
  so packet mode has no update to carry them). CSV adapters: NOT_SUPPLIED (CICFlowMeter flag counts
  and Argus `State` letters show flags seen, not which event ended the flow; Argus may also split
  flows by its own timer: unverified).
- **Stands for:** encoding detail of D-53.
- **Why NOT_SUPPLIED for open flows:** an end reason does not exist before the end, which is how the
  PCAP adapter already treats statistics that do not exist yet (a variance before the second
  inter-arrival). A "none" code would be an observed value claiming something about the end that has
  not happened. The flow's other fields being observed tell the model it is a flow sensor, so the
  status of this field reads as "not ended yet" for PCAP sources. Codes start at 1 so that no code
  can be mistaken for an absent value written as 0 (D-41 spirit).
- **Evidence:** code run: `test_each_end_reason_and_unanswered` (fin, rst, idle, idle, rst,
  capture_end on six crafted flows; NOT_SUPPLIED on the open flows' first updates; packet mode fin/rst
  on the closing packets only).

### AS-338 `flow.unanswered` as its own field
- **Assumed:** a separate CATEGORICAL field, 1 if the flow ended without the responder ever sending a
  packet, else 0; OBSERVED exactly when `flow.end_reason` is (PCAP), NOT_SUPPLIED while the flow is
  open (the responder may still answer). CICFlowMeter: `Tot Bwd Pkts` = 0 → 1 (within CICFlowMeter's
  flow delimitation, which may split long flows: unverified); CTU-13: `TotBytes` = `SrcBytes` → 1
  (no byte from the destination means no packet from it).
- **Stands for:** encoding choice left open by D-53 ("its own code or a separate field").
- **Why a separate field, not an end-reason code:** "unanswered" is orthogonal to the cause. An
  unanswered flow can end by idle (a SYN to a filtered port), by the initiator's own RST, or at the
  end of the capture; one code would erase the cause, and a cross-product of codes would split the
  same evidence over several embedding rows. Two fields give two learned embeddings that the
  attention pool combines, and the CSV sources, which know "unanswered" but not the cause, can supply
  one without the other. CATEGORICAL with two values (not a 1-bit BITMASK) gives "answered" its own
  learned row instead of an all-zero bit sum.
- **Evidence:** code run: `test_each_end_reason_and_unanswered` (an unanswered SYN → (idle, 1); an
  answered RST → (rst, 0); the initiator's own RST without reply → (rst, 1)); `test_data_csv.py`
  (CIC one-packet SYN → 1; CTU ARP with no reply bytes → 1).
