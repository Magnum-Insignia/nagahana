"""Data pipeline: telemetry files → state updates → windows → (WindowBatch, LabelBatch).

README
======

What this package does
----------------------
It turns open datasets and captures into exactly what the model trains on (build-spec §1.1, §2.2,
§3, §4b items 5–7), with labels kept apart from observations at every step.

End to end
----------
1. **Ingest** (`nagahana.ingest`, not this package). An adapter reads one file into the event log
   `ColumnarUpdates` (one row = one state update, D-30; absent fields NOT_SUPPLIED, never 0, D-41):
       PCAP            `ingest.pcap.PcapSource(path, sandboxed=…).columnar()`
       CIC-IDS2017/18  `ingest.csv_flows.CICFlowSource(path, …).read()`
       CTU-13          `ingest.csv_flows.CTU13Source(path, …).read()`
       CIC-IoT-2023    `ingest.csv_flows.CICIoT2023Source(path, synthetic_internal=…).read()`
   A CSV adapter also returns the raw label table (seq, record, label_raw) and read statistics.
2. **Labels** (`data.labels`, AS-34). `map_labels(raw, dataset)` maps each dataset label to an ATT&CK
   stage, a technique ID (slot via `technique_slot`, AS-20), a malicious flag (1 / 0 / NaN unknown),
   a family (dataset-independent, for balance and known/novel splits) and the *actor role* (who
   performs the stage, for infiltration, AS-18). The sample PCAP slice is labelled by rules from its
   FACTS.md (`label_cic2018_infiltration_slice`).
3. **Sources** (`data.windows`). `SourceData(updates, labels, network, …)` → `prepare_source` sorts
   by event time and gathers per-source arrays once (entity kinds, internal flags, first infiltration
   time of every entity).
4. **Windows and splits** (`data.sampling`). `index_windows` plans non-overlapping windows of at most
   `window_updates` updates and `max_entities` entities and describes them (time span, families,
   network). `assign_splits(records, SplitPolicy.from_config(cfg, …))` makes the manifest:
   60/20/20 train/test/val chronologically within each (network, family), zero-shot = novel families
   and held-out networks (AS-35), generated only in train; 70/30 for pretraining.
   `manifest.validate()` runs the decided rules of `pipeline.splits`.
5. **One window** (`data.windows.build_window`). Field matrices in the canonical column layout with
   stable slots; window-local entity table; 2 positions per update sorted by (time, update, role)
   with `next_index` / `next_dt`; fixed-cadence triggers with `entity_latest`; labels (per update,
   per entity infiltration time, per trigger malicious share); structure from the graph builder
   (`graph.window.build_window_structure`, engineer A).
6. **Stream order (D-51)** (`data.stream`). Training runs on consecutive windows with the previous
   windows' Environment carried as read-only memory. `plan_stream(src, cfg)` gives the consecutive
   windows with trigger ranges that partition the cadence grid; `segment_records` groups them into
   segments (≥ K·60 s, AS-333), which are the unit of splitting and of class-balanced sampling;
   `StreamLoader` walks B lanes of segments window by window and yields (WindowBatch, LabelBatch,
   [StreamContext]); a context carries the stable entity keys, `key_to_index`, `carried_keys`,
   `carried_index`, `reset` and `origin_shift` that TSTCT's carry needs. `sampling.label_limits`
   stops label look-ahead at the next segment of another split (AS-334).
7. **Batches** (`data.collate`, `data.dataset`). `WindowDataset.from_manifest(…)` + `make_loader(…,
   balanced_samples=n)` → DataLoader of (`WindowBatch`, `LabelBatch`), class-balanced (§4b.6).
   `validate_window_batch` checks any batch against the `models/batch.py` contract.

Example
-------
    from nagahana.models.config import preset
    from nagahana.ingest.csv_flows import CICFlowSource
    from nagahana.data import labels, windows, sampling, dataset

    cfg = preset("tiny")
    read = CICFlowSource("Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv", utc_offset_hours=-4).read()
    src = windows.prepare_source(windows.SourceData(read.updates, labels.map_labels(read.labels, "cic-ids2018"),
                                                    network="cic-ids2018"))
    recs = sampling.index_windows([src], cfg)
    manifest = sampling.assign_splits(recs, sampling.SplitPolicy.from_config(cfg))
    train = dataset.WindowDataset.from_manifest([src], manifest, [sampling.Role.TRAIN], cfg, perturb_seed=0)
    for window_batch, label_batch in dataset.make_loader(train, balanced_samples=1000):
        ...

Rules this package keeps
------------------------
- Labels never enter `WindowBatch` (they live only in `LabelBatch`).
- No future leakage in inputs: positions, `entity_latest` and the as-of graphs use times ≤ their own;
  label *targets* may look ahead (the infiltration horizon, AS-319), because they are not inputs.
- Times are float64 seconds relative to each window's origin (D-49); no index-based positions.
- Absence is never zero (D-41): non-contributing cells are NaN with an excluded status.

Assumptions: AS-300 … AS-338 in `docs/assumptions/data.md`.
"""
