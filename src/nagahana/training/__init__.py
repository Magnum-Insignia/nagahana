"""Training of NagaHana, stages 3–6 (build-spec §3; D-22, D-23, D-51).

- `carry.StreamBridge`: the D-51 carry across consecutive windows (stores, cumulative contacts,
  carried entities, long-term memory) on top of `data.stream.StreamLoader`.
- `stage3`: CVG-AE, Decoder, TSTCT self-supervised (reconstruction, edges, balanced KL on both priors,
  first-state KL, physics, gate L1).
- `stage4`: TAAFT self-supervised with the perceptors frozen and verified by fingerprint.
- `stage5`: TAAFT readouts, Forecaster, Advisor (STAGED coupling); Verifier only under a HumanCommand.
- `stage6`: zero-shot evaluation (real data only, novel and known separately) and human-gated site
  calibration (LoRA adapters + Verifier temperature).
- `variants`: Generator variants mixed into stage-3 training (training split only, never zero-shot).
- `common`: optimiser, precision, checkpoints, logging.
"""
