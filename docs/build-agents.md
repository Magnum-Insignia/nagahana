# Rules for every engineer (agent) building NagaHana L (2026-10-02)

Read first, in this order:
1. `docs/build-spec.md`: the design (philosophy → logic → maths → contracts). It is binding.
2. `docs/architecture.md`, and ADRs 0007 and 0008 in `docs/adr/`.
3. `src/nagahana/models/batch.py` (tensor contracts), `src/nagahana/models/config/` (configs),
   `src/nagahana/models/vocab.py` (codes), `src/nagahana/nn/` (shared primitives: use them, do not
   re-implement attention or blocks), `src/nagahana/governance/assumptions.py` (AS-01…AS-41) and
   `src/nagahana/governance/decisions.py` (D-xx; D-49 and D-50 are new).
4. `src/nagahana/testing/synthetic.py`: `make_window_batch(preset("tiny"))` gives a consistent
   synthetic `WindowBatch` + `LabelBatch` for your tests.

## Hard rules
- **Scope:** edit only the files your brief assigns to you.
  - If you need a change elsewhere (`nn/`, `batch.py`, `vocab.py`, `config/__init__.py`), do not make
    it; put it in your final report as a requested change, with the exact diff.
  - You MAY add fields (with an L default and a comment) to *your own* component's dataclass in
    `models/config/components.py`. Touch nothing else in that file.
  - You may add your component's tiny overrides only by reporting them.
- **Vocabulary:** say "state", "state update", "transition", "entity state". Never "token" for
  network states. "Token" is acceptable only for an attention position of a transformer sequence
  when unavoidable; prefer "position" or "slot".
- **No new dependencies.** PyTorch, NumPy and pandas only. No pip installs.
- **No silent defaults on held decisions.** Where the design needs something undecided, use an
  existing assumption (`from nagahana.governance.assumptions import assume`; call
  `assume("AS-xx", by=__name__)` where the code relies on it).
  - If you need a new assumption, take the next ID from your range.
  - Write it to `docs/assumptions/<your-area>.md` with: ID, what is assumed, which held decision or
    detail it stands for, reasoning, and evidence.
- **Citations must be real.** Cite only papers you are sure of (authors, venue/arXiv ID). If you are
  not sure of an identifier, write "(citation to verify)" instead of guessing. Never invent a
  result. A finding from your own code run is cited as "code run: <test name / script>, <number>".
- **Documentation standard (ADR-0001):**
  - every module docstring states purpose, owner sources (Q-/A- IDs if known), decisions (D-),
    assumptions (AS-), the maths (equations), invariants, and extension points;
  - **every non-trivial block gets a comment** saying what it does and why;
  - shapes are written in comments, e.g. `# [B, H, P, d_h]`.
- **Engineering:**
  - modular and decoupled; register implementations in the existing registries where one exists;
  - type hints everywhere;
  - `ruff check` and `mypy` clean on your files (project settings in `pyproject.toml`);
  - no `print` in library code.
- **Numerics:** time in float64 seconds relative to the window origin; softmax, norms and energies in
  float32; never let NaN from an excluded cell reach arithmetic (mask before computing).
- **Tests** go in `tests/test_<area>_*.py` (pytest + hypothesis allowed), at `preset("tiny")`.
  - Test maths against closed forms where they exist.
  - Test invariants: no future leakage, permutation equivariance, monotonicity, provenance, gating.
  - Test that gradients reach every trainable parameter that should get one.
  - Run them: `python -m pytest tests/test_<area>_*.py -q`. Run the full suite at the end
    (`python -m pytest -q`) to check you broke nothing.
- **Never** run git commands that change state (no commit, no push, no checkout). Never contact the
  network.

## Final report (your last message)
1. The files created or changed, each with a one-line purpose.
2. The public API: class and function signatures other components will call.
3. The tests: names, what they prove, and the pass count (paste the pytest summary line). ruff and
   mypy results.
4. New assumptions (IDs and one line each), and anything you could not do or left as a template,
   with the reason.
5. Requested changes to files outside your scope, as exact diffs.
6. Honest caveats: anything approximate, unverified or risky.
