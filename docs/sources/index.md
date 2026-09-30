# Sources of truth

This codebase implements a design that lives in documents. When code and a document disagree, the
document wins until the code is updated. Every disagreement found should be logged in
`DESIGN_LOG.md` (workspace root).

| Source | Where | What it governs |
|---|---|---|
| `CLAUDE.md` | workspace root | Mandate (the problem statement), working rules: evidence, no faking, no pruning of ambition |
| `ai-mod-arch` | workspace root | The owner's component and role design; current names (Simulator, Forecaster, Advisor, Verifier; CVG-AE, TSTCT, TAAFT); training pipeline |
| `ARCHITECTURE.md` | workspace root | Consolidated design rationale (§1–§17) and the owner's ideas #1–#26, including the tech stack (#18) |
| `DESIGN_LOG.md` | workspace root | Living record: §2 the owner's held decisions, §3 engineering proposals, dated log |
| `datamodel.md` | workspace root | Data-model objectives: Kafka-compatible superset, passive only, OCSF/CSTS |
| `refs.md` | workspace root | Research taxonomy (47 parts). Code cites it as `refs.md#L<line>` |
| `docs/sources/quotes.md` | private workspace (not published) | The owner's statements, quoted verbatim with IDs (Q-, A-, I-) |
| `docs/decisions.md` | this repo | Generated from `nagahana.governance.decisions`: decided / held / proposed, and the code that depends on each |
| `docs/adr/` | this repo | Architecture decision records for choices that shape the code |

## Citation conventions used in code comments
- `[Q-25]`, `[A-12]`, `[I-01]`: the owner's words, from `quotes.md`.
- `D-12`: a decision in the registry (`governance/decisions.py`). It is either DECIDED, HELD or PROPOSED.
- `P-18`: an engineering proposal, not approved. Code behind it runs only when the proposal is enabled in config.
- `ARCH §5.2` / `ARCH #18`: `ARCHITECTURE.md` sections, or the owner's numbered ideas in it.
- `refs.md#L2423`: an entry in the research taxonomy.
- A paper cited by name and year was checked against its public record before being cited. The check date appears in `DESIGN_LOG.md`.
