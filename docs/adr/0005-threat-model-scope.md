# ADR-0005: Threat-model scope

- **Status:** Accepted (owner, 2026-09-29)
- **Decision IDs:** D-17; related: D-08 (document), P-05 (formal model)
- **Sources:** [A-18], [A-21], [Q-16]

## Decision
Analyst and human feedback is **not** part of the threat model: "you have to trust yourself and your
guardians … if you take that into threat model then you can't trust yourself" (ai-mod-arch). All
telemetry the model receives **is** part of the threat model [A-18].

## Consequences
- Ingest is treated as untrusted. Examples: PCAP uploads are parsed in a sandbox; queue lag and
  backpressure are monitored; clock quality is recorded, not trusted.
- Human feedback enters the Verifier as supplied truth, with an audit trail for accountability.
- Superseded wording: "analyst-loop poisoning" in older documents (ARCHITECTURE.md §13/§14).
