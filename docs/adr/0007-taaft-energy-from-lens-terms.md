# ADR-0007: TAAFT's energy is the sum of its lens terms

- **Status:** Accepted (owner, 2026-09-30)
- **Decision IDs:** D-42; partly settles D-11b; related: P-09, D-24, D-26, D-37
- **Sources:** owner, 2026-09-30, chat; [A-19]; refs.md#L1031 (energy-based transformers);
  Hinton, "Training products of experts by minimizing contrastive divergence", Neural Computation
  14(8), 2002; Du, Li and Mordatch, "Compositional Visual Generation and Inference with Energy Based
  Models", NeurIPS 2020 (arXiv:2004.06030)

## Context
The owner lists the energy-based transformer as TAAFT's core and the other analyses (belief and
trust, game theory, information theory, topology, time series, causal inference) beside it [A-19].
The code and the brief treated "energy" as one lens among the others. That left open how the lenses
and the energy relate, and D-11b (which energy jobs are in v1) could not say what TAAFT's own energy
is.

## Decision
The lenses are terms of the energy. TAAFT's total energy is the sum of the lens energies plus the
physics term:

    E_total = E_belief-and-trust + E_game + E_information + E_topology + E_time + E_cause + λ·Φ_phys

Refinement descends E_total.

## Consequences
- A sum of energies is a product of the matching distributions (a product of experts, Hinton 2002).
  Du, Li and Mordatch (NeurIPS 2020, around their Equation 4) compose concepts this way: "the
  likelihood of an output given a set of specific concepts is equal to the product of the likelihood
  of each individual concept, we have Equation 4, which is also known as the product of experts
  [Hinton, 2002]". A low E_total means the lenses jointly find the candidate
  compatible. It does not prove that each lens does: a very low term can offset a high one, which is
  one more reason to report the terms separately (next point).
- Explanations come from the decomposition. Each refinement step moves along
  −∇E_total = −Σ ∇E_lens − λ·∇Φ_phys, so each lens's share of the step, and of E_total, can be
  reported. This supports the problem statement's "driving features" requirement.
- Because the lens terms are learned, the formula puts no fixed weights on them. Each term's scale is
  learned, and a constant offset in one term cannot be told apart from an offset in another. Per-lens
  shares are therefore read relative to each term's own reference level (e.g. its value on benign
  traffic), not as absolute numbers.
- `energy_descent` (`models/taaft/energy.py`) already takes any energy callable, and its docstring
  form E_tot = E_θ + λ_phys·Φ_phys matches D-42 with E_θ = the sum of the lens terms. In
  `models/taaft/lenses.py` the `energy` entry now names the sum, not a seventh term; its docstring
  should say so when that file is next edited.
- `conf/model/taaft/taaft.yaml` lists the six terms under `energy.terms` and leaves λ as `???`.
- Not decided: λ; whether the held lenses mechanism design (D-24) and noise (D-26) add terms of their
  own or sit inside E_game and E_information; whether Advisor shaping and a novelty signal are v1
  energy jobs (the rest of D-11b); how the energy is conditioned (proposal P-09, unchanged).
