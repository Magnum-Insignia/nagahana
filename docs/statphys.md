# Statistical physics of the network state (D-56)

NagaHana's analyser, TAAFT, refines its beliefs by descending an energy, E_total = sum of the lens
energies + lambda * Phi_phys (D-42). This module reads that energy, together with the traffic and the
multiplex graph of the network, as a thermodynamic trajectory. It also watches those trajectories for
the generic signs of an approaching transition (critical slowing down). Architecture section 6 lists the
outputs it provides:
- "Energy per entity and per time";
- "Energy and entropy growth as an early warning";
- "Novelty (high energy means unfamiliar, not proof of attack)".

Code: `src/nagahana/statphys/`. Assumptions: `docs/assumptions/statphys.md` (AS-760 to AS-779).
Configuration: `conf/statphys/statphys.yaml`, generated from the dataclasses in
`statphys/config.py` (the single source of truth).

## 1. Thermodynamic readouts (`statphys/thermo.py`)

For an ensemble of members i with energies E_i, base measure g_i (multiplicity, default 1) and
temperature T:

    a_i = -E_i / T + log g_i,       log Z = logsumexp_i a_i,       p_i = exp(a_i - log Z)
    F = -T log Z                    free energy (novelty: low Z means every member is unfamiliar)
    U = sum_i p_i E_i               mean energy
    S = (U - F) / T = log Z + U / T Gibbs entropy (Shannon entropy of p when g = 1)
    C = Var_p(E) / T^2 = dU/dT      heat capacity
    chi_O = Var_p(O) / T            susceptibility of an observable O (response to E -> E - h O)
    d<O>/dT = Cov_p(O, E) / T^2     thermal response of the Gibbs mean of O

All identities are checked against finite differences, and the Schottky two-level system against its
closed form (tests/test_statphys_thermo.py). Growth rates are least-squares slopes per second over the
last `growth_window` valid points (AS-763).

Ensembles read at every trigger (AS-760, AS-779):

| ensemble | members | energies | where |
|---|---|---|---|
| entities | active entity tokens | TAAFT per-token energies at y_hat | TAAFT readouts, engine |
| slots | active adversary-hypothesis slots | same | TAAFT readouts, engine |
| stage (per entity) | the 15 stage classes | -logit_s | TAAFT readouts, engine |
| routes | the N imagined routes (each draw a microstate) | sum_k E(null, y_hat_{n,k}) | engine |
| imagined (per entity) | imagined steps that target the entity | E(null, y_hat_{n,k}) | engine |

Temperature (AS-761): TAAFT's readouts use the native T = 1. The engine uses the Verifier temperature in
force for the family `gibbs.verifier_family` (default `p_inf`), which changes only under a human command
(D-21).

## 2. Network entropies

Shannon entropies (`statphys/entropy.py`, AS-764 to AS-766). They are read over the state window
(tau - 300 s, tau], per entity and over the network, for these distributions:
- ports, protocol, TCP flag combination and packet-weighted flags;
- peers (per entity), and initiator and responder entities (over the network).

Estimators are plug-in, Miller-Madow (default) and Chao-Shen. Only contributing cells count (D-41); a
distribution without samples has no entropy (NaN). The Jensen-Shannon divergence between the
distributions of successive triggers measures how fast the traffic mix moves.

Graph entropies (`statphys/spectral.py`, `statphys/graphs.py`, AS-767 to AS-771):
- the activity multiplex of the state window, one layer per relation plane (AS-01), with the star
  semantics of D-52;
- von Neumann entropy of rho = L / tr L (Braunstein, Ghosh and Severini 2006), per plane and for the
  aggregate;
- spectral entropy of rho_tau = exp(-tau L) / Z (De Domenico and Biamonte 2016), with its free energy,
  mean and heat capacity, at tau = 0.1, 1, 10;
- the quantum Jensen-Shannon divergence between planes and between successive states;
- the relative entropy q of the multiplex and its reduction by hierarchical clustering of the planes
  (De Domenico, Nicosia, Arenas and Latora 2015).

Connected components of up to 512 nodes are diagonalised exactly. Larger ones use stochastic Lanczos
quadrature (Ubaru, Chen and Saad 2017), with exact zero-mode deflation, low-mode deflation, a polynomial
control variate and adaptive probe and depth control. Its standard errors are reported with every
estimate.

## 3. Early warning (`statphys/ews.py`, AS-772 to AS-778)

For each monitored series (`ews_series`), sampled on the cadence triggers:
1. causal Gaussian detrending (two-sided for offline analysis);
2. rolling indicators over 60 samples: variance, lag-1 autocorrelation, skewness, kurtosis, return rate
   -ln(rho_1) / dt, spectral ratio and exponent, DFA exponent;
3. Kendall tau-b trends;
4. offline significance against phase-randomised and AR(1) surrogates;
5. composite level and trend scores;
6. an alarm at a split-conformal threshold calibrated on benign trajectories (`statphys calibrate`). The
   target rate of 0.01 per trigger is shared over the series. Without a calibration file no alarm is
   raised; a threshold is never defaulted.

Every streaming form (`StreamingEWS`, `SlidingKendall`, `TrafficEntropyTracker`, `SlidingMultiplex`)
equals its batch form. Its work per update does not grow with the stream (tested).

## 4. Wiring

TAAFT readouts (`models/taaft/readouts.py`, `models/taaft/model.py`): every trigger adds
`thermo/total_energy`, `thermo/entities/*`, `thermo/slots/*` (log_partition, free_energy, mean_energy,
entropy, heat_capacity, size, valid), `thermo/occupation` and `thermo/stage/*`. All are float64.
Stacked over the triggers of a call, they are the energy, entropy and free-energy trajectories. No
parameter or buffer is added.

Inference engine (`inference/engine.py`): `EngineSettings.statphys` configures it (enabled by default).
Every `TriggerResult` carries `statphys: StatPhysReading` next to the forecast (P_inf). The reading holds:
- the ensembles, the route response and the per-entity readings;
- the traffic and graph readings;
- the flat `series`, their `growth` rates, the early-warning steps and the alarm.

Evaluation (`statphys/evaluation.py`): `component_arrays(readings)` fills `ModelOutputs.component` with
the `statphys.*` names listed in that module. `window_component_arrays` serves the batch path. Scores:
- `novelty_auroc` (Mann-Whitney);
- `alarm_lead_times`;
- `false_alarm_rate`;
- `early_warning_report`.

Command line (`statphys/cli.py`, `register_cli`): `python -m nagahana statphys` with the subcommands
`write-config`, `check-config`, `show`, `ews`, `calibrate` and `alarm`. They work on trajectories saved by
`statphys/io.py`, or on the statphys component arrays of a ModelOutputs file.

## 5. References

- Gibbs ensembles and fluctuation-dissipation:
  - Callen, Thermodynamics and an Introduction to Thermostatistics, 2nd ed., Wiley 1985.
  - Kubo, Reports on Progress in Physics 29:255, 1966.
- Energy-based novelty:
  - Liu, Wang, Owens and Li, NeurIPS 2020, arXiv:2010.03759.
  - Grathwohl et al., ICLR 2020, arXiv:1912.03263.
- Traffic entropy: Lakhina, Crovella and Diot, SIGCOMM 2005.
- Entropy estimators:
  - Miller 1955.
  - Paninski, Neural Computation 15:1191, 2003.
  - Chao and Shen, Environmental and Ecological Statistics 10:429, 2003.
- Graph entropies:
  - Braunstein, Ghosh and Severini, Annals of Combinatorics 10:291, 2006.
  - Passerini and Severini 2009.
  - De Domenico and Biamonte, Physical Review X 6:041062, 2016.
  - De Domenico, Nicosia, Arenas and Latora, Nature Communications 6:6864, 2015.
- Quantum Jensen-Shannon divergence:
  - Lamberti et al., Physical Review A 77:052311, 2008.
  - Virosztek, Advances in Mathematics 380:107595, 2021.
- Stochastic trace estimation:
  - Hutchinson 1989.
  - Ubaru, Chen and Saad, SIAM J. Matrix Anal. Appl. 38:1075, 2017.
  - Avron and Toledo, J. ACM 58:8, 2011.
  - Meyer et al., SOSA 2021, arXiv:2010.09649.
- Critical slowing down:
  - Wissel, Oecologia 65:101, 1984.
  - Scheffer et al., Nature 461:53, 2009.
  - Dakos et al., PNAS 105:14308, 2008.
  - Dakos et al., PLoS ONE 7:e41010, 2012.
  - Held and Kleinen 2004.
  - Kleinen, Held and Petschel-Held 2003.
  - Biggs, Carpenter and Brock 2009.
  - Guttal and Jayaprakash 2008.
  - Peng et al. 1994.
  - Livina and Lenton 2007.
- Surrogates:
  - Theiler et al., Physica D 58:77, 1992.
  - North, Curtis and Sham 2002.
- Conformal thresholds:
  - Vovk, Gammerman and Shafer 2005.
  - Angelopoulos and Bates, arXiv:2107.07511.
