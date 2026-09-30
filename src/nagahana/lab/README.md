# `nagahana.lab`: the JAX + Equinox research bench

"this way we can keep track of things, validate, test & analyze, conduct analytics on data using our
approaches, do statistical testing, do the jax & equinox operations to test for things like this;
test for physics simulations" [Q-40]. "use various analysis, do jax & equinox, see what the model
must see, make it happen" [Q-44].

The lab is **isolated** from the core. Nothing in `nagahana` imports from `nagahana.lab`, and the lab
imports JAX lazily. Experiments here cannot change production behaviour. Results reach the core only
through a reviewed change.

## Planned modules (proposals, awaiting approval)
| Module | Proposal | What it answers |
|---|---|---|
| `world_sim.py` | P-14 | A ground-truth world: simulated IT+OT networks where the hidden state (attacker position, stage, intent) is **known** and observation is modelled explicitly (taps, encryption, NetFlow-only sampling). Public datasets have label errors and no hidden-state truth, so belief and forecast quality cannot be measured on them alone. |
| `info_audit.py` | P-15 | "See what the model must see": for each technique and observation regime, how much do the observables reveal about the hidden stage? On the simulator this gives a Bayes-optimal ceiling; NagaHana is judged by its gap to it. |
| (next) mechanistic probes | P-16 | Probes on latents, and interventions (drop a plane, perturb a feature). |

## Why JAX here
- `vmap`: thousands of simulated networks in parallel.
- `grad`: sensitivity of outcomes to inputs.
- `jit`: fast fixed-shape maths.
- Diffrax/Optimistix: continuous-time dynamics and constraint solves (approved with D-09).

The dynamic, event-driven core stays in PyTorch [Q-35].
