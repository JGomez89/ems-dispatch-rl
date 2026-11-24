# EMS Dispatch Project: Team Work Allocation

## Purpose and scope

This plan assigns the proposal's core work across the four teammates and records
the agreed handoffs, experiment ownership, and checkpoints. The target is to
complete the core project by **December 10, 2026**. Stretch goals are out of
scope unless core work is complete and time remains.

The split balances substantial ownership lanes and concrete outputs; it does
not claim equal hours. Primary ownership stays fixed. Friday checkpoints are
for progress and unblocking, not reassignment.

## Ownership

| Owner | Primary lane | Accountable outputs |
|---|---|---|
| **Justin** | Simulator, environment, and agent interface | Simulator/environment behavior; Gymnasium observation/action interface; runtime dispatch loop; environment acceptance checks; simulator-side map consumption and reach-time behavior. Deliver a stable environment-agent contract and an integrated, tested environment. |
| **Anjali** | Data, calibration, maps, and scenarios | Mine the DataSF dataset and fit the proposal's data-dependent call, severity, service-time, and travel inputs. Generate/configure the core 20-map set, including 10 held-out maps, and deliver documented map, travel, and scenario inputs. Validate E4 scenarios and provide data-specific analysis. |
| **Josh** | PPO agent and training | Implement and train the masked PPO agent and semi-Markov training components. Deliver reproducible policy checkpoints and learning results. Run and interpret E1, lead E2 reward-arm comparisons, provide the PPO side of E3, and supply policy checkpoints for E4. |
| **Adelyn (Addy) Lampley** | Evaluation and report integration | Own the shared evaluation protocol/harness, baseline definitions, common metrics and plots, and evaluation consistency. Own the imitation-learning/behavioral-cloning side and matched comparison for E3; aggregate E4 metrics; integrate and edit the final report. |

Each owner drafts their own technical report section and supplies the evidence
for its claims. Addy is the editor/integrator, not the sole source of technical
results.

## Experiment ownership

| Proposal experiment | Lead and contributions |
|---|---|
| **E1 — Baseline comparison** | Josh runs and interprets the agent-versus-baseline comparison. Addy supplies the shared evaluation infrastructure, baseline definitions, and common metrics. Justin supplies the environment; Anjali supplies calibrated scenarios. |
| **E2 — Reward arms** | Josh leads agent changes, training, and comparison of reward variants. Addy applies the common scoring and reporting protocol. |
| **E3 — PPO versus imitation learning** | Josh supplies the PPO policy and results. Addy owns the imitation-learning/behavioral-cloning side and matched comparison under the shared protocol. |
| **E4 — Generalization** | Anjali prepares and validates held-out map/load/surge scenarios and provides data-specific analysis. Josh supplies the trained policy. Addy runs common metrics and aggregation. |

The evaluation protocol should use the proposal's common settings and report
the specified outcomes, including on-time response and lateness measures,
zone equity, non-closest dispatch and HOLD rates, confidence intervals, and
learning curves. Keep runs comparable by using the proposal's shared maps,
loads, episodes, and seeds.

## Parallel work and handoffs

All four lanes start in parallel against lightweight shared contracts. Josh
may use a temporary stub while Justin's environment is in progress. Do not
wait for the entire data-to-environment-to-agent chain before starting
independent work.

1. **Anjali to Justin:** map configurations and fitted data/travel/scenario
   inputs; Justin owns loading and using them in the simulator.
2. **Justin to Josh:** stable observation/action interface and environment
   behavior; Justin owns the environment-agent integration contract, while
   Josh owns policy learning and training.
3. **Josh to Addy:** reproducible policy/checkpoint artifacts and experiment
   results for evaluation.
4. **Addy to the team:** shared protocol, comparable evaluation outputs, and
   report draft/results integration.

Freeze the key handoff shapes by **Friday, November 27, 2026**: environment
observation/action interface, data/scenario configuration, policy/checkpoint
interface, and evaluation inputs/outputs. Later changes must be communicated
to affected owners. Justin owns environment-agent interface consistency; Addy
owns evaluation consistency.

## Checkpoints and delivery

Provide a short written update each Friday with an artifact/link, status, next
step, and blocker. Use a live discussion only when needed. Checkpoints should
unblock parallel work; primary ownership remains fixed and task redistribution
is not planned.

| Date | Checkpoint |
|---|---|
| **Friday, November 27, 2026** | Readiness and contract-freeze gate: initial fitted-data/map/scenario package; environment/interface smoke test; policy/action and first training smoke test (a stub is acceptable); frozen evaluation protocol and baseline/metric plan. |
| **Friday, December 4, 2026** | Integration/results gate: E1–E4 run status and reproducible outputs; preliminary proposal metrics/figures; complete report draft. If a core result is not ready, state the blocker and a precise finish plan. |
| **Thursday, December 10, 2026** | Final handoff: reproducible core data/config and maps, environment, trained agent, E1–E4 evaluation outputs, metrics/figures, and final report. |

## Proposal boundary

The core work follows the proposal's simulator, PPO, baseline, evaluation, and
generalization plan. Real OSM roads and the other listed stretch goals are
excluded unless the core deliverables are complete and the team has time.
