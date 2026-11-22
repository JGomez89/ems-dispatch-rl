# Interface contract

The agreements that let four people build in parallel. **Change anything here only after
telling the whole team**, then update this file in the same PR as the code.

Open decisions are listed at the end as **D1–D7**. Each one is marked `OPEN DECISION Dn`
in the code at the place it affects. Settle them at kickoff, record the answer here, and
delete the marker.

## 0. Conventions

| Thing | Convention |
|---|---|
| Time | seconds, float. Episode time starts at 0; `State.hour` gives time of day |
| Cells | `(row, col)`, 0–15. Flat index `row * 16 + col` (for `travel_s`). Plots convert with `plot._xy` |
| Units | indices 0–5; action 6 = HOLD (`config.HOLD`) |
| Severity | 0 = critical, 1 = urgent, 2 = non-urgent (`sim.Severity`). Every per-tier list in `params.yaml` uses this order |
| Constants | only in `config/params.yaml`, read through `config.P`. No magic numbers in code |

## 1. Simulator ↔ environment (M1 ↔ M3)

`sim.Simulator` and `stub_sim.StubSimulator` have the same interface:

```python
sim = Simulator(map_name, load, call_seed, surge=False, record=False)
sim.reset(map_name=None, load=None, call_seed=None, surge=None)
step = sim.next_decision()   # DecisionStep(dt_s, reached: list[Reached], done)
state = sim.state()          # State, valid only after a step with done=False
sim.dispatch(unit_idx)       # 0-5, or None = HOLD
```

- `State` and `UnitView` (in `sim.py`) are the only things the env and baselines read.
  Nobody reads `env.sim` internals.
- **Observability.** `UnitView.expected_free_in_s` (from median service times) and
  `State.observed_rate_per_h` are actor-visible. `UnitView.true_free_in_s` and
  `State.true_rate_per_h` are privileged: critic only.
- **Reward timing.** A call produces a `Reached` record when its first unit arrives. Its
  reward is credited in the env step during which that happens (see D2 for discounting).
- **Drain.** After `episode_hours`, no calls arrive. The last `next_decision()` runs the
  queue down and returns every remaining `Reached` with `done=True`.
- **Fixed rules** are listed in the `sim.py` docstring and are the same for every policy.

## 2. Environment API (M3 ↔ M4)

- `EMSDispatchEnv(maps, loads, reward, seed, surge, record, sim_cls)`. `reset()` picks a
  map and load from the lists. `reset(options={"map", "load", "call_seed"})` fixes them (evaluation).
- Observation: `OBS_DIM` floats. The first `N_ACTOR_OBS` are actor-visible and the rest are
  privileged. The exact slice table is in the `env.py` docstring, which is the source of truth.
- `action_masks()`: bool, shape (7,). The rules are in the `env.py` docstring.
- `info`: `dt`, `reached`, `state`, `map`, `load`, `call_seed`.

## 3. Policies

Every policy, learned or not:

```python
policy(obs, mask, info) -> int   # mask[action] must be True
```

Baselines are registered in `baselines.POLICIES`. `evaluate.load_policy(name_or_zip)` returns
one for either a baseline name or a saved model.

## 4. Seeds

- **Call seed** fixes the whole call stream (times, cells, severities, transports, service
  times). All of it is drawn at `reset`, so every policy sees identical calls and service times.
- **Policy/training seed** (`train.py --seed`) controls network init and PPO sampling only.
- Training episodes use call seeds derived from the training seed. Evaluation episode `k` uses
  `call_seed = 100000 + k` (`params.yaml: seeds.eval_call_seed_offset`), which training never uses.

## 5. Outputs

| What | Where | Committed? |
|---|---|---|
| Fitted parameters | `config/params.yaml` | yes |
| λ̄ per map and load | `config/lambda_bar.json` (from `calibrate.py`) | yes |
| Raw DataSF CSV | `data/raw/` | no (download) |
| BC dataset | `data/bc_<teacher>_<n>.npz` | no (team Drive) |
| Training run | `results/<arm>_<reward>_seed<n>/{model.zip, ckpt_*.zip, progress.csv}` | no (team Drive) |
| Evaluation | `results/eval/<policy>_<maps>[_surge].csv`, columns `evaluate.CSV_COLUMNS` | no (team Drive) |

## 6. Workflow

- Python 3.12, packages pinned in `requirements.txt`. Create a venv on each laptop.
- Branch per task (`git switch -c <name>/<topic>`), PR into `main`, one reviewer.
- `python -m pytest` must pass before merging. Tests for unfinished parts skip themselves.
- Edit only the files you own (README table). For a change to someone else's file, ask them.

## Open decisions (settle at kickoff)

| ID | Decision | Options / current placeholder | Affects |
|---|---|---|---|
| D1 | Reward cap | The per-tier cap of 10 makes a critical call cost nothing more once it is ~190 s late, and that happens at 80% load and in E4 surges. Options: Huber-style penalty (quadratic, then linear), or move the cap to the summed step reward. Optional: `(x/σ) + (x/σ)²` to remove the flat start. If changed, update the proposal's reward box. | `rewards.py`, `params.yaml`, `tests/test_rewards.py` (xfail) |
| D2 | Discounting inside a step | Rewards from several calls reached between two decisions: plain sum (current) or each discounted by γ^(arrival − decision time) | `rewards.step_reward` only |
| D3 | "Coverage lost if it leaves its post" | Suggestion: hotspot-weighted share of cells that no other available unit can reach within 240 s | `env.py` obs slot `[5i+4]` |
| D4 | BC teacher | `closest_travel_time` (stronger; current) or `closest_straight_line` | `bc.py` |
| D5 | Equity zones | 4×4-cell blocks → 16 zones (current) | `params.yaml: zones` |
| D6 | Physical details | Road speeds; which posts hold two units; nearest hospital for transport; where a returning unit is when dispatched mid-route (suggestion: its next cell on the path) | `params.yaml`, `maps.py`, `sim.py` |
| D7 | Training loads | Mix of 40/60/80% (current default) or 60% only | `train.py --loads` |
| D8 | γ and its time unit | `ppo_smdp.DT_UNIT_S = 240`: γ is the discount per 240 s. Pick γ (e.g. 0.99) at T19 | `ppo_smdp.py`, `train.py` |
| — | Knee baseline (optional, not in the proposal) | "Closest unless coverage lost > θ", with θ tuned in hindsight. Strengthens the case that RL is needed | `baselines.py` |
