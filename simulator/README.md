# Simulator

Event-driven EMS dispatch simulator for a 16 × 16 grid city, with a Gymnasium
interface for the PPO agent. Owner: Justin.

The agent's only decision is **which unit to send** when a call arrives (or
HOLD). Everything else follows fixed rules.

**Agent-facing spec: [`agent_interface/CONTRACT.md`](agent_interface/CONTRACT.md)**
(v1.0, frozen Fri Nov 27, 2026). This README covers how the simulator works
inside.

## Quick start

From the repo root:

```bash
pip install -r requirements.txt
pytest                                      # acceptance checks, ~1 s
```

```python
from simulator.environment.dispatch_env import DispatchEnv
env = DispatchEnv()                          # default map, all three loads
obs, info = env.reset(seed=0, options={"load": 0.6})
obs, reward, terminated, truncated, info = env.step(info["closest_road"])
```

Run code from the repo root so the `simulator` package imports.

## Layout

```text
simulator/
├── simulator_config.py   # every number (placeholders marked)
├── engine/
│   ├── city.py           # map JSON -> road graph, all-pairs Dijkstra cache
│   └── sim.py            # discrete-event core
├── environment/
│   ├── dispatch_env.py   # DispatchEnv (real) + reward_d
│   └── mock_env.py       # MockDispatchEnv (random dynamics, same contract)
├── agent_interface/
│   ├── contract.py       # spaces, OBS_SLICES, info keys
│   └── CONTRACT.md       # the agent-facing spec
├── maps/default.json     # hand-made smoke-test map
├── tests/test_env.py     # acceptance checks (plain asserts)
└── simulator.ipynb       # smoke test and sanity plots
```

Dependency direction: `environment` imports `engine` and `agent_interface`.
`engine` and `agent_interface` import only `simulator_config`. The engine never
imports Gymnasium.

Dependencies (`requirements.txt` at the repo root): numpy, gymnasium,
matplotlib (notebook only).

## engine/

| Piece | Spec |
|---|---|
| Map | 16 × 16 grid of intersections, 1/8-mi blocks. Side and main roads per row/column, highways, blocked cells, posts, hospitals, demand. Loaded from JSON (see `maps/`). Validated on load. |
| Travel | Dijkstra on the road graph, all-pairs times and paths cached per map (`City.T`, `City.path`). Straight-line miles in `City.miles` (the straight-line baseline needs it). |
| Moving units | Every drive stores a timetable of `(node, arrival_time)` along its route. A unit dispatched mid-drive first finishes its current edge (no U-turns): `next_node, t_arrive` = first timetable entry with `t_arrive >= now`, `ETA = (t_arrive - now) + road_time[next_node][call]`, and the new route starts from `next_node` at `t_arrive`. Pessimistic, no position snapping, and the predicted ETA equals the real arrival (tested). |
| Fleet | 6 ambulances, 4 posts, 1–2 hospitals. Unit → post assignment comes from the map file. |
| Calls | Poisson, λ(t) = λ̄ · h(hour), piecewise constant per hour. Location drawn from the map's demand grid. 24 hourly weights, λ̄ (for 40/60/80% load) and severity shares π_s are fitted inputs from Anjali. |
| Service | 60 s turnout (only when dispatched idle at post) → travel → on scene (lognormal) → transport with prob. p_s to the nearest hospital by road time → unloading (lognormal) → free. |
| Fixed rules | Freed unit takes the most severe waiting call, oldest first, else returns to post. No backfilling of empty posts. No redirecting en route units. If no unit is free, the call queues and no decision is made. |

Unit states: `IDLE_AT_POST`, `EN_ROUTE`, `ON_SCENE`, `TRANSPORTING`,
`UNLOADING`, `RETURNING`.

Event types: `CALL_ARRIVAL`, `UNIT_ARRIVES_SCENE`, `SCENE_DONE`,
`ARRIVE_HOSPITAL`, `UNLOAD_DONE`, `ARRIVE_POST`. Every processed event is
appended to `sim.log` as `(t, event, unit_id, call_id)`.

**Determinism / common random numbers.** When a `Sim` is created, it
pre-generates the full call list from its seed: arrival time, location,
severity, on-scene duration, transport flag, unloading duration. Nothing else
in the engine is random, so every policy sees identical calls for the same
(map, load, seed, start hour). The evaluation protocol requires this, and a
test checks it.

## environment/

`DispatchEnv` wraps the engine in a Gymnasium semi-MDP. One `step()` = one
dispatch decision. `reset` draws map, load, start hour and the simulation seed
from `np_random` in a fixed order, then applies any pinned `options`. Cold
start: all units idle at their posts. Episodes run 6 h of arrivals, then drain
until every call is served. The reward function is a constructor argument
(`reward_fn(severity, T)`, default `reward_d`) so E2 can run arms A–D. Spaces,
mask, observation, reward and `info` are specified in `CONTRACT.md`.

`MockDispatchEnv` has the same contract with random dynamics, for building
the agent pipeline before the engine was ready. It is shape-correct but not
learnable.

## maps/

One JSON file per map. `default.json` is a hand-made smoke-test map. Anjali
delivers the real 20-map set in this format.

| Field | Type | Meaning |
|---|---|---|
| `name` | str | map name |
| `grid_size` | int | 16 |
| `row_types` | list[str], len 16 | road type of each row: `"side"` or `"main"` |
| `col_types` | list[str], len 16 | road type of each column |
| `highways` | list of polylines `[[r, c], ...]` | interchange nodes joined in order at highway speed. An edge can span many blocks; length is straight-line |
| `blocked` | list `[r, c]` | intersections removed (park, island, water) |
| `posts` | 4 × `[r, c]` | ambulance posts |
| `hospitals` | 1–2 × `[r, c]` | hospitals |
| `unit_posts` | list[int], len 6 | post index of each unit |
| `demand` | 16 × 16 float | relative call density per intersection (set to 0 on blocked cells) |

On load, the map is checked for: list lengths, posts and hospitals not on
blocked cells, non-negative demand, and every demand cell reachable from the
posts. Road speeds are in `simulator_config.ROAD_SPEED_MPH`.

## tests/

`pytest` from the repo root (`pytest.ini` sets `testpaths` and puts the repo
root on the import path, so no `__init__.py` files are needed). Plain-assert
`test_*` functions, no fixtures:

- **Both envs:** observations inside the contract space, mask is bool (7,)
  with at least one `True`, `info` and call-record keys, same seed →
  identical trajectory.
- **Real env:**
  - every call served by episode end, including held calls
  - mask never allows a busy unit
  - HOLD masked when nothing is busy
  - predicted ETA = real response time for every dispatch
  - call sequence independent of the policy
  - step reward = sum of the step's call records, and every call is credited
    exactly once
  - rewards match hand-computed values

Deferred to Dec 4 (λ̄ is still a placeholder): the closest-unit baseline's busy
fraction lands near the target load.

## simulator.ipynb

Smoke test (load map, run one closest-unit episode, print events) and sanity
plots (response-time histogram, fleet utilization). Later: calibrating λ̄ for
40/60/80% load. Code lives in the `.py` files; the notebook only imports and
plots.

## Inputs and outputs

| From / to | What |
|---|---|
| Anjali → simulator | map files (format above), hourly weights h(hour), severity shares π_s, p_s, service-time lognormal params, λ̄ per load |
| simulator → Josh | `DispatchEnv`, `MockDispatchEnv`, `CONTRACT.md` |
| simulator → Addy | per-call records in `info["calls"]` (call time, severity, zone, unit sent, both closest units, T, HOLD flag, reward) |

## Stretch (not in v1.0)

- **Call-rate memory:** an observed-rate feature that tracks rate and severity
  mix varying within a map, with older observations weighted less (e.g.
  exponential decay). Dropped from v1.0. The agent has no rate memory; only
  the critic sees true λ(t).
