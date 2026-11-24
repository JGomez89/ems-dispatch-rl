# Simulator

Event-driven EMS dispatch simulator for a 16 × 16 grid city, with a Gymnasium
interface for the PPO agent. Owner: Justin.

The agent's only decision is **which unit to send** when a call arrives (or
HOLD). Everything else follows fixed rules.

## Layout

```
simlulator/
├── engine/            # pure simulation: no Gymnasium, no RL
├── environment/       # Gymnasium env wrapping the engine
├── agent_interface/   # observation / action / mask contract shared with the agent
├── tests/             # acceptance checks (plain asserts)
└── simulator.ipynb    # smoke tests and development experiments
```

Dependency direction: `engine` ← `environment` ← `agent_interface`. The engine
never imports from the other two.

### engine/

Discrete-event core. Owns time, the event queue, and unit and call state.

| Piece | Spec |
|---|---|
| Map | 16 × 16 grid, 2 × 2 mi, 1/8-mi blocks. Side roads, main roads (every 4th row/col), highways. Variants differ in hotspots, post layout, obstacles (park, island) and hospitals. Map configs come from Anjali. |
| Travel | Dijkstra on the road graph, all-pairs cached per map. Road-time and straight-line distance both exposed (the baselines need both). |
| Fleet | 6 ambulances, 4 posts, 1–2 hospitals. |
| Calls | Poisson, λ(t) = λ̄ · h(hour). 24 hourly weights, λ̄ (for 40/60/80% load) and severity shares π_s are fitted inputs from Anjali. |
| Service | 60 s turnout (only when dispatched idle at post) → travel → on scene (lognormal) → transport with prob. p_s → unloading (lognormal) → free. |
| Fixed rules | Freed unit takes the most severe waiting call, oldest first, else returns to post. No backfilling of empty posts. No redirecting en route units. If no unit is free, the call queues and no decision is made. |

Unit states: `IDLE_AT_POST`, `EN_ROUTE`, `ON_SCENE`, `TRANSPORTING`,
`UNLOADING`, `RETURNING`.

Event types: `CALL_ARRIVAL`, `UNIT_ARRIVES_SCENE`, `SCENE_DONE`,
`ARRIVE_HOSPITAL`, `UNLOAD_DONE`, `ARRIVE_POST`.

The engine must be deterministic given a seed. Identical call sequences
across policies are a requirement of the evaluation protocol.

### environment/

Gymnasium `Env` (semi-MDP). One `step()` = one dispatch decision.

- `reset(seed)`: picks a uniformly random start hour, runs events until the
  first call that needs a decision.
- `step(action)`: applies the dispatch, then runs events until the next call
  with at least one free unit.
- Episode: 6 simulated hours of arrivals, then no new calls, then run until
  every waiting call is served (every call gets a real response time T).
  The episode ends after the drain.
- Reward per call, credited when the first unit arrives
  (T = queue wait + turnout + travel):

  ```
  r = w_s * ( +1                              if T <= tau_s
              -min(((T - tau_s)/sigma_s)^k_s, 10)  otherwise )
  ```

  | | critical | urgent | non-urgent |
  |---|---|---|---|
  | w_s | 3 | 1.5 | 0.5 |
  | tau_s (s) | 300 | 300 | 600 |
  | sigma_s (s) | 60 | 120 | 300 |
  | k_s | 2 | 2 | 1 |

  The reward function is swappable so E2 can run arms A–D.

### agent_interface/

The contract Josh trains against. **Frozen Fri Nov 27, 2026.** Changes after
that go to Josh and Addy first.

- **Action:** `Discrete(7)`. 0–5 = send unit 1–6 (idle or returning), 6 = HOLD.
- **Mask** (`action_masks()`, MaskablePPO convention): removes busy units.
  Removes HOLD for a critical call while any unit is free, unless a
  transporting/unloading unit would arrive ≥ 30 s sooner than the best free
  unit and no other critical call is waiting.
- **Observation:** fixed-length float vector.
  - Per unit (×6): status, ETA to call, distance rank, expected time until
    free, coverage lost if it leaves its post.
  - Call: severity.
  - Queue: count per severity, longest wait.
  - Global: fleet status fractions, time of day, observed call rate
    (~60 min moving average).
  - Times ÷ 240 s. Rates ÷ the map's warm-up mean.
- **Privileged critic obs** (training only): true simulator values such as
  λ(t). Returned in `info`, never in the policy observation.
- **`info` per step:** `dt` (seconds since previous decision, for γ^Δt),
  privileged obs, closest-unit actions (straight-line and road time) for
  baselines and BC labels, and per-call records for metrics.

### tests/

Acceptance checks the environment must pass before handoff: determinism under
a fixed seed, mask never allows a busy unit, every call is served by episode
end, rewards match hand-computed values, closest-unit baseline busy fraction
lands near the target load.

## simulator.ipynb

The working notebook for this folder. Use it for:

- smoke tests (build a map, run one episode, print events)
- sanity plots (unit utilization, response-time histograms, call rate by hour)
- running the closest-unit policy to calibrate λ̄ for 40/60/80% load
- quick checks before pushing an interface change

Code lives in the `.py` files. The notebook imports from them and holds no
logic of its own.

## Inputs and outputs

| From / to | What |
|---|---|
| Anjali → simulator | map configs, hourly weights h(hour), severity shares π_s, p_s, service-time lognormal params, λ̄ per load |
| simulator → Josh | Gymnasium env + the contract above |
| simulator → Addy | per-call records (call time, severity, zone, unit sent, closest unit, T, HOLD flag) for metrics |
