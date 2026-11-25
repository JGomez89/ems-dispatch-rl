# Environment–agent contract v1.0

Frozen **Fri Nov 27, 2026**. Changes after that go to Josh and Addy first and
bump `CONTRACT_VERSION` in `contract.py`. Owner: Justin.

Both envs implement this exactly:

```python
from simulator.environment.mock_env import MockDispatchEnv   # random dynamics, available now
from simulator.environment.dispatch_env import DispatchEnv   # real simulator
```

Swapping one for the other is the only change the agent code should need.
Code constants live in `agent_interface/contract.py` (`OBSERVATION_SPACE`,
`ACTION_SPACE`, `OBS_SLICES`, `UNIT_FIELDS`, `CRITIC_SLICES`, `INFO_KEYS`,
`CALL_RECORD_KEYS`). Never hardcode indices.

## Episode

- Semi-MDP: one `step()` = one dispatch decision for the call that just
  arrived. Time between decisions varies; discount with γ^`info["dt"]`.
- `DispatchEnv(maps=[paths], loads=[0.4, 0.6, 0.8], reward_fn=reward_d)`.
  Each `reset` samples map, load and start hour uniformly.
- `reset(seed, options={"map_id": int, "load": float, "start_hour": int})`:
  any option given pins that part. `map_id` indexes the `maps` list. The same
  seed gives the same episode whichever options are pinned.
- Cold start: all units idle at their posts.
- 6 h of arrivals, then the queue drains. `terminated=True` on the step after
  which no call needs a decision; `truncated` is always `False`.

## Action

`Discrete(7)`: 0–5 = send unit 0–5, 6 = HOLD (the call waits in the queue; the
next freed unit takes the most severe waiting call, oldest first).

## Mask: `env.action_masks()`

Bool array, shape `(7,)`, at least one `True` (MaskablePPO convention).

- Unit `i` is legal only if `IDLE_AT_POST` or `RETURNING`.
- HOLD is illegal when no unit is `EN_ROUTE`, `ON_SCENE`, `TRANSPORTING` or
  `UNLOADING` (nothing would ever serve the held call).
- For a critical call, HOLD is also illegal unless a transporting/unloading
  unit's expected ETA beats the best free unit's ETA by ≥ 30 s and no other
  critical call is waiting.

## Observation: `Dict`

| Key | Shape | Who may read it |
|---|---|---|
| `"policy"` | `Box(77,)` float32 | actor and critic |
| `"critic"` | `Box(10,)` float32 | critic only (privileged, training only) |

### `"policy"` (77)

| Slice (`OBS_SLICES`) | Dims | Encoding |
|---|---|---|
| `units` | 6 × 10 | per unit, in fixed index order (index i = action i); see below |
| `severity` | 3 | one-hot: critical, urgent, non-urgent |
| `queue_count` | 3 | per severity, `min(count / 6, 2)` |
| `queue_wait` | 3 | per severity, longest current wait / 240 |
| `fleet` | 6 | share of units in each state |
| `time_of_day` | 2 | sin, cos of 2π · hour / 24 |

Per-unit block (`UNIT_FIELDS`, offsets within each 10-dim block):

| Field | Offset | Meaning |
|---|---|---|
| `status` | 0–5 | one-hot over `IDLE_AT_POST, EN_ROUTE, ON_SCENE, TRANSPORTING, UNLOADING, RETURNING` |
| `eta` | 6 | expected time to reach the call / 240 (see ETA) |
| `rank` | 7 | rank of `eta` among all 6 units (0 = soonest) / 5 |
| `time_to_free` | 8 | expected time until free / 240; 0 if idle or returning |
| `coverage_lost` | 9 | demand share within 240 s of travel from idle-at-post units, minus the share without this unit; 0 if not idle at post |

ETA:

- idle at post: 60 s turnout + road time from the post
- returning: time to finish its current road edge + road time from that node
  (no U-turns, no turnout). This equals the real response time if sent.
- busy: expected time until free + road time from where it will be freed,
  averaged over transport / no transport with probability p_s

Expected time until free uses known travel legs plus the medians of the
remaining on-scene and unloading phases. The transport leg is weighted by p_s
until the transport outcome is known.

All time features are / 240 s and clipped at 10. Observations are clipped to
the declared Box bounds.

There is no observed-call-rate feature in v1.0. The policy has no rate memory.

### `"critic"` (10)

| Slice (`CRITIC_SLICES`) | Dims | Meaning |
|---|---|---|
| `lambda` | 1 | true λ(t) / the map's 60%-load mean rate, clipped at 10 |
| `true_time_to_free` | 6 | true remaining time until free per unit (actual sampled durations) / 240, clipped at 10 |
| `load` | 3 | one-hot over 40 / 60 / 80% |

## Reward

Per call, credited when its first unit arrives (T = queue wait + turnout +
travel):

```text
r = w_s * ( +1                                    if T <= tau_s
            -min(((T - tau_s)/sigma_s)^k_s, 10)   otherwise )
```

| | critical | urgent | non-urgent |
|---|---|---|---|
| w_s | 3 | 1.5 | 0.5 |
| tau_s (s) | 300 | 300 | 600 |
| sigma_s (s) | 60 | 120 | 300 |
| k_s | 2 | 2 | 1 |

The **step reward** is the sum over calls whose first unit arrived between this
decision and the next one. That includes held and queued calls, and the last
step includes the whole drain. Credit is delayed and not tied to the action
just taken. E2 arms pass their own `reward_fn(severity, T)`.

## `info`

| Key | Type | Meaning |
|---|---|---|
| `dt` | float | seconds since the previous decision (0 after `reset`) |
| `closest_straight` | int | closest free unit by straight-line distance for the current call (-1 at terminal) |
| `closest_road` | int | free unit with the lowest ETA for the current call (-1 at terminal) |
| `calls` | list[dict] | one record per call credited this step |

Call record keys (`CALL_RECORD_KEYS`): `call_id`, `t_call` (s since episode
start), `severity`, `zone` (4 × 4-block zone, 0–15), `unit`,
`closest_straight`, `closest_road` (-1 if the call queued without a decision),
`T`, `held`, `reward`.

## Mock env

`MockDispatchEnv` has the same spaces, mask and `info` keys, with random
dynamics. It is seeded and deterministic. Episodes last Poisson(λ̄ · 6)
decisions. Observations are uniform within bounds (one-hots stay valid). The
mask is random with at least one legal unit and HOLD masked on critical calls.
`dt` ~ Exponential(mean 3600/λ̄), and rewards ~ U[-30, 3]. Nothing in it is
learnable: a flat learning curve is expected.
