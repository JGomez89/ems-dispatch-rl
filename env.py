"""Gymnasium wrapper: observation vector, Discrete(7) actions (6 units + HOLD), action masks.

Owner: M3 (task T11). Build and test against stub_sim.StubSimulator until sim.py is done:

    env = EMSDispatchEnv(maps=TRAIN_MAPS, loads=[0.4, 0.6, 0.8], reward="D", seed=0)
    env = EMSDispatchEnv(maps=["heldout_03"], loads=[0.8], surge=True, sim_cls=StubSimulator)
    obs, info = env.reset()                   # picks a map and load, draws a call seed
    obs, r, terminated, truncated, info = env.step(action)
    env.action_masks()                        # bool array, shape (7,)

Actions: 0-5 send that unit, 6 = HOLD.

Observation (float32, length OBS_DIM = N_ACTOR_OBS + N_PRIVILEGED). Times / 240 s; rates /
the map's historical mean rate. Keep this table and the slices below in sync.

    per unit i (6 x 5 = 30):
      [5i+0] available (idle at post or returning) 1/0
      [5i+1] ETA to the call: turnout (if at post) + road time; 0 if unavailable  (/240)
      [5i+2] distance rank among available units, scaled to [0, 1]; 1 if unavailable
      [5i+3] EXPECTED time until free (median-based, never the true draw)         (/240)
      [5i+4] coverage lost if it leaves its post   -- OPEN DECISION D3: definition
    call and queue (30-35):
      [30:33] current call severity, one-hot
      [33:36] queued calls per severity                                            (count / 6)
      [36]    longest current wait                                                 (/240)
    fleet and time (37-45):
      [37:43] fraction of units in each UnitStatus
      [43:45] time of day as sin, cos
      [45]    observed call rate, last 60 min (/ map mean)
    PRIVILEGED, critic only (46-...):
      [46:52] TRUE time until free per unit                                        (/240)
      [52]    true lambda(t) (/ map mean)

The actor must only read obs[:N_ACTOR_OBS]; ppo_smdp.PrivilegedCriticPolicy enforces this.
Privileged values are put in the observation (not info) because Stable-Baselines3 never
passes info to the networks.

Mask: unit i allowed iff available. HOLD (6) is allowed, EXCEPT for a critical call while
any unit is available, unless a transporting/unloading unit would arrive at least 30 s
sooner than the best available unit (expected time until free + travel vs. eta_s) and no
other critical call is waiting. Use EXPECTED times here, never true ones.

info keys (CONTRACT.md section 2):
    "dt"        seconds since the last decision (for gamma ** dt; ppo_smdp reads it)
    "reached"   list[sim.Reached] since the last step (evaluate.py writes these to CSV)
    "state"     sim.State at the new decision point (baselines read it; None when done)
    "map", "load", "call_seed"   episode identifiers for logging
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

from config import HOLD, N_ACTIONS, N_UNITS, P
from rewards import REWARDS, step_reward
from sim import Simulator

PER_UNIT = 5
N_ACTOR_OBS = PER_UNIT * N_UNITS + 3 + 3 + 1 + 6 + 2 + 1  # 46
N_PRIVILEGED = N_UNITS + 1                                 # 7
OBS_DIM = N_ACTOR_OBS + N_PRIVILEGED                       # 53
TIME_SCALE_S = P["time"]["obs_time_scale_s"]


class EMSDispatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, maps: str | Sequence[str] = "train_00", loads: float | Sequence[float] = 0.6,
                 reward: str = "D", seed: int = 0, surge: bool = False, record: bool = False,
                 sim_cls=Simulator):
        self.maps = [maps] if isinstance(maps, str) else list(maps)
        self.loads = [loads] if isinstance(loads, (int, float)) else list(loads)
        self.reward_fn = REWARDS[reward]
        self.surge, self.record, self.sim_cls = surge, record, sim_cls
        self.base_seed = seed  # call seeds for training episodes derive from this
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.sim = None

    def reset(self, *, seed=None, options=None):
        """Pick map and load (uniformly from self.maps / self.loads) and a call seed.

        options may fix them for evaluation: {"map": "heldout_03", "load": 0.8, "call_seed": 100007}.
        Then sim.reset(...) and sim.next_decision() to reach the first decision.
        """
        raise NotImplementedError

    def step(self, action):
        """sim.dispatch(None if action == HOLD else action), then sim.next_decision().

        reward = rewards.step_reward(self.reward_fn, step.reached, decision_time_s).
        terminated = step.done. On the final (drain) step, return a zero observation.
        """
        raise NotImplementedError

    def action_masks(self) -> np.ndarray:
        raise NotImplementedError

    def _obs(self, state) -> np.ndarray:
        raise NotImplementedError


if __name__ == "__main__":
    # Smoke test: one episode of random valid actions on the stub simulator.
    from baselines import random_valid
    from stub_sim import StubSimulator

    env = EMSDispatchEnv(sim_cls=StubSimulator)
    obs, info = env.reset()
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, terminated, truncated, info = env.step(random_valid(obs, env.action_masks(), info))
        total, steps, done = total + r, steps + 1, terminated or truncated
    print(f"{steps} decisions, return {total:.1f}")
