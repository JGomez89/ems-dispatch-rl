"""Gymnasium wrapper: observation vector, Discrete(7) actions (6 units + HOLD), action masks.

Interface everyone else codes against (keep this docstring up to date):

    env = EMSDispatchEnv(map_name="train_00", reward="D", seed=0, load=0.6)
    obs, info = env.reset()
    obs, r, terminated, truncated, info = env.step(action)
    env.action_masks()      # bool array, shape (7,)

Actions: 0-5 send that unit, 6 = HOLD.
Observation: TODO list each slice and its normalization, e.g.
    [0:6]   unit status (idle/returning = 1)
    ...
info keys:
    "dt"        seconds since the last decision (for gamma ** dt discounting)
    "privileged" true simulator values (e.g. lambda(t)) for the critic only
    "calls"     calls finished since the last step (for evaluate.py metrics)
"""

import gymnasium as gym
import numpy as np

from rewards import REWARDS
from sim import N_UNITS, Simulator

HOLD = N_UNITS


class EMSDispatchEnv(gym.Env):
    def __init__(self, map_name="train_00", reward="D", seed=0, load=0.6, record=False):
        self.sim = Simulator(map_name=map_name, seed=seed, load=load, record=record)
        self.reward_fn = REWARDS[reward]
        self.action_space = gym.spaces.Discrete(N_UNITS + 1)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))  # TODO: real size

    def reset(self, *, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def action_masks(self):
        raise NotImplementedError


if __name__ == "__main__":
    # Smoke test: one episode of random valid actions.
    from baselines import random_valid

    env = EMSDispatchEnv()
    obs, info = env.reset()
    total, steps, done = 0.0, 0, False
    while not done:
        obs, r, terminated, truncated, info = env.step(random_valid(env, obs))
        total, steps, done = total + r, steps + 1, terminated or truncated
    print(f"{steps} decisions, return {total:.1f}")
