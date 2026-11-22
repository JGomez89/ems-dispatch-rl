"""Non-learned policies. Every policy has the same signature, so evaluate.py and plot.py
can run any of them by name (like the `algs` dict in Assignment 1):

    policy(env, obs) -> action   # int in [0, 7), must be allowed by env.action_masks()
"""

import numpy as np


def closest_straight_line(env, obs):
    """Current practice: closest available unit by straight-line distance."""
    raise NotImplementedError


def closest_travel_time(env, obs):
    """Closest available unit by road travel time (turnout + Dijkstra time)."""
    raise NotImplementedError


def random_valid(env, obs):
    """Uniform over allowed actions. Sanity check: everything should beat this."""
    return int(np.random.choice(np.flatnonzero(env.action_masks())))


POLICIES = {
    "closest_straight_line": closest_straight_line,
    "closest_travel_time": closest_travel_time,
    "random": random_valid,
}
