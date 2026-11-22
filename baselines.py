"""Non-learned policies. Owner: M4 (task T14).

Every policy, learned or not, has the same signature (CONTRACT.md section 3), so evaluate.py,
bc.py and plot.py can run any of them by name:

    policy(obs, mask, info) -> action   # int in [0, 7); mask[action] must be True

Baselines read info["state"] (a sim.State), never env.sim, so the simulator can change
internally without breaking them. They follow the same fixed rules as the agent and never HOLD.
"""

from __future__ import annotations

import numpy as np


def closest_straight_line(obs, mask, info):
    """Current practice: the available unit closest to the call by straight-line distance."""
    raise NotImplementedError


def closest_travel_time(obs, mask, info):
    """Available unit with the smallest ETA (turnout + Dijkstra time). Also the BC teacher
    (OPEN DECISION D4). Equivalent to info["state"].closest_unit."""
    return int(info["state"].closest_unit)


def random_valid(obs, mask, info):
    """Uniform over allowed actions. Sanity check: everything should beat this."""
    return int(np.random.choice(np.flatnonzero(mask)))


POLICIES = {
    "closest_straight_line": closest_straight_line,
    "closest_travel_time": closest_travel_time,
    "random": random_valid,
}
