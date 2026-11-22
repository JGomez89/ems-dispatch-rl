"""Event-driven EMS simulator: calls, ambulances and the event loop on a CityMap.

No RL code here. env.py wraps this for Gymnasium; maps.py builds the city.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from maps import CityMap, get_map

# Values marked X in the proposal: fill in from DataSF.
N_UNITS = 6
TURNOUT_S = 60
EPISODE_HOURS = 6


class Severity(IntEnum):
    CRITICAL = 0
    URGENT = 1
    NON_URGENT = 2


class UnitStatus(IntEnum):
    IDLE_AT_POST = 0
    EN_ROUTE = 1
    ON_SCENE = 2
    TRANSPORTING = 3
    UNLOADING = 4
    RETURNING = 5


@dataclass
class Call:
    time_s: float
    location: tuple[int, int]
    severity: Severity


@dataclass
class Unit:
    post: tuple[int, int]
    location: tuple[int, int]
    status: UnitStatus = UnitStatus.IDLE_AT_POST
    busy_until_s: float = 0.0


@dataclass
class Snapshot:
    """One frame for plot.animate_episode. Appended after every event when record=True."""
    time_s: float
    units: list[tuple[tuple[float, float], UnitStatus]]  # (x, y) position, status
    waiting: list[Call]  # calls not yet reached by a unit
    dispatch: tuple[int, tuple[int, int]] | None = None  # (unit_idx, call location) if a decision was made


class Simulator:
    def __init__(self, map_name: str = "train_00", seed: int = 0, load: float = 0.6, record: bool = False):
        self.city: CityMap = get_map(map_name)
        self.rng = np.random.default_rng(seed)
        self.load = load  # 0.4 / 0.6 / 0.8 unit utilization under closest-unit dispatch
        self.trace: list[Snapshot] | None = [] if record else None

    def reset(self):
        """Place units at posts, draw the start hour and the call sequence for this episode."""
        raise NotImplementedError

    def next_decision(self):
        """Run events until a call arrives with a free unit. Return (call, dt_s, finished_calls)."""
        raise NotImplementedError

    def dispatch(self, unit_idx):
        """Send unit_idx to the current call, or HOLD if unit_idx is None."""
        raise NotImplementedError
