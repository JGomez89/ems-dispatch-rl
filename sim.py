"""Event-driven EMS simulator: calls, ambulances and the event loop on a CityMap.

Owner: M1 (tasks T04-T06). No RL code here. env.py wraps this for Gymnasium; maps.py builds
the city; calls.py draws the call stream. Until this is implemented, everyone else codes
against stub_sim.StubSimulator, which has the same interface.

Interface (CONTRACT.md section 1):

    sim = Simulator(map_name="train_00", load=0.6, call_seed=0)
    sim.reset()                          # optionally reset(map_name=..., load=..., call_seed=...)
    step = sim.next_decision()           # run events until a call needs a decision
    state = sim.state()                  # what the env and baselines read
    sim.dispatch(unit_idx)               # 0-5, or None for HOLD
    step = sim.next_decision()           # ... until step.done

Fixed rules (proposal section 3), implemented here, never by the policy:
  - If no unit is free when a call arrives, it queues and no decision is made.
  - A freed unit takes the most severe waiting call, oldest first; else returns to its post.
  - Empty posts are not backfilled. En route units are not redirected.
  - HOLD puts the current call in the queue; the fixed rule above serves it later.
  - After episode_hours no calls arrive; next_decision() then runs the queue down and
    returns every remaining reached call with done=True (the "drain").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from config import N_UNITS, P
from maps import Cell, CityMap, get_map

TURNOUT_S = P["time"]["turnout_s"]
EPISODE_HOURS = P["time"]["episode_hours"]


class Severity(IntEnum):
    CRITICAL = 0
    URGENT = 1
    NON_URGENT = 2


class UnitStatus(IntEnum):
    IDLE_AT_POST = 0
    EN_ROUTE = 1       # includes the 60 s turnout when dispatched from post
    ON_SCENE = 2
    TRANSPORTING = 3
    UNLOADING = 4
    RETURNING = 5


AVAILABLE = (UnitStatus.IDLE_AT_POST, UnitStatus.RETURNING)  # dispatchable statuses


@dataclass
class Call:
    """One call, fully drawn at reset by calls.generate_calls (common random numbers)."""
    id: int
    time_s: float
    cell: Cell
    severity: Severity
    zone: int
    transport: bool          # goes to hospital after scene
    on_scene_s: float        # pre-drawn service times; the sim never draws during a run
    unloading_s: float


@dataclass
class Unit:
    post: Cell
    cell: Cell               # current cell; for RETURNING units, see OPEN DECISION D6
    status: UnitStatus = UnitStatus.IDLE_AT_POST
    call_id: int | None = None
    true_free_s: float = 0.0       # actual time it becomes free (privileged)
    expected_free_s: float = 0.0   # same, but from median service times (actor-visible)


@dataclass
class UnitView:
    """Read-only per-unit view handed to env/baselines."""
    status: UnitStatus
    cell: Cell
    post: Cell
    available: bool
    eta_s: float             # turnout (if at post) + travel to the current call; inf if busy
    expected_free_in_s: float  # expected seconds until free (0 if available). Actor-visible
    true_free_in_s: float      # actual seconds until free. PRIVILEGED: critic only


@dataclass
class State:
    """Everything the env needs to build an observation at a decision point."""
    time_s: float                    # seconds since episode start
    hour: float                      # time of day in hours, [0, 24)
    call: Call                       # the call awaiting a decision
    units: list[UnitView]
    queue: list[Call]                # waiting calls (not including `call`)
    closest_unit: int                # available unit with the smallest eta_s (for flags/baselines)
    observed_rate_per_h: float       # calls in the last observed_rate_window_s, per hour
    true_rate_per_h: float           # lambda(t). PRIVILEGED: critic only


@dataclass
class Reached:
    """A call whose first unit arrived. Rewards and per-call CSV rows are built from these."""
    call: Call
    unit: int
    arrive_s: float
    response_s: float                # arrive_s - call.time_s (queue + turnout + travel)
    decided_s: float | None          # when the agent chose this unit; None if a fixed rule did
    held: bool                       # the agent chose HOLD for this call at least once
    non_closest: bool                # the agent sent a unit other than state.closest_unit


@dataclass
class DecisionStep:
    dt_s: float                      # seconds since the previous decision
    reached: list[Reached] = field(default_factory=list)  # reached since the previous decision
    done: bool = False               # episode over (drain finished, nothing left to decide)


@dataclass
class Snapshot:
    """One frame for plot.animate_episode. Appended after every event when record=True."""
    time_s: float
    units: list[tuple[tuple[float, float], UnitStatus]]  # ((row, col) position, status)
    waiting: list[Call]
    dispatch: tuple[int, Cell] | None = None  # (unit_idx, call cell) if a decision was made


class Simulator:
    def __init__(self, map_name: str = "train_00", load: float = 0.6, call_seed: int = 0,
                 surge: bool = False, record: bool = False):
        self.map_name, self.load, self.call_seed, self.surge = map_name, load, call_seed, surge
        self.record = record
        self.city: CityMap = get_map(map_name)
        self.trace: list[Snapshot] | None = [] if record else None

    def reset(self, map_name: str | None = None, load: float | None = None,
              call_seed: int | None = None, surge: bool | None = None) -> None:
        """Place units at home posts, then draw the start hour, surge start (if surge) and the
        whole call stream from np.random.default_rng(call_seed) via calls.generate_calls."""
        raise NotImplementedError

    def next_decision(self) -> DecisionStep:
        """Run events until a call arrives while a unit is available, or the drain ends."""
        raise NotImplementedError

    def state(self) -> State:
        """Snapshot at the current decision point. Only valid after next_decision() with done=False."""
        raise NotImplementedError

    def dispatch(self, unit_idx: int | None) -> None:
        """Send unit_idx (must be available) to the current call, or HOLD if None."""
        raise NotImplementedError
