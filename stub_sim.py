"""Stand-in simulator with the same interface as sim.Simulator (task T03).

Owner: M1. It needs no map, no Dijkstra and no DataSF values, so M3 (env, PPO) and M4
(baselines, evaluation, BC) can build and test their code before the real simulator exists:

    env = EMSDispatchEnv(sim_cls=StubSimulator)

The numbers are random but every rule of the interface holds: deterministic for a given
call_seed, valid statuses, closest_unit is available, reached calls have consistent times,
and the episode ends with done=True after a drain step. Do not report results from it.
"""

from __future__ import annotations

import numpy as np

from config import N_UNITS, P
from maps import GRID_SIZE, zone_of
from sim import (AVAILABLE, EPISODE_HOURS, TURNOUT_S, Call, DecisionStep, Reached, Severity,
                 State, UnitStatus, UnitView)


class StubSimulator:
    def __init__(self, map_name: str = "train_00", load: float = 0.6, call_seed: int = 0,
                 surge: bool = False, record: bool = False):
        self.map_name, self.load, self.call_seed, self.surge = map_name, load, call_seed, surge
        self.trace = [] if record else None
        self.city = None

    def reset(self, map_name=None, load=None, call_seed=None, surge=None) -> None:
        if map_name is not None:
            self.map_name = map_name
        if load is not None:
            self.load = load
        if call_seed is not None:
            self.call_seed = call_seed
        if surge is not None:
            self.surge = surge
        self.rng = np.random.default_rng(self.call_seed)  # call stream: never touched by dispatch()
        self.out_rng = np.random.default_rng([self.call_seed, 1])  # HOLD outcomes only
        self.t = 0.0
        self.start_hour = int(self.rng.integers(24))
        self.next_id = 0
        self.pending: list[tuple[Call, int, float, float | None, bool, bool]] = []
        self.current: Call | None = None
        self.current_held = False
        self.done = False
        self._draw_state()

    def _new_call(self) -> Call:
        cell = (int(self.rng.integers(GRID_SIZE)), int(self.rng.integers(GRID_SIZE)))
        sev = Severity(int(self.rng.choice(3, p=P["calls"]["severity_share"])))
        call = Call(id=self.next_id, time_s=self.t, cell=cell, severity=sev, zone=zone_of(cell),
                    transport=bool(self.rng.random() < 0.6), on_scene_s=900.0, unloading_s=1200.0)
        self.next_id += 1
        return call

    def _draw_state(self) -> None:
        statuses = self.rng.choice(len(UnitStatus), size=N_UNITS)
        statuses[int(self.rng.integers(N_UNITS))] = UnitStatus.IDLE_AT_POST  # >= 1 available
        self.units = []
        for s in statuses:
            s = UnitStatus(int(s))
            avail = s in AVAILABLE
            free_in = 0.0 if avail else float(self.rng.uniform(60, 1800))
            eta = (TURNOUT_S if s == UnitStatus.IDLE_AT_POST else 0) + float(self.rng.uniform(30, 600))
            self.units.append(UnitView(
                status=s, cell=(int(self.rng.integers(GRID_SIZE)), int(self.rng.integers(GRID_SIZE))),
                post=(0, 0), available=avail, eta_s=eta if avail else float("inf"),
                expected_free_in_s=free_in, true_free_in_s=free_in * float(self.rng.uniform(0.7, 1.3))))
        self.rates = (float(self.rng.uniform(2, 6)), float(self.rng.uniform(2, 6)))

    def next_decision(self) -> DecisionStep:
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        reached = [r for r in self._settle(lambda arrive: arrive <= self.t)]
        if self.t >= EPISODE_HOURS * 3600:  # drain: everything left is reached now
            self.done = True
            reached += self._settle(lambda arrive: True)
            return DecisionStep(dt_s=0.0, reached=reached, done=True)
        dt = float(self.rng.exponential(900.0))
        self.t += dt
        reached += self._settle(lambda arrive: arrive <= self.t)
        self.current, self.current_held = self._new_call(), False
        self._draw_state()
        return DecisionStep(dt_s=dt, reached=reached, done=False)

    def _settle(self, due) -> list[Reached]:
        out, keep = [], []
        for call, unit, arrive, decided, held, non_closest in self.pending:
            if due(arrive):
                out.append(Reached(call=call, unit=unit, arrive_s=arrive, response_s=arrive - call.time_s,
                                   decided_s=decided, held=held, non_closest=non_closest))
            else:
                keep.append((call, unit, arrive, decided, held, non_closest))
        self.pending = keep
        return out

    def state(self) -> State:
        avail = [i for i, u in enumerate(self.units) if u.available]
        closest = min(avail, key=lambda i: self.units[i].eta_s)
        return State(time_s=self.t, hour=(self.start_hour + self.t / 3600) % 24, call=self.current,
                     units=self.units, queue=[], closest_unit=closest,
                     observed_rate_per_h=self.rates[0], true_rate_per_h=self.rates[1])

    def dispatch(self, unit_idx: int | None) -> None:
        st = self.state()
        if unit_idx is None:  # HOLD: a fixed rule serves it later, a bit slower
            arrive = self.t + TURNOUT_S + float(self.out_rng.uniform(300, 1200))
            self.pending.append((self.current, int(self.out_rng.integers(N_UNITS)), arrive, None, True, False))
        else:
            if not self.units[unit_idx].available:
                raise ValueError(f"unit {unit_idx} is not available")
            arrive = self.t + self.units[unit_idx].eta_s
            self.pending.append((self.current, unit_idx, arrive, self.t, False, unit_idx != st.closest_unit))
        self.current = None
