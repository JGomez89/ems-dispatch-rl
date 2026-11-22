"""Interface checks that BOTH sim.Simulator and stub_sim.StubSimulator must pass."""

import math

from config import N_UNITS
from sim import AVAILABLE


def play(sim_cls, call_seed, choose, max_steps=10_000, **kw):
    """Run one episode, choosing actions with choose(state) -> unit index or None (HOLD).

    Returns (calls_seen, reached) where calls_seen is [(id, time_s, cell, severity)].
    """
    sim = sim_cls(call_seed=call_seed, **kw)
    sim.reset()
    calls, reached = [], []
    for _ in range(max_steps):
        step = sim.next_decision()
        assert step.dt_s >= 0
        reached += step.reached
        if step.done:
            return calls, reached
        st = sim.state()
        check_state(st)
        c = st.call
        calls.append((c.id, c.time_s, c.cell, int(c.severity)))
        sim.dispatch(choose(st))
    raise AssertionError("episode did not end")


def check_state(st):
    assert len(st.units) == N_UNITS
    assert any(u.available for u in st.units), "a decision needs an available unit"
    for u in st.units:
        assert u.available == (u.status in AVAILABLE)
        assert (u.eta_s < math.inf) == u.available
        assert u.expected_free_in_s >= 0 and u.true_free_in_s >= 0
    assert st.units[st.closest_unit].available
    best = min(u.eta_s for u in st.units if u.available)
    assert st.units[st.closest_unit].eta_s == best
    assert 0 <= st.hour < 24


def check_reached(reached):
    ids = [r.call.id for r in reached]
    assert len(ids) == len(set(ids)), "a call was reached twice"
    for r in reached:
        assert r.response_s >= 0
        assert abs(r.response_s - (r.arrive_s - r.call.time_s)) < 1e-6


def closest(st):
    return st.closest_unit


def hold_noncritical_else_last(st):
    if int(st.call.severity) != 0:
        return None
    return max(i for i, u in enumerate(st.units) if u.available)
