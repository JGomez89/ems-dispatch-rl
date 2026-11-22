"""The stand-in simulator keeps the interface promises, so others can rely on it."""

from stub_sim import StubSimulator
from tests.sim_checks import check_reached, closest, hold_noncritical_else_last, play


def test_episode_runs_and_drains():
    calls, reached = play(StubSimulator, 0, closest)
    assert calls
    check_reached(reached)
    assert {r.call.id for r in reached} == {c[0] for c in calls}, "drain must reach every call"


def test_same_calls_for_every_policy():
    a, _ = play(StubSimulator, 3, closest)
    b, _ = play(StubSimulator, 3, hold_noncritical_else_last)
    assert a == b
