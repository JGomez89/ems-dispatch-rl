"""The real simulator (M1). Tests skip until sim.py is implemented, then must pass.

Same checks as the stub, plus the determinism promise in CONTRACT.md: the call stream
depends only on the call seed, never on the policy.
"""

from sim import Simulator
from tests.helpers import skip_if_unimplemented
from tests.sim_checks import check_reached, closest, hold_noncritical_else_last, play


def test_episode_runs_and_drains():
    calls, reached = skip_if_unimplemented(play, Simulator, 0, closest)
    check_reached(reached)
    assert {r.call.id for r in reached} >= {c[0] for c in calls}, "drain must reach every call"


def test_same_calls_for_every_policy():
    a, _ = skip_if_unimplemented(play, Simulator, 3, closest)
    b, _ = skip_if_unimplemented(play, Simulator, 3, hold_noncritical_else_last)
    assert a == b


def test_reached_records_agent_flags():
    _, reached = skip_if_unimplemented(play, Simulator, 5, hold_noncritical_else_last)
    assert any(r.held for r in reached)
    for r in reached:
        if r.held:
            assert r.decided_s is None, "a held call is assigned later by a fixed rule"
