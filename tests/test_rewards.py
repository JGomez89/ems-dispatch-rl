"""Reward arms (rewards.py). Owner: M3."""

import pytest

from rewards import MAX_PENALTY, REWARDS, TARGET_S, reward_d

CRIT, URG, NON = 0, 1, 2


def test_on_time_values():
    for s in (CRIT, URG, NON):
        assert REWARDS["A"](s, TARGET_S[s]) == 1.0
        assert REWARDS["C"](s, TARGET_S[s]) == 1.0
    assert REWARDS["B"](CRIT, 100) == 3.0
    assert REWARDS["D"](NON, 100) == 0.5


def test_proposal_examples():
    # critical, 60 s late: -3 * (60/60)^2
    assert reward_d(CRIT, 360) == pytest.approx(-3.0)
    # urgent, 240 s late: -1.5 * (240/120)^2
    assert reward_d(URG, 540) == pytest.approx(-6.0)
    # non-urgent, 300 s late: -0.5 * (300/300)^1
    assert reward_d(NON, 900) == pytest.approx(-0.5)


def test_late_is_never_better_than_on_time():
    for arm in "ABCD":
        for s in (CRIT, URG, NON):
            assert REWARDS[arm](s, TARGET_S[s] + 1) < REWARDS[arm](s, TARGET_S[s])


def _cost(s, lateness):
    """What a call loses by being `lateness` s past target instead of on time (arm D)."""
    return reward_d(s, TARGET_S[s]) - reward_d(s, TARGET_S[s] + lateness)


@pytest.mark.xfail(strict=True, reason=(
    "OPEN DECISION D1: the per-tier cap (max_penalty) makes extra delay free for critical "
    "calls ~190 s past target. Remove this marker once the cap is fixed."))
def test_marginal_cost_ordering():
    """One more second of delay must cost critical >= urgent >= non-urgent at every lateness.

    Starts at 10 s late: squared penalties are flat right at the target (minor, see D1).
    """
    for x in range(10, 3600, 10):
        slope = [_cost(s, x + 1) - _cost(s, x) for s in (CRIT, URG, NON)]
        assert slope[0] >= slope[1] >= slope[2] > 0, (x, slope)


def test_total_cost_ordering():
    """Being late always costs critical > urgent > non-urgent in total (holds today)."""
    for x in range(1, 3600, 5):
        c = [_cost(s, x) for s in (CRIT, URG, NON)]
        assert c[0] > c[1] > c[2], (x, c)


def test_cap_value_is_shared():
    assert MAX_PENALTY > 0
