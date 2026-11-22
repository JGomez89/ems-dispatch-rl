"""Reward functions for E2. Each arm is its own function so arms can be added without conflicts.

Response time T runs from call arrival to the first unit's arrival (queue + turnout + travel).
"""

from __future__ import annotations

from collections.abc import Callable

from sim import Severity

#                 critical urgent non-urgent
WEIGHT = {Severity.CRITICAL: 3.0, Severity.URGENT: 1.5, Severity.NON_URGENT: 0.5}
TARGET_S = {Severity.CRITICAL: 300.0, Severity.URGENT: 300.0, Severity.NON_URGENT: 600.0}
SCALE_S = {Severity.CRITICAL: 60.0, Severity.URGENT: 120.0, Severity.NON_URGENT: 300.0}
EXPONENT = {Severity.CRITICAL: 2.0, Severity.URGENT: 2.0, Severity.NON_URGENT: 1.0}
MAX_PENALTY = 10.0


def reward_a(severity: Severity, response_s: float) -> float:
    """A: on-time count."""
    return 1.0 if response_s <= TARGET_S[severity] else 0.0


def reward_b(severity: Severity, response_s: float) -> float:
    """B: on-time count weighted by w_s."""
    return WEIGHT[severity] * reward_a(severity, response_s)


def reward_c(severity: Severity, response_s: float) -> float:
    """C: +1 on time, else -min(((T - tau) / sigma) ** k, 10). Unweighted."""
    tau = TARGET_S[severity]
    if response_s <= tau:
        return 1.0
    lateness = (response_s - tau) / SCALE_S[severity]
    return -min(lateness ** EXPONENT[severity], MAX_PENALTY)


def reward_d(severity: Severity, response_s: float) -> float:
    """D: reward C weighted by w_s (main reward)."""
    return WEIGHT[severity] * reward_c(severity, response_s)


REWARDS: dict[str, Callable[[Severity, float], float]] = {
    "A": reward_a,
    "B": reward_b,
    "C": reward_c,
    "D": reward_d,
}
