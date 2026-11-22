"""Reward arms for E2. Each arm is its own function so arms can be added without conflicts.

Owner: M3 (task T12). Constants live in config/params.yaml ("reward").

Response time T runs from call arrival to the first unit's arrival (queue + turnout + travel).
Rewards are credited when the unit arrives (sim.Reached), not at the dispatch decision.

    r = w_s * ( +1                              if T <= tau_s
              ( -min(((T - tau_s)/sigma_s)^k_s, cap)  otherwise

OPEN DECISION D1 (CONTRACT.md): the per-tier cap. tests/test_rewards.py::test_late_cost_ordering
is marked xfail until it is settled.
OPEN DECISION D2 (CONTRACT.md): how per-call rewards are combined into one env step
(step_reward below).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from config import P

_R = P["reward"]
WEIGHT = list(map(float, _R["weight"]))      # indexed by Severity (0 critical, 1 urgent, 2 non-urgent)
TARGET_S = list(map(float, _R["target_s"]))
SCALE_S = list(map(float, _R["scale_s"]))
EXPONENT = list(map(float, _R["exponent"]))
MAX_PENALTY = float(_R["max_penalty"])


def reward_a(severity: int, response_s: float) -> float:
    """A: on-time count."""
    return 1.0 if response_s <= TARGET_S[severity] else 0.0


def reward_b(severity: int, response_s: float) -> float:
    """B: on-time count weighted by w_s."""
    return WEIGHT[severity] * reward_a(severity, response_s)


def reward_c(severity: int, response_s: float) -> float:
    """C: +1 on time, else -min(((T - tau) / sigma) ** k, cap). Unweighted."""
    tau = TARGET_S[severity]
    if response_s <= tau:
        return 1.0
    lateness = (response_s - tau) / SCALE_S[severity]
    return -min(lateness ** EXPONENT[severity], MAX_PENALTY)


def reward_d(severity: int, response_s: float) -> float:
    """D: reward C weighted by w_s (main reward)."""
    return WEIGHT[severity] * reward_c(severity, response_s)


REWARDS: dict[str, Callable[[int, float], float]] = {
    "A": reward_a,
    "B": reward_b,
    "C": reward_c,
    "D": reward_d,
}


def step_reward(reward_fn: Callable[[int, float], float], reached: Sequence, decision_time_s: float,
                gamma_per_s: float | None = None) -> float:
    """Combine the calls reached since the last decision into one step reward.

    Current behaviour: plain sum, no discount inside the step.
    OPEN DECISION D2: if the team chooses to discount each reward by
    gamma_per_s ** (r.arrive_s - decision_time_s), change it HERE only, so the env, BC and
    evaluation all stay consistent.
    """
    return float(sum(reward_fn(int(r.call.severity), r.response_s) for r in reached))
