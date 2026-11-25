"""Environment-agent contract v1.0. See CONTRACT.md. Both envs build their spaces from here."""
import numpy as np
from gymnasium import spaces

from simulator import simulator_config as cfg

CONTRACT_VERSION = "1.0"

N_STATES = len(cfg.UNIT_STATES)
N_SEV = len(cfg.SEVERITIES)
TIME_CLIP = 10.0      # time features are /240 s, clipped here (= 40 min)
QUEUE_CLIP = 2.0      # queue counts are /N_UNITS, clipped here

# Per-unit block layout (10 per unit, units in fixed index order).
UNIT_FIELDS = {"status": slice(0, N_STATES), "eta": N_STATES, "rank": N_STATES + 1,
               "time_to_free": N_STATES + 2, "coverage_lost": N_STATES + 3}
UNIT_DIM = N_STATES + 4

# name -> (start, stop) in the "policy" vector
_blocks = [("units", cfg.N_UNITS * UNIT_DIM), ("severity", N_SEV), ("queue_count", N_SEV),
           ("queue_wait", N_SEV), ("fleet", N_STATES), ("time_of_day", 2)]
OBS_SLICES, _i = {}, 0
for _name, _n in _blocks:
    OBS_SLICES[_name] = slice(_i, _i + _n)
    _i += _n
POLICY_DIM = _i       # 77

CRITIC_SLICES = {"lambda": slice(0, 1), "true_time_to_free": slice(1, 1 + cfg.N_UNITS),
                 "load": slice(1 + cfg.N_UNITS, 1 + cfg.N_UNITS + len(cfg.LAMBDA_BAR))}
CRITIC_DIM = 1 + cfg.N_UNITS + len(cfg.LAMBDA_BAR)   # 10
LOADS = sorted(cfg.LAMBDA_BAR)                       # one-hot order for "load"


def _policy_bounds():
    lo, hi = np.zeros(POLICY_DIM, np.float32), np.ones(POLICY_DIM, np.float32)
    for u in range(cfg.N_UNITS):
        b = u * UNIT_DIM
        hi[b + UNIT_FIELDS["eta"]] = hi[b + UNIT_FIELDS["time_to_free"]] = TIME_CLIP
    hi[OBS_SLICES["queue_count"]] = QUEUE_CLIP
    hi[OBS_SLICES["queue_wait"]] = TIME_CLIP
    lo[OBS_SLICES["time_of_day"]] = -1.0
    return lo, hi


POLICY_LOW, POLICY_HIGH = _policy_bounds()
CRITIC_LOW = np.zeros(CRITIC_DIM, np.float32)
CRITIC_HIGH = np.ones(CRITIC_DIM, np.float32)
CRITIC_HIGH[CRITIC_SLICES["lambda"]] = TIME_CLIP     # lambda ratio, clipped at 10 too
CRITIC_HIGH[CRITIC_SLICES["true_time_to_free"]] = TIME_CLIP

OBSERVATION_SPACE = spaces.Dict({
    "policy": spaces.Box(POLICY_LOW, POLICY_HIGH, dtype=np.float32),
    "critic": spaces.Box(CRITIC_LOW, CRITIC_HIGH, dtype=np.float32),
})
ACTION_SPACE = spaces.Discrete(cfg.N_ACTIONS)

# info keys every step must carry
INFO_KEYS = ("dt", "closest_straight", "closest_road", "calls")
# keys of each record in info["calls"] (one per call whose first unit arrived this step)
CALL_RECORD_KEYS = ("call_id", "t_call", "severity", "zone", "unit", "closest_straight",
                    "closest_road", "T", "held", "reward")
