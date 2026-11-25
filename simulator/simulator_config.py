"""Simulator configuration.

Values fixed by the proposal are final.
Values marked PLACEHOLDER are dummies until Anjali delivers the DataSF fits.
"""

# ---------------------------------------------------------------------------
# Map (fixed by proposal)
# ---------------------------------------------------------------------------
GRID_SIZE = 16                # 16 x 16 intersections
BLOCK_MILES = 1 / 8           # 2 x 2 mi city
MAIN_ROAD_EVERY = 4           # every 4th row/col is a main road

# Road speeds in mph                                      # PLACEHOLDER
ROAD_SPEED_MPH = {
    "side": 15.0,
    "main": 25.0,
    "highway": 45.0,
}

# ---------------------------------------------------------------------------
# Fleet (fixed by proposal)
# ---------------------------------------------------------------------------
N_UNITS = 6
N_POSTS = 4
N_HOSPITALS = 2               # proposal allows 1-2; per-map override later

UNIT_STATES = ["IDLE_AT_POST", "EN_ROUTE", "ON_SCENE",
               "TRANSPORTING", "UNLOADING", "RETURNING"]   # one-hot order in obs

# ---------------------------------------------------------------------------
# Severity tiers
# ---------------------------------------------------------------------------
SEVERITIES = ["critical", "urgent", "non_urgent"]   # index 0, 1, 2

SEVERITY_SHARES = [0.20, 0.40, 0.40]                # PLACEHOLDER (pi_s)
TRANSPORT_PROB = [0.80, 0.60, 0.30]                 # PLACEHOLDER (p_s)

# ---------------------------------------------------------------------------
# Call arrivals: lambda(t) = LAMBDA_BAR[load] * HOURLY_WEIGHTS[hour]
# ---------------------------------------------------------------------------
# 24 weights, mean 1.0. Rough day/night shape.            # PLACEHOLDER
HOURLY_WEIGHTS = [
    0.6, 0.5, 0.45, 0.4, 0.4, 0.5,     # 00-05
    0.7, 0.9, 1.1, 1.2, 1.25, 1.3,     # 06-11
    1.3, 1.3, 1.3, 1.3, 1.3, 1.3,      # 12-17
    1.25, 1.2, 1.1, 1.0, 0.85, 0.7,    # 18-23
]
_mean = sum(HOURLY_WEIGHTS) / len(HOURLY_WEIGHTS)
HOURLY_WEIGHTS = [w / _mean for w in HOURLY_WEIGHTS]

# Mean calls per hour for each target utilization.        # PLACEHOLDER
# Real values come from calibrating closest-unit dispatch to 40/60/80% busy.
LAMBDA_BAR = {
    0.4: 3.6,
    0.6: 5.4,
    0.8: 7.2,
}

# ---------------------------------------------------------------------------
# Service times (seconds)
# ---------------------------------------------------------------------------
TURNOUT_S = 60                # fixed by proposal

# Lognormal: (median_s, sigma of underlying normal)        # PLACEHOLDER
ON_SCENE_LOGNORMAL = (900.0, 0.5)
UNLOAD_LOGNORMAL = (1200.0, 0.5)

# ---------------------------------------------------------------------------
# Reward (fixed by proposal), indexed by severity
# ---------------------------------------------------------------------------
REWARD_W = [3.0, 1.5, 0.5]          # importance weight w_s
REWARD_TAU = [300, 300, 600]        # on-time target tau_s (s)
REWARD_SIGMA = [60, 120, 300]       # lateness scale sigma_s (s)
REWARD_K = [2, 2, 1]                # lateness exponent k_s
REWARD_LATE_CAP = 10.0

# ---------------------------------------------------------------------------
# Episode (fixed by proposal)
# ---------------------------------------------------------------------------
EPISODE_HOURS = 6             # arrivals stop after this, then drain the queue

# ---------------------------------------------------------------------------
# Agent interface (fixed by proposal)
# ---------------------------------------------------------------------------
N_ACTIONS = N_UNITS + 1       # send unit 0-5, or HOLD
HOLD_ACTION = N_UNITS
TIME_NORM_S = 240.0           # observation times divided by this
HOLD_ADVANTAGE_S = 30         # busy unit must beat best free unit by this
CALL_RATE_WINDOW_S = 3600     # observed-rate moving average window

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DEFAULT_SEED = 0