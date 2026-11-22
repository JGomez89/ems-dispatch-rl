"""Find lambda-bar for each map and load level (task T17). Owner: M2.

For every map in maps.MAPS and every load in params.yaml "loads", search for the mean call
rate (calls/hour) at which closest_travel_time dispatch keeps units busy that fraction of
the time, averaged over a few warm-up episodes. Also record each map's historical mean
call rate, used to normalize the observed-rate feature (env.py).

Writes config/lambda_bar.json, which IS committed so everyone trains on the same rates:
    {"train_00": {"0.4": 2.7, "0.6": 4.1, "0.8": 5.6, "mean_rate": 4.1}, ...}

python calibrate.py
"""

from __future__ import annotations


def utilization(map_name: str, lam_bar_per_h: float, n_episodes: int = 5) -> float:
    """Mean fraction of unit-time spent busy under closest_travel_time at this rate."""
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
