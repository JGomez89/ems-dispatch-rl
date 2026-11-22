"""Call generator: the whole call stream for one episode, drawn up front.

Owner: M2 (task T10). Split out of sim.py so M1 and M2 never edit the same file.

Everything random about a call (time, cell, severity, transport, service times) is drawn
here from the CALL seed at reset. The simulator never draws during the run, so every
policy faces exactly the same calls and service times (common random numbers).
"""

from __future__ import annotations

import numpy as np

from config import P
from maps import CityMap
from sim import Call


def generate_calls(
    city: CityMap,
    lam_bar_per_h: float,
    start_hour: int,
    rng: np.random.Generator,
    surge_start_s: float | None = None,
) -> list[Call]:
    """Non-homogeneous Poisson calls over P["time"]["episode_hours"].

    lambda(t) = lam_bar * hourly_weight[(start_hour + t // 3600) % 24], multiplied by
    P["surge"]["multiplier"] for P["surge"]["duration_s"] starting at surge_start_s (E4).
    Cells are drawn from city.hotspot; severity from severity_share; transport with
    transport_prob[s]; on-scene and unloading times from the lognormals in params.yaml.
    Returns calls sorted by time_s with ids 0, 1, 2, ...
    """
    raise NotImplementedError


def true_rate_per_h(lam_bar_per_h: float, start_hour: int, t_s: float,
                    surge_start_s: float | None = None) -> float:
    """lambda(t) in calls/hour. Privileged: critic only, never the actor (CONTRACT.md)."""
    h = P["calls"]["hourly_weight"][(start_hour + int(t_s // 3600)) % 24]
    rate = lam_bar_per_h * h
    if surge_start_s is not None and surge_start_s <= t_s < surge_start_s + P["surge"]["duration_s"]:
        rate *= P["surge"]["multiplier"]
    return rate
