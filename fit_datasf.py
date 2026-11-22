"""Fit the simulator's X values from DataSF Fire Department Calls for Service (task T07). Owner: M2.

Download the dataset CSV from data.sfgov.org into data/raw/ (not committed), then:

    python fit_datasf.py data/raw/Fire_Department_Calls_for_Service.csv

Prints the fitted values in params.yaml format. Paste them into config/params.yaml and tell
the team (the file is shared and committed). Fits, per the proposal:
    calls.hourly_weight       24 weights from call counts by hour, normalized to mean 1
    calls.severity_share      shares of critical / urgent / non-urgent (map call types first)
    calls.transport_prob      share transported, by severity
    calls.on_scene_*          lognormal median and log-sd of on-scene time
    calls.unloading_*         lognormal median and log-sd of hospital time
    roads.speed_mph           optional sanity check against en-route times
"""

from __future__ import annotations

import sys


def main(path: str) -> None:
    raise NotImplementedError(path)


if __name__ == "__main__":
    main(sys.argv[1])
