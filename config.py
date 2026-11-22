"""Shared constants, loaded once from config/params.yaml and config/lambda_bar.json.

    from config import P, lambda_bar
    P["time"]["turnout_s"]          # 60
    lambda_bar("train_00", 0.6)     # calls per hour that gives 60% utilization

Owner: M2. Values marked X in params.yaml are placeholders until the DataSF fits land.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / "config" / "params.yaml"
LAMBDA_BAR_PATH = ROOT / "config" / "lambda_bar.json"  # written by calibrate.py (T17)

with open(PARAMS_PATH) as f:
    P: dict = yaml.safe_load(f)

N_UNITS: int = P["fleet"]["n_units"]
HOLD: int = N_UNITS  # action index for HOLD
N_ACTIONS: int = N_UNITS + 1
N_SEVERITIES = 3


@cache
def _lambda_bar_table() -> dict:
    if not LAMBDA_BAR_PATH.exists():
        return {}
    with open(LAMBDA_BAR_PATH) as f:
        return json.load(f)


def lambda_bar(map_name: str, load: float) -> float:
    """Mean calls/hour for this map at this target load. Run calibrate.py first.

    Until calibration exists, returns a placeholder so the stub simulator and tests run.
    """
    table = _lambda_bar_table()
    key = f"{load:.1f}"
    if map_name in table and key in table[map_name]:
        return float(table[map_name][key])
    return 4.0 * load / 0.6  # PLACEHOLDER, not calibrated
