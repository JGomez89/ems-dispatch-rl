"""Run any policy on fixed maps/loads/call seeds and write one row per call. Owner: M4 (T15).

python evaluate.py --policy closest_travel_time
python evaluate.py --policy results/scratch_D_seed0/model.zip --maps heldout
python evaluate.py --policy results/scratch_D_seed0/model.zip --maps heldout --surge   # E4

Every policy sees the same call sequences: episode k on map m at load L always uses
call_seed = eval_call_seed_offset + k (params.yaml), which no training episode uses.

Writes results/eval/<policy>_<maps>[_surge].csv with CSV_COLUMNS; plot.py reads those.
Metrics (proposal section 5) are computed from the CSV, not during the run.
"""

from __future__ import annotations

import argparse

from baselines import POLICIES
from config import P

CSV_COLUMNS = [
    "policy", "maps", "surge", "map", "load", "episode", "call_seed",
    "call_id", "severity", "zone", "arrival_s", "unit", "unit_arrival_s", "response_s",
    "on_time", "non_closest", "held", "decided_by",   # decided_by: "agent" or "rule"
]


def eval_call_seed(episode: int) -> int:
    return P["seeds"]["eval_call_seed_offset"] + episode


def load_policy(name_or_path):
    """A name in baselines.POLICIES, or a path to a saved model .zip.

    Returns policy(obs, mask, info) -> action, the same signature as the baselines.
    """
    if name_or_path in POLICIES:
        return POLICIES[name_or_path]
    from ppo_smdp import SMDPMaskablePPO

    model = SMDPMaskablePPO.load(name_or_path, device="cpu")

    def ppo_policy(obs, mask, info):
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)

    return ppo_policy


def run_episode(policy, map_name, load, call_seed, surge=False, record=False, sim_cls=None):
    """Play one episode with env.reset(options={"map":..., "load":..., "call_seed":...}).

    Return (rows, env): one CSV_COLUMNS dict per sim.Reached in info["reached"], and the env
    (plot.animate_episode reads env.sim.trace).
    """
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--maps", default="train", choices=["train", "heldout"])
    p.add_argument("--load", type=float, nargs="+", default=P["loads"])
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--surge", action="store_true", help="E4: lambda doubled for one hour")
    args = p.parse_args()
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
