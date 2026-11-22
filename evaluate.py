"""Run policies on the same maps/seeds and report metrics (on-time rate, p90, HOLD rate, ...).

python evaluate.py --policy closest_travel_time
python evaluate.py --policy results/ppo_D_seed0.zip --maps heldout

Writes one row per call to results/eval_<policy>.csv; plot.py makes figures from those.
"""

import argparse

from baselines import POLICIES
from env import EMSDispatchEnv


def load_policy(name_or_path):
    """A name in baselines.POLICIES, or a path to a saved MaskablePPO .zip.

    Returns policy(env, obs) -> action, the same signature as the baselines.
    """
    if name_or_path in POLICIES:
        return POLICIES[name_or_path]
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(name_or_path)

    def ppo_policy(env, obs):
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(action)

    return ppo_policy


def run_episode(policy, map_name, seed, load, record=False):
    """Play one episode. Return (list of finished calls with response times, env)."""
    raise NotImplementedError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--maps", default="train", choices=["train", "heldout"])
    p.add_argument("--load", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    p.add_argument("--episodes", type=int, default=10)
    args = p.parse_args()
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
