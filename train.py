"""Train one agent for E1-E3. Owner: M3 (PPO), with M4's bc.py for the BC arms.

python train.py --arm scratch --reward D --seed 0
python train.py --arm bc_ppo_aux --seed 2
python train.py --arm scratch --reward A --seed 0 --resume     # continue after a laptop restart

Trains on maps.TRAIN_MAPS only, at the loads in --loads (OPEN DECISION D7: mixed vs. 60% only).
Writes, under results/<run_name>/:
    model.zip          final policy (evaluate.load_policy and plot.py read it)
    ckpt_<steps>.zip   wall-clock checkpoints, every --checkpoint-min minutes
    progress.csv       SB3 logger output for plot.learning_curves

Run names: <arm>_<reward>_seed<n>, e.g. scratch_D_seed0, bc_ppo_D_seed3, scratch_A_seed1.
The reward-D scratch run is the E1 agent, the E2 arm D and the E3 "scratch" arm at once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ARMS = ["scratch", "bc_only", "bc_ppo", "bc_ppo_aux"]
RESULTS = Path("results")


def run_name(arm: str, reward: str, seed: int) -> str:
    return f"{arm}_{reward}_seed{seed}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="scratch", choices=ARMS)
    p.add_argument("--reward", default="D", choices=["A", "B", "C", "D"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000,
                   help="PPO decisions. Set from laptop profiling (T18) so one run fits ~5 h")
    p.add_argument("--loads", type=float, nargs="+", default=[0.4, 0.6, 0.8])
    p.add_argument("--checkpoint-min", type=float, default=30)
    p.add_argument("--resume", action="store_true", help="continue from the newest ckpt_*.zip")
    p.add_argument("--stub", action="store_true", help="use stub_sim.StubSimulator (pipeline test only)")
    args = p.parse_args()

    out = RESULTS / run_name(args.arm, args.reward, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    # TODO(M3):
    #   env = EMSDispatchEnv(maps=list(TRAIN_MAPS), loads=args.loads, reward=args.reward,
    #                        seed=args.seed, sim_cls=StubSimulator if args.stub else Simulator)
    #   model = SMDPMaskablePPO("PrivilegedCriticPolicy", env, seed=args.seed,
    #                           policy_kwargs=make_policy_kwargs(), ...)  or load newest ckpt if --resume
    #   BC arms: bc.collect_dataset / pretrain_actor / warmup_critic (M4); bc_only saves and stops
    #   callbacks: wall-clock checkpoint every args.checkpoint_min, CSV logger to out/progress.csv
    #   model.learn(args.steps, reset_num_timesteps=not args.resume); model.save(out / "model.zip")
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
