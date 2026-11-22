"""Train PPO (MaskablePPO), optionally warm-started with behavior cloning (E3).

python train.py --reward D --seed 0
python train.py --bc --seed 0

Trains on maps.TRAIN_MAPS only. Saves results/ppo_<reward>[_bc]_seed<n>.zip (loadable by
evaluate.py and plot.py) and a progress CSV next to it for plot.learning_curves.
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward", default="D", choices=["A", "B", "C", "D"])
    p.add_argument("--bc", action="store_true", help="behavior-cloning warm start")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=1_000_000)
    args = p.parse_args()
    raise NotImplementedError(args)


if __name__ == "__main__":
    main()
