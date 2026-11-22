"""Figures for the report and an episode animation. Reads results/, never trains or evaluates.

python plot.py animate --policy closest_travel_time --map train_00 --seed 0
python plot.py animate --policy results/ppo_D_seed0.zip --map heldout_03 --seed 7 --out results/ppo.gif
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sim import UnitStatus

STATUS_COLOR = {
    UnitStatus.IDLE_AT_POST: "tab:green",
    UnitStatus.EN_ROUTE: "tab:red",
    UnitStatus.ON_SCENE: "tab:orange",
    UnitStatus.TRANSPORTING: "tab:purple",
    UnitStatus.UNLOADING: "tab:brown",
    UnitStatus.RETURNING: "tab:blue",
}
SEVERITY_MARKER = {0: "X", 1: "^", 2: "o"}  # critical, urgent, non-urgent


def draw_map(city, ax):
    """Roads, hotspot shading, posts (squares) and hospitals (crosses)."""
    ax.imshow(city.hotspot, cmap="Greys", alpha=0.4, origin="lower")
    ax.imshow(city.road, cmap="Blues", alpha=0.3, origin="lower")
    ax.scatter(*zip(*city.posts), marker="s", s=80, facecolors="none", edgecolors="k", label="post")
    ax.scatter(*zip(*city.hospitals), marker="P", s=120, c="tab:red", label="hospital")
    ax.set_title(city.name)
    ax.set_xticks([])
    ax.set_yticks([])


def learning_curves(csv_paths, out):
    """E3: episode return vs. env steps, one line per run (mean +- 95% CI over seeds)."""
    raise NotImplementedError


def tier_bars(eval_csvs, out):
    """E1/E2: on-time rate and p90 response time per severity tier, one bar group per policy."""
    raise NotImplementedError


def animate_episode(policy_name, map_name, seed, load=0.6, out="results/episode.gif", fps=10):
    """Replay one episode from the simulator trace (needs Simulator(record=True) snapshots)."""
    from evaluate import load_policy, run_episode

    _, env = run_episode(load_policy(policy_name), map_name, seed, load, record=True)
    trace = env.sim.trace

    fig, ax = plt.subplots(figsize=(5, 5))

    def frame(i):
        snap = trace[i]
        ax.clear()
        draw_map(env.sim.city, ax)
        for call in snap.waiting:
            ax.scatter(*call.location, marker=SEVERITY_MARKER[int(call.severity)], s=70, c="k")
        for (x, y), status in snap.units:
            ax.scatter(x, y, s=60, c=STATUS_COLOR[status])
        if snap.dispatch is not None:
            unit_idx, (cx, cy) = snap.dispatch
            ux, uy = snap.units[unit_idx][0]
            ax.annotate("", (cx, cy), (ux, uy), arrowprops=dict(arrowstyle="->", color="tab:red"))
        h, m = divmod(int(snap.time_s) // 60, 60)
        ax.set_title(f"{map_name} | {policy_name} | t = {h:02d}:{m:02d}")

    anim = FuncAnimation(fig, frame, frames=len(trace), interval=1000 / fps)
    anim.save(out, writer="pillow", fps=fps)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("animate")
    a.add_argument("--policy", required=True, help="baseline name or path to a PPO .zip")
    a.add_argument("--map", default="train_00")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--load", type=float, default=0.6)
    a.add_argument("--out", default="results/episode.gif")
    args = p.parse_args()
    if args.cmd == "animate":
        animate_episode(args.policy, args.map, args.seed, args.load, args.out)


if __name__ == "__main__":
    main()
