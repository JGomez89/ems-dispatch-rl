"""City maps: 16x16 grid roads, call hotspots, posts, hospitals and cached travel times.

Maps are looked up by name (like the named worlds in Assignment 1) so every policy runs
on exactly the same cities. Train only on TRAIN_MAPS; HELDOUT_MAPS are for E4.

    from maps import get_map
    city = get_map("train_00")

python maps.py        # draw every map to results/maps/ to eyeball them
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

GRID_SIZE = 16  # 2 x 2 mi, 1/8-mi blocks
N_POSTS = 4


@dataclass
class CityMap:
    name: str
    road: np.ndarray  # (16, 16) int: 0 = no road (park/water), 1 = side, 2 = main, 3 = highway
    hotspot: np.ndarray  # (16, 16) float: share of calls per cell, sums to 1
    posts: list[tuple[int, int]]
    hospitals: list[tuple[int, int]]  # 1-2
    travel_s: np.ndarray | None = None  # (256, 256) road travel seconds between cells (Dijkstra)


def make_map(name: str, variant: str | None, seed: int) -> CityMap:
    """Build one map: roads (main every 4th row/col), obstacles, hotspots, posts, hospitals."""
    raise NotImplementedError


def build_travel_times(city: CityMap) -> None:
    """Fill city.travel_s with all-pairs Dijkstra times over the road grid."""
    raise NotImplementedError


# name -> (variant, seed). Held-out maps must use variants never seen in training (E4).
# TODO: pick the variant families (e.g. hotspot layout, park, island, 1 vs 2 hospitals).
TRAIN_MAPS = {f"train_{i:02d}": (None, i) for i in range(10)}
HELDOUT_MAPS = {f"heldout_{i:02d}": (None, 100 + i) for i in range(10)}
MAPS = {**TRAIN_MAPS, **HELDOUT_MAPS}


@cache
def get_map(name: str) -> CityMap:
    """Return a built map with travel times. Cached, so Dijkstra runs once per map per process."""
    variant, seed = MAPS[name]
    city = make_map(name, variant, seed)
    build_travel_times(city)
    return city


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    from plot import draw_map

    out = Path("results/maps")
    out.mkdir(parents=True, exist_ok=True)
    for name in MAPS:
        fig, ax = plt.subplots(figsize=(4, 4))
        draw_map(get_map(name), ax)
        fig.savefig(out / f"{name}.png", dpi=120)
        plt.close(fig)
    print(f"wrote {len(MAPS)} maps to {out}/")
