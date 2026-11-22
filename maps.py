"""City maps: 16x16 grid roads, call hotspots, posts, hospitals and cached travel times.

Owner: M2 (tasks T08, T09).

Conventions (CONTRACT.md): a cell is (row, col), 0-15, row 0 at the bottom when plotted.
Flat cell index = row * 16 + col, used to index travel_s.

Maps are looked up by name so every policy runs on exactly the same cities.
Train only on TRAIN_MAPS; HELDOUT_MAPS are for E1/E4 held-out evaluation.

    from maps import get_map
    city = get_map("train_00")

python maps.py        # draw every map to results/maps/ to eyeball them
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from config import P

GRID_SIZE = 16  # 2 x 2 mi, 1/8-mi blocks
N_CELLS = GRID_SIZE * GRID_SIZE
N_POSTS = P["fleet"]["n_posts"]
ROAD_NONE, ROAD_SIDE, ROAD_MAIN, ROAD_HIGHWAY = 0, 1, 2, 3
ZONE_SIZE = P["zones"]["zone_size"]
N_ZONES = (GRID_SIZE // ZONE_SIZE) ** 2

Cell = tuple[int, int]  # (row, col)


@dataclass
class CityMap:
    name: str
    road: np.ndarray  # (16, 16) int: 0 = no road (park/water), 1 = side, 2 = main, 3 = highway
    hotspot: np.ndarray  # (16, 16) float: share of calls per cell, sums to 1, ZERO on non-road cells
    posts: list[Cell]  # N_POSTS cells; unit i's home post is posts[P["fleet"]["unit_posts"][i]]
    hospitals: list[Cell]  # 1-2
    travel_s: np.ndarray | None = None  # (256, 256) road travel seconds between cells (Dijkstra)

    def travel(self, a: Cell, b: Cell) -> float:
        """Road travel seconds from cell a to cell b (no turnout)."""
        return float(self.travel_s[cell_index(a), cell_index(b)])

    def nearest_hospital(self, cell: Cell) -> Cell:
        return min(self.hospitals, key=lambda h: self.travel(cell, h))


def cell_index(cell: Cell) -> int:
    return cell[0] * GRID_SIZE + cell[1]


def zone_of(cell: Cell) -> int:
    """Equity zone of a cell (row-major blocks of ZONE_SIZE x ZONE_SIZE)."""
    per_row = GRID_SIZE // ZONE_SIZE
    return (cell[0] // ZONE_SIZE) * per_row + cell[1] // ZONE_SIZE


def seconds_per_block(road_type: int) -> float:
    """Travel seconds across one 1/8-mi block on this road type (from params.yaml)."""
    name = {ROAD_SIDE: "side", ROAD_MAIN: "main", ROAD_HIGHWAY: "highway"}[road_type]
    mph = P["roads"]["speed_mph"][name]
    return P["roads"]["block_mi"] / mph * 3600


def make_map(name: str, variant: str | None, seed: int) -> CityMap:
    """Build one map: roads (main every 4th row/col), obstacles, hotspots, posts, hospitals.

    Must: zero the hotspot on non-road cells (no calls in parks or water), keep every road
    cell reachable from every post, and place posts and hospitals on road cells.
    """
    raise NotImplementedError


def build_travel_times(city: CityMap) -> None:
    """Fill city.travel_s with all-pairs Dijkstra times over the road grid (seconds_per_block)."""
    raise NotImplementedError


# name -> (variant, seed). Held-out maps must use variants never seen in training (E4).
# TODO(M2): pick the variant families (e.g. hotspot layout, park, island, 1 vs 2 hospitals).
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
