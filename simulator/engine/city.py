"""Road graph for one map file, with all-pairs road times and paths cached."""
import heapq
import json

import numpy as np

from simulator import simulator_config as cfg


class City:
    def __init__(self, m):
        G = self.G = m["grid_size"]
        self.name = m.get("name", "")
        self.n = G * G
        node = lambda rc: rc[0] * G + rc[1]
        blocked = {node(b) for b in m["blocked"]}
        self.posts = [node(p) for p in m["posts"]]
        self.hospitals = [node(h) for h in m["hospitals"]]
        self.unit_posts = list(m["unit_posts"])
        demand = np.asarray(m["demand"], float).reshape(-1)

        # validate at the trust boundary: map files come from outside this module
        assert len(m["row_types"]) == len(m["col_types"]) == G and demand.size == self.n
        assert len(self.posts) == cfg.N_POSTS and 1 <= len(self.hospitals) <= 2
        assert len(self.unit_posts) == cfg.N_UNITS and all(0 <= p < cfg.N_POSTS for p in self.unit_posts)
        assert not blocked & set(self.posts + self.hospitals), "post/hospital on a blocked cell"
        assert (demand >= 0).all() and demand.sum() > 0

        rc = np.array([(i // G, i % G) for i in range(self.n)], float)
        self.coords = rc
        sec = lambda blocks, road: blocks * cfg.BLOCK_MILES / cfg.ROAD_SPEED_MPH[road] * 3600
        adj = [[] for _ in range(self.n)]

        def add(a, b, t):
            if a not in blocked and b not in blocked:
                adj[a].append((b, t))
                adj[b].append((a, t))

        for r in range(G):
            for c in range(G):
                if c + 1 < G:
                    add(node((r, c)), node((r, c + 1)), sec(1, m["row_types"][r]))
                if r + 1 < G:
                    add(node((r, c)), node((r + 1, c)), sec(1, m["col_types"][c]))
        for line in m["highways"]:   # interchanges joined in order; edges may span many blocks
            for a, b in zip(line, line[1:]):
                add(node(a), node(b), sec(float(np.hypot(a[0] - b[0], a[1] - b[1])), "highway"))

        self.T = np.full((self.n, self.n), np.inf)          # road seconds
        self.pred = np.full((self.n, self.n), -1, int)      # pred[s, v] = node before v on s->v path
        for s in range(self.n):
            if s not in blocked:
                self._dijkstra(s, adj)
        self.miles = np.hypot(*(rc[:, None, :] - rc[None, :, :]).transpose(2, 0, 1)) * cfg.BLOCK_MILES

        demand[list(blocked)] = 0.0
        self.demand = demand / demand.sum()
        assert np.isfinite(self.T[np.ix_(self.posts, np.flatnonzero(self.demand))]).all(), \
            "some demand cell is unreachable from a post"

    def _dijkstra(self, s, adj):
        T, pred = self.T[s], self.pred[s]
        T[s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d, v = heapq.heappop(pq)
            if d > T[v]:
                continue
            for w, t in adj[v]:
                if d + t < T[w]:
                    T[w], pred[w] = d + t, v
                    heapq.heappush(pq, (d + t, w))

    def path(self, s, t):
        """Nodes from s to t inclusive along the cached shortest path."""
        out = [t]
        while out[-1] != s:
            out.append(self.pred[s, out[-1]])
        return out[::-1]

    def zone(self, v):
        """4x4-block zones (16 on a 16x16 map), for equity metrics."""
        return (v // self.G) // 4 * (self.G // 4) + (v % self.G) // 4

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))
