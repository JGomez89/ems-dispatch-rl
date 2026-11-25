"""Gymnasium semi-MDP: one step() = one dispatch decision. Contract: agent_interface/CONTRACT.md."""
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from simulator import simulator_config as cfg
from simulator.agent_interface import contract as C
from simulator.engine.city import City
from simulator.engine.sim import BUSY, FREE, TRANSPORTING, UNLOADING, Sim

DEFAULT_MAP = Path(__file__).resolve().parents[1] / "maps" / "default.json"
CRITICAL = 0


def reward_d(sev, T):
    """Proposal reward (arm D). Swap via DispatchEnv(reward_fn=...) for E2 arms A-C."""
    w, tau = cfg.REWARD_W[sev], cfg.REWARD_TAU[sev]
    if T <= tau:
        return w
    return -w * min(((T - tau) / cfg.REWARD_SIGMA[sev]) ** cfg.REWARD_K[sev], cfg.REWARD_LATE_CAP)


class DispatchEnv(gym.Env):
    observation_space = C.OBSERVATION_SPACE
    action_space = C.ACTION_SPACE

    def __init__(self, maps=None, loads=None, reward_fn=reward_d):
        self.cities = [City.load(p) for p in (maps or [DEFAULT_MAP])]
        self.loads = list(loads or C.LOADS)
        self.reward_fn = reward_fn

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        r, o = self.np_random, options or {}
        # always draw all four in this order so the same seed gives the same episode whatever is pinned
        m, load = int(r.integers(len(self.cities))), self.loads[r.integers(len(self.loads))]
        hour, sim_seed = int(r.integers(24)), int(r.integers(2**31))
        self.map_id = o.get("map_id", m)
        self.sim = Sim(self.cities[self.map_id], o.get("load", load), o.get("start_hour", hour), sim_seed)
        self.call = self.sim.advance()
        assert self.call is not None, "episode produced no dispatch decision"
        self.t_prev = self.sim.now
        return self._obs(), self._info(0.0, [])

    def step(self, action):
        action = int(action)
        assert self.action_masks()[action], f"masked action {action}"
        self.sim.dispatch(self.call, action)
        self.call = self.sim.advance()
        records = [self._record(c) for c in self.sim.pop_credited()]
        dt, self.t_prev = self.sim.now - self.t_prev, self.sim.now
        reward = float(sum(rec["reward"] for rec in records))
        return self._obs(), reward, self.call is None, False, self._info(dt, records)

    def action_masks(self):
        if self.call is None:   # terminal: nothing to decide
            return np.ones(cfg.N_ACTIONS, bool)
        s, units = self.sim, self.sim.units
        mask = np.array([u.state in FREE for u in units] + [False])
        hold = any(u.state in BUSY for u in units)   # else a held call would never be served
        if hold and self.call.sev == CRITICAL:
            node = self.call.node
            best_free = min(s.eta(u, node) for u in units if u.state in FREE)
            best_busy = min((s.eta(u, node) for u in units if u.state in (TRANSPORTING, UNLOADING)),
                            default=math.inf)
            hold = (best_busy <= best_free - cfg.HOLD_ADVANTAGE_S
                    and not any(q.sev == CRITICAL for q in s.queue))
        mask[cfg.HOLD_ACTION] = hold
        return mask

    # ---- contract encoding ----------------------------------------------
    def _obs(self):
        s, c, norm = self.sim, self.call, cfg.TIME_NORM_S
        pol = np.zeros(C.POLICY_DIM, np.float32)
        if c is not None:
            etas = [s.eta(u, c.node) for u in s.units]
            rank = np.argsort(np.argsort(etas, kind="stable"), kind="stable")
            cov = s.coverage_lost()
            for i, u in enumerate(s.units):
                b, f = i * C.UNIT_DIM, C.UNIT_FIELDS
                pol[b + u.state] = 1.0
                pol[b + f["eta"]] = etas[i] / norm
                pol[b + f["rank"]] = rank[i] / (cfg.N_UNITS - 1)
                pol[b + f["time_to_free"]] = s.time_to_free(u) / norm
                pol[b + f["coverage_lost"]] = cov[i]
            pol[C.OBS_SLICES["severity"].start + c.sev] = 1.0
            for k in range(C.N_SEV):
                q = [x for x in s.queue if x.sev == k]
                pol[C.OBS_SLICES["queue_count"].start + k] = len(q) / cfg.N_UNITS
                pol[C.OBS_SLICES["queue_wait"].start + k] = max((s.now - x.t for x in q), default=0.0) / norm
            for u in s.units:
                pol[C.OBS_SLICES["fleet"].start + u.state] += 1.0 / cfg.N_UNITS
        hour = (s.start_hour + s.now / 3600) % 24
        pol[C.OBS_SLICES["time_of_day"]] = (math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24))

        crit = np.zeros(C.CRITIC_DIM, np.float32)
        crit[C.CRITIC_SLICES["lambda"]] = s.lam(s.now) / cfg.LAMBDA_BAR[0.6]
        crit[C.CRITIC_SLICES["true_time_to_free"]] = [s.time_to_free(u, true=True) / norm for u in s.units]
        crit[C.CRITIC_SLICES["load"].start + C.LOADS.index(s.load)] = 1.0
        return {"policy": np.clip(pol, C.POLICY_LOW, C.POLICY_HIGH),
                "critic": np.clip(crit, C.CRITIC_LOW, C.CRITIC_HIGH)}

    def _info(self, dt, records):
        straight, road = self.sim.closest(self.call) if self.call is not None else (-1, -1)
        return {"dt": dt, "closest_straight": straight, "closest_road": road, "calls": records}

    def _record(self, c):
        T = c.t_first - c.t
        return dict(call_id=c.id, t_call=c.t, severity=c.sev, zone=self.sim.city.zone(c.node),
                    unit=c.unit, closest_straight=c.closest_straight, closest_road=c.closest_road,
                    T=T, held=c.held, reward=self.reward_fn(c.sev, T))
