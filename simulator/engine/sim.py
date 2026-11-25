"""Discrete-event EMS simulation. No Gymnasium, no RL: the env drives it via advance()/dispatch()."""
import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from simulator import simulator_config as cfg

IDLE, EN_ROUTE, ON_SCENE, TRANSPORTING, UNLOADING, RETURNING = range(6)   # cfg.UNIT_STATES order
FREE = (IDLE, RETURNING)                                 # dispatchable
BUSY = (EN_ROUTE, ON_SCENE, TRANSPORTING, UNLOADING)     # will be freed later
HOLD = cfg.HOLD_ACTION
MED_SCENE, SIG_SCENE = cfg.ON_SCENE_LOGNORMAL
MED_UNLOAD, SIG_UNLOAD = cfg.UNLOAD_LOGNORMAL


@dataclass(eq=False)
class Call:
    id: int
    t: float
    node: int
    sev: int
    on_scene: float      # pre-drawn so every policy sees the same episode
    transport: bool
    unload: float
    hospital: int
    unit: int = -1
    t_first: float = math.nan   # first unit arrival
    held: bool = False
    closest_straight: int = -1
    closest_road: int = -1


@dataclass(eq=False)
class Unit:
    id: int
    post: int
    node: int                    # where it is when not driving
    state: int = IDLE
    call: Call = None
    phase_start: float = 0.0
    timetable: list = field(default_factory=list)   # [(node, arrival_time)] of current drive
    drive_id: int = 0                               # bumps per drive; stale events are dropped


class Sim:
    def __init__(self, city, load, start_hour, seed):
        self.city, self.load, self.start_hour = city, load, start_hour
        self.now = 0.0
        self.units = [Unit(i, city.posts[p], city.posts[p]) for i, p in enumerate(city.unit_posts)]
        self.queue, self.credited, self.log = [], [], []
        self.events, self._seq = [], 0
        self.calls = self._gen_calls(np.random.default_rng(seed))
        for c in self.calls:
            self._push(c.t, "CALL_ARRIVAL", c, 0)

    # ---- arrivals --------------------------------------------------------
    def lam(self, t):
        """True call rate (calls/hour) at sim time t."""
        return cfg.LAMBDA_BAR[self.load] * cfg.HOURLY_WEIGHTS[int(self.start_hour + t // 3600) % 24]

    def _gen_calls(self, rng):
        c, end, t, calls = self.city, cfg.EPISODE_HOURS * 3600, 0.0, []
        while t < end:   # piecewise-constant Poisson, restarted at each hour boundary (memoryless)
            hour_end = (t // 3600 + 1) * 3600
            t_next = t + rng.exponential(3600 / self.lam(t))
            if t_next >= hour_end:
                t = hour_end
                continue
            t = t_next
            node = int(rng.choice(c.n, p=c.demand))
            sev = int(rng.choice(len(cfg.SEVERITIES), p=cfg.SEVERITY_SHARES))
            scene = MED_SCENE * math.exp(SIG_SCENE * rng.standard_normal())
            transport = bool(rng.random() < cfg.TRANSPORT_PROB[sev])
            unload = MED_UNLOAD * math.exp(SIG_UNLOAD * rng.standard_normal())
            hosp = min(c.hospitals, key=lambda h: c.T[node, h])
            calls.append(Call(len(calls), t, node, sev, scene, transport, unload, hosp))
        return calls

    # ---- event loop ------------------------------------------------------
    def _push(self, t, kind, obj, drive_id):
        heapq.heappush(self.events, (t, self._seq, kind, obj, drive_id))
        self._seq += 1

    def advance(self):
        """Run events until a call needs a decision (returned) or everything is done (None)."""
        while self.events:
            t, _, kind, obj, drive_id = heapq.heappop(self.events)
            if kind != "CALL_ARRIVAL" and drive_id != obj.drive_id:
                continue   # a returning unit was re-dispatched; its ARRIVE_POST is stale
            self.now = t
            unit = None if kind == "CALL_ARRIVAL" else obj
            call = obj if unit is None else obj.call
            self.log.append((t, kind, unit and unit.id, call and call.id))
            if kind == "CALL_ARRIVAL":
                if any(u.state in FREE for u in self.units):
                    return obj
                self.queue.append(obj)
            elif kind == "UNIT_ARRIVES_SCENE":
                u, c = obj, obj.call
                u.state, u.node, u.phase_start = ON_SCENE, c.node, t
                c.t_first = t
                self.credited.append(c)
                self._push(t + c.on_scene, "SCENE_DONE", u, u.drive_id)
            elif kind == "SCENE_DONE":
                if obj.call.transport:
                    obj.state = TRANSPORTING
                    self._drive(obj, obj.node, t, obj.call.hospital, "ARRIVE_HOSPITAL")
                else:
                    self._free(obj)
            elif kind == "ARRIVE_HOSPITAL":
                obj.state, obj.node, obj.phase_start = UNLOADING, obj.call.hospital, t
                self._push(t + obj.call.unload, "UNLOAD_DONE", obj, obj.drive_id)
            elif kind == "UNLOAD_DONE":
                self._free(obj)
            elif kind == "ARRIVE_POST":
                obj.state, obj.node, obj.timetable = IDLE, obj.post, []
        return None

    def dispatch(self, call, action):
        call.closest_straight, call.closest_road = self.closest(call)
        if action == HOLD:
            call.held = True
            self.queue.append(call)
            return
        u = self.units[action]
        assert u.state in FREE, f"unit {action} is busy"
        self._assign(u, call, *self.position(u))

    def pop_credited(self):
        out, self.credited = self.credited, []
        return out

    def _drive(self, u, src, t0, dst, arrive_event):
        u.drive_id += 1
        u.timetable = [(v, t0 + self.city.T[src, v]) for v in self.city.path(src, dst)]
        self._push(u.timetable[-1][1], arrive_event, u, u.drive_id)

    def _assign(self, u, call, src, t0):
        call.unit, u.call, u.state = u.id, call, EN_ROUTE
        self._drive(u, src, t0, call.node, "UNIT_ARRIVES_SCENE")

    def _free(self, u):
        """Fixed rule: take the most severe waiting call (oldest first), else return to post."""
        u.call = None
        if self.queue:
            c = min(self.queue, key=lambda q: (q.sev, q.t))
            self.queue.remove(c)
            self._assign(u, c, u.node, self.now)   # no turnout: freed away from post
        else:
            u.state = RETURNING
            self._drive(u, u.node, self.now, u.post, "ARRIVE_POST")

    # ---- queries the env builds observations from ------------------------
    def position(self, u):
        """(node, time) a free unit can start a new route from. Mid-drive: finish the current edge."""
        if u.state == IDLE:
            return u.post, self.now + cfg.TURNOUT_S
        return next((v, t) for v, t in u.timetable if t >= self.now)

    def outcomes(self, u, true=False):
        """[(prob, time_until_free, node_freed_at)] for a busy unit. Expected uses medians and p_s."""
        if u.state in FREE:
            return [(1.0, 0.0, u.node)]
        c, now, T = u.call, self.now, self.city.T
        scene = c.on_scene if true else MED_SCENE
        unload = c.unload if true else MED_UNLOAD
        if u.state in (EN_ROUTE, ON_SCENE):
            left = (u.timetable[-1][1] - now + scene) if u.state == EN_ROUTE \
                else max(0.0, u.phase_start + scene - now)   # shortcut: median not conditioned on elapsed time
            with_t = (left + T[c.node, c.hospital] + unload, c.hospital)
            if true:
                return [(1.0, *with_t)] if c.transport else [(1.0, left, c.node)]
            p = cfg.TRANSPORT_PROB[c.sev]
            return [(1 - p, left, c.node), (p, *with_t)]
        if u.state == TRANSPORTING:
            return [(1.0, u.timetable[-1][1] - now + unload, c.hospital)]
        return [(1.0, max(0.0, u.phase_start + unload - now), c.hospital)]   # UNLOADING

    def time_to_free(self, u, true=False):
        return sum(p * t for p, t, _ in self.outcomes(u, true))

    def eta(self, u, node):
        if u.state in FREE:
            v, t = self.position(u)
            return t - self.now + self.city.T[v, node]
        return sum(p * (t + self.city.T[v, node]) for p, t, v in self.outcomes(u))

    def closest(self, call):
        """(by straight-line distance, by ETA) among free units: the two baselines and BC labels."""
        free = [u for u in self.units if u.state in FREE]
        straight = min(free, key=lambda u: self.city.miles[self.position(u)[0], call.node])
        road = min(free, key=lambda u: self.eta(u, call.node))
        return straight.id, road.id

    def coverage_lost(self, radius_s=240.0):
        """Per unit: demand share covered by idle-at-post units minus the share without it."""
        idle = [u for u in self.units if u.state == IDLE]
        T, d = self.city.T, self.city.demand
        cov = lambda us: float(d[(T[[u.post for u in us]] <= radius_s).any(0)].sum()) if us else 0.0
        full = cov(idle)
        return [full - cov([v for v in idle if v is not u]) if u in idle else 0.0 for u in self.units]
