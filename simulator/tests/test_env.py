"""Acceptance checks. Run from the repo root: pytest"""
import numpy as np

from simulator import simulator_config as cfg
from simulator.agent_interface import contract as C
from simulator.engine.sim import BUSY, FREE
from simulator.environment.dispatch_env import DispatchEnv, reward_d
from simulator.environment.mock_env import MockDispatchEnv


def check_step(env, obs, info):
    assert env.observation_space.contains(obs), "obs outside contract space/dtype"
    assert set(C.INFO_KEYS) <= info.keys()
    assert info["dt"] >= 0
    for rec in info["calls"]:
        assert set(rec) == set(C.CALL_RECORD_KEYS)


def check_real_mask(env, mask):
    units = env.sim.units
    for i, u in enumerate(units):
        assert not mask[i] or u.state in FREE, "mask allows a busy unit"
    if not any(u.state in BUSY for u in units):
        assert not mask[cfg.HOLD_ACTION], "HOLD allowed with nothing to wait for"


def run(env, seed):
    """One episode under a uniformly random legal policy. Returns the trajectory."""
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    check_step(env, obs, info)
    traj, done, predicted = [], False, {}
    while not done:
        mask = env.action_masks()
        assert mask.dtype == bool and mask.shape == (cfg.N_ACTIONS,) and mask.any()
        if isinstance(env, DispatchEnv):
            check_real_mask(env, mask)
        a = int(rng.choice(np.flatnonzero(mask)))
        if isinstance(env, DispatchEnv) and a != cfg.HOLD_ACTION:
            predicted[env.call.id] = env.sim.eta(env.sim.units[a], env.call.node)
        obs, r, done, trunc, info = env.step(a)
        check_step(env, obs, info)
        assert not trunc
        for rec in info["calls"]:   # timetable rule: predicted ETA is the real response time
            if rec["call_id"] in predicted and isinstance(env, DispatchEnv):
                assert abs(rec["T"] - predicted[rec["call_id"]]) < 1e-6, (rec, predicted[rec["call_id"]])
        traj.append((a, r, info["dt"], obs["policy"].tobytes(), obs["critic"].tobytes()))
    return traj


def test_interface_and_determinism():
    for Env in (MockDispatchEnv, DispatchEnv):
        for seed in range(3):
            assert run(Env(), seed) == run(Env(), seed), f"{Env.__name__} not deterministic"


def test_real_env_serves_every_call():
    env, held = DispatchEnv(), 0
    for seed in range(6):
        for load in C.LOADS:
            env.reset(seed=seed, options={"load": load})
            rng = np.random.default_rng(seed)
            done = False
            while not done:
                mask = env.action_masks()
                check_real_mask(env, mask)
                _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
            calls = env.sim.calls
            assert not env.sim.queue and all(np.isfinite(c.t_first) for c in calls), "unserved call"
            held += sum(c.held for c in calls)
    assert held > 0, "random policy never held a call; HOLD path untested"


def test_same_calls_across_policies():
    a, b = DispatchEnv(), DispatchEnv()
    a.reset(seed=7, options={"load": 0.6})
    b.reset(seed=7, options={"load": 0.6})
    key = lambda env: [(c.t, c.node, c.sev, c.on_scene, c.transport, c.unload) for c in env.sim.calls]
    assert key(a) == key(b)
    for env, pick in ((a, lambda e: e.action_masks()[:cfg.N_UNITS].argmax()),
                      (b, lambda e: np.flatnonzero(e.action_masks())[-1])):
        done = False
        while not done:
            _, _, done, _, _ = env.step(int(pick(env)))
    assert key(a) == key(b), "call sequence depended on the policy"


def test_reward_hand_computed():
    cases = [(0, 300, 3.0), (0, 360, -3.0), (1, 540, -6.0), (2, 600, 0.5),
             (2, 900, -0.5), (0, 10_000, -30.0)]
    for sev, T, want in cases:
        assert abs(reward_d(sev, T) - want) < 1e-9, (sev, T, reward_d(sev, T), want)


def test_step_reward_sums_call_records():
    env = DispatchEnv()
    _, info = env.reset(seed=3)
    done, total, n_calls = False, 0.0, 0
    while not done:
        _, r, done, _, info = env.step(info["closest_road"])
        assert abs(r - sum(rec["reward"] for rec in info["calls"])) < 1e-9
        n_calls += len(info["calls"])
    assert n_calls == len(env.sim.calls), "every call credited exactly once"
