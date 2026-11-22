"""ppo_smdp.py: gamma ** dt GAE and the privileged-critic split. Runs without the simulator."""

import gymnasium as gym
import numpy as np
import torch as th

from env import N_ACTOR_OBS, OBS_DIM
from ppo_smdp import DT_UNIT_S, SMDPMaskablePPO, SMDPRolloutBuffer, make_policy_kwargs


class ToyEnv(gym.Env):
    """Random obs of the real size, 7 actions, random dt in info."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
    action_space = gym.spaces.Discrete(7)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self.observation_space.sample().astype(np.float32) * 0, {"dt": 0.0}

    def step(self, action):
        self.t += 1
        obs = self.np_random.normal(size=OBS_DIM).astype(np.float32)
        return obs, float(action == 0), self.t >= 20, False, {"dt": float(self.np_random.uniform(10, 900))}

    def action_masks(self):
        m = np.ones(7, dtype=bool)
        m[3] = False
        return m


def test_gae_uses_gamma_to_the_dt():
    buf = SMDPRolloutBuffer(3, ToyEnv.observation_space, ToyEnv.action_space, gamma=0.9, gae_lambda=0.8)
    dts, rewards, values = [240.0, 480.0, 120.0], [1.0, 2.0, 3.0], [0.5, 0.2, 0.1]
    for dt, r, v in zip(dts, rewards, values):
        buf.next_dt[:] = dt
        buf.add(np.zeros((1, OBS_DIM), np.float32), np.zeros((1, 1)), np.array([r]), np.array([False]),
                th.tensor([v]), th.tensor([0.0]), action_masks=np.ones((1, 7)))
    buf.compute_returns_and_advantage(th.tensor([0.7]), np.array([False]))

    g = [0.9 ** (dt / DT_UNIT_S) for dt in dts]
    nxt = values[1:] + [0.7]
    adv, expected = 0.0, []
    for k in reversed(range(3)):
        delta = rewards[k] + g[k] * nxt[k] - values[k]
        adv = delta + g[k] * 0.8 * adv
        expected.insert(0, adv)
    np.testing.assert_allclose(buf.advantages[:, 0], expected, rtol=1e-5)


def test_actor_ignores_privileged_inputs_and_critic_uses_them():
    model = SMDPMaskablePPO("PrivilegedCriticPolicy", ToyEnv(), n_steps=32, batch_size=16, seed=0,
                            policy_kwargs=make_policy_kwargs())
    obs = th.randn(4, OBS_DIM)
    obs2 = obs.clone()
    obs2[:, N_ACTOR_OBS:] += 5.0
    pol = model.policy
    with th.no_grad():
        l1 = pol.get_distribution(obs).distribution.logits
        l2 = pol.get_distribution(obs2).distribution.logits
        v1, v2 = pol.predict_values(obs), pol.predict_values(obs2)
    assert th.allclose(l1, l2)
    assert not th.allclose(v1, v2)


def test_learn_save_load(tmp_path):
    model = SMDPMaskablePPO("PrivilegedCriticPolicy", ToyEnv(), n_steps=32, batch_size=16, seed=0,
                            policy_kwargs=make_policy_kwargs())
    model.learn(64)
    assert np.any(model.rollout_buffer.dts > 0)
    model.save(tmp_path / "m.zip")
    loaded = SMDPMaskablePPO.load(tmp_path / "m.zip", device="cpu")
    a, _ = loaded.predict(np.zeros(OBS_DIM, np.float32), action_masks=ToyEnv().action_masks(), deterministic=True)
    assert int(a) != 3
