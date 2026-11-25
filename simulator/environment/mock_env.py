"""Random-dynamics env with the real contract, so the agent pipeline can be built
before DispatchEnv is ready. Shape-correct, not learnable."""
import gymnasium as gym
import numpy as np

from simulator import simulator_config as cfg
from simulator.agent_interface import contract as C

CRITICAL = 0


class MockDispatchEnv(gym.Env):
    observation_space = C.OBSERVATION_SPACE
    action_space = C.ACTION_SPACE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.load = C.LOADS[self.np_random.integers(len(C.LOADS))]
        self.steps_left = max(1, self.np_random.poisson(cfg.LAMBDA_BAR[self.load] * cfg.EPISODE_HOURS))
        self._new_decision()
        return self._obs, self._info(0.0, [])

    def step(self, action):
        assert self._mask[action], f"illegal action {action}"
        self.steps_left -= 1
        reward = float(self.np_random.uniform(-30.0, 3.0))
        dt = float(self.np_random.exponential(3600.0 / cfg.LAMBDA_BAR[self.load]))
        record = dict(call_id=int(self.steps_left), t_call=0.0, severity=self._sev, zone=0,
                      unit=int(action), closest_straight=0, closest_road=0,
                      T=300.0, held=action == cfg.HOLD_ACTION, reward=reward)
        self._new_decision()
        return self._obs, reward, self.steps_left <= 0, False, self._info(dt, [record])

    def action_masks(self):
        return self._mask.copy()

    def _new_decision(self):
        r = self.np_random
        pol = r.uniform(C.POLICY_LOW, C.POLICY_HIGH).astype(np.float32)
        for u in range(cfg.N_UNITS):   # one-hot blocks must be valid one-hots
            s = slice(u * C.UNIT_DIM, u * C.UNIT_DIM + C.N_STATES)
            pol[s] = np.eye(C.N_STATES)[r.integers(C.N_STATES)]
        self._sev = int(r.integers(C.N_SEV))
        pol[C.OBS_SLICES["severity"]] = np.eye(C.N_SEV)[self._sev]
        crit = r.uniform(C.CRITIC_LOW, C.CRITIC_HIGH).astype(np.float32)
        crit[C.CRITIC_SLICES["load"]] = np.eye(len(C.LOADS))[C.LOADS.index(self.load)]
        self._obs = {"policy": pol, "critic": crit}

        mask = r.random(cfg.N_ACTIONS) < 0.5
        if not mask[:cfg.N_UNITS].any():
            mask[r.integers(cfg.N_UNITS)] = True
        if self._sev == CRITICAL:
            mask[cfg.HOLD_ACTION] = False
        self._mask = mask
        legal = np.flatnonzero(mask[:cfg.N_UNITS])
        self._closest = int(r.choice(legal))

    def _info(self, dt, calls):
        return {"dt": dt, "closest_straight": self._closest, "closest_road": self._closest,
                "calls": calls}
