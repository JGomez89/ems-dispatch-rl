"""MaskablePPO for the semi-MDP: gamma ** dt discounting and a critic with privileged inputs.

Owner: M3 (task T13). Two pieces, both needed because Stable-Baselines3 alone cannot do them:

1. gamma ** dt. Decisions happen only when calls arrive, so the time between steps varies.
   SMDPRolloutBuffer stores info["dt"] for every step and runs GAE with a per-step discount
       g_t = gamma ** (dt_t / DT_UNIT_S)
       delta_t = r_t + g_t * V(s_{t+1}) * (1 - done) - V(s_t)
       A_t     = delta_t + g_t * lambda * (1 - done) * A_{t+1}
   so `gamma` is the discount per DT_UNIT_S seconds (default 240 s), not per decision.

2. Privileged critic. env.py appends true simulator values (true time until free, true
   lambda(t)) after obs[:N_ACTOR_OBS]. PrivilegedCriticPolicy gives the actor only
   obs[:N_ACTOR_OBS] and the critic the whole vector. Actor and critic are separate MLPs
   (2 x 64, proposal section 4). Only training uses the critic, so nothing privileged
   reaches the deployed policy.

    from ppo_smdp import SMDPMaskablePPO, make_policy_kwargs
    model = SMDPMaskablePPO("PrivilegedCriticPolicy", env, gamma=0.99,
                            policy_kwargs=make_policy_kwargs())

Rewards inside one step are combined by rewards.step_reward (OPEN DECISION D2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from torch import nn

from env import N_ACTOR_OBS

DT_UNIT_S = 240.0


class SMDPRolloutBuffer(MaskableRolloutBuffer):
    """MaskableRolloutBuffer that also stores dt per step and discounts GAE by gamma ** dt."""

    def reset(self) -> None:
        super().reset()
        self.dts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.next_dt = np.zeros(self.n_envs, dtype=np.float32)  # set by SMDPMaskablePPO before add()

    def add(self, *args, **kwargs) -> None:
        self.dts[self.pos] = self.next_dt
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values = last_values.clone().cpu().numpy().flatten()
        discounts = self.gamma ** (self.dts / DT_UNIT_S)
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            g = discounts[step]
            delta = self.rewards[step] + g * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + g * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values


class SMDPMaskablePPO(MaskablePPO):
    """MaskablePPO that feeds info["dt"] into SMDPRolloutBuffer.

    _update_info_buffer is called once per env step, after env.step and before
    rollout_buffer.add, so it is where the step's dt is handed to the buffer.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rollout_buffer_class", SMDPRolloutBuffer)
        super().__init__(*args, **kwargs)

    def _update_info_buffer(self, infos: list[dict[str, Any]], dones: np.ndarray | None = None) -> None:
        super()._update_info_buffer(infos, dones)
        if isinstance(getattr(self, "rollout_buffer", None), SMDPRolloutBuffer):
            self.rollout_buffer.next_dt[:] = [float(info["dt"]) for info in infos]


class _SplitActorCritic(nn.Module):
    """Separate actor and critic MLPs; the actor only sees the first n_actor features."""

    def __init__(self, n_obs: int, n_actor: int, pi: list[int], vf: list[int], activation_fn):
        super().__init__()
        self.n_actor = n_actor

        def mlp(n_in, sizes):
            layers = []
            for size in sizes:
                layers += [nn.Linear(n_in, size), activation_fn()]
                n_in = size
            return nn.Sequential(*layers), n_in

        self.policy_net, self.latent_dim_pi = mlp(n_actor, pi)
        self.value_net, self.latent_dim_vf = mlp(n_obs, vf)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features[..., : self.n_actor])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class PrivilegedCriticPolicy(MaskableActorCriticPolicy):
    """Actor sees obs[:n_actor_obs]; critic sees the full observation (incl. privileged)."""

    def __init__(self, *args, n_actor_obs: int = N_ACTOR_OBS, **kwargs):
        self.n_actor_obs = n_actor_obs
        kwargs.setdefault("share_features_extractor", True)  # identity Flatten, no parameters
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        arch = self.net_arch if isinstance(self.net_arch, dict) else dict(pi=self.net_arch, vf=self.net_arch)
        self.mlp_extractor = _SplitActorCritic(self.features_dim, self.n_actor_obs,
                                               arch.get("pi", []), arch.get("vf", []), self.activation_fn)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data["n_actor_obs"] = self.n_actor_obs
        return data


SMDPMaskablePPO.policy_aliases = {**MaskablePPO.policy_aliases,
                                  "PrivilegedCriticPolicy": PrivilegedCriticPolicy}


def make_policy_kwargs() -> dict:
    """Proposal section 4: separate 2 x 64 actor and critic MLPs."""
    return dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=nn.Tanh)
