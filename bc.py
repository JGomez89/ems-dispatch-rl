"""Behavior cloning for E3. Owner: M4 (tasks T20-T22).

Split out of train.py so M3 (PPO) and M4 (BC) never edit the same file. train.py calls these.

E3 arms (train.py --arm):
    scratch      PPO from random init                             (no BC)
    bc_only      collect -> pretrain_actor; saved without PPO     (no PPO)
    bc_ppo       collect -> pretrain_actor -> warmup_critic -> PPO
    bc_ppo_aux   as bc_ppo, plus aux_bc_coef(t) * BC loss during PPO, decaying to 0

Teacher: baselines.closest_travel_time (OPEN DECISION D4 in CONTRACT.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_BC_DECISIONS = 200_000


@dataclass
class BCDataset:
    obs: np.ndarray       # (N, OBS_DIM) float32
    masks: np.ndarray     # (N, 7) bool
    actions: np.ndarray   # (N,) int, the teacher's choice
    returns: np.ndarray   # (N,) float, discounted return-to-go under the teacher (critic warm-up)


def collect_dataset(env_fn, teacher="closest_travel_time", n=N_BC_DECISIONS, seed=0) -> BCDataset:
    """Run the teacher on TRAIN_MAPS / all loads and record (obs, mask, action, return-to-go).

    Discount returns-to-go with the same gamma ** (dt / DT_UNIT_S) as ppo_smdp.
    Save to data/bc_<teacher>_<n>.npz so every E3 run uses the same dataset.
    """
    raise NotImplementedError


def pretrain_actor(model, data: BCDataset, epochs: int = 10, batch_size: int = 256) -> None:
    """Cross-entropy on the masked actor logits (model.policy.get_distribution(obs, masks))."""
    raise NotImplementedError


def warmup_critic(model, data: BCDataset, epochs: int = 5, batch_size: int = 256) -> None:
    """MSE of model.policy.predict_values(obs) against data.returns, actor frozen."""
    raise NotImplementedError


def aux_bc_coef(progress_remaining: float, start: float = 1.0, decay_until: float = 0.5) -> float:
    """Auxiliary BC loss weight. Linear decay to 0 by the time `decay_until` of training is
    left, so it cannot cap the agent at the teacher (proposal E3). progress_remaining goes 1 -> 0."""
    done = 1.0 - progress_remaining
    return max(0.0, start * (1.0 - done / (1.0 - decay_until)))
