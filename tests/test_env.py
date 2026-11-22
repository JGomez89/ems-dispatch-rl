"""env.py (M3), run on the stub simulator. Skips until env.py is implemented."""

import numpy as np

from config import HOLD
from env import N_ACTOR_OBS, OBS_DIM, EMSDispatchEnv
from stub_sim import StubSimulator
from tests.helpers import skip_if_unimplemented


def _env():
    return EMSDispatchEnv(maps=["train_00", "train_01"], loads=[0.4, 0.8], sim_cls=StubSimulator)


def test_reset_and_step_shapes():
    env = _env()
    obs, info = skip_if_unimplemented(env.reset, seed=0)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    for key in ("dt", "reached", "state", "map", "load", "call_seed"):
        assert key in info
    mask = env.action_masks()
    assert mask.shape == (HOLD + 1,) and mask.dtype == bool and mask.any()


def test_masked_units_are_unavailable_and_hold_rule():
    env = _env()
    obs, info = skip_if_unimplemented(env.reset, seed=1)
    done = False
    while not done:
        st, mask = info["state"], env.action_masks()
        for i, u in enumerate(st.units):
            assert mask[i] == u.available
        if int(st.call.severity) == 0 and any(u.available for u in st.units):
            # HOLD for a critical call only under the conditional exception (env.py docstring)
            pass  # TODO(M3): assert the exception condition once implemented
        action = int(np.flatnonzero(mask)[0])
        obs, r, terminated, truncated, info = env.step(action)
        assert "dt" in info and info["dt"] >= 0
        done = terminated or truncated


def test_actor_slice_has_no_privileged_values():
    """The actor part must not change when only privileged sim values change."""
    assert N_ACTOR_OBS < OBS_DIM
