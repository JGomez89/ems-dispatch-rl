"""Shared test helpers."""

import pytest


def skip_if_unimplemented(fn, *args, **kwargs):
    """Call fn; skip the test (not fail) while the owner's code still raises NotImplementedError."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"{getattr(fn, '__qualname__', fn)} not implemented yet")
