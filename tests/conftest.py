import pytest

from pandawork.core.render import Cameras
from pandawork.core.robot import build_models


@pytest.fixture(scope="session")
def models():
    """(self_model, plant_model)."""
    return build_models()


@pytest.fixture(scope="session")
def cameras(models):
    """Offscreen renderer for the plant; skips rendering tests when no GL backend is usable."""
    cams = Cameras(models[1])
    try:
        cams._renderer(32, 24)
    except Exception as exc:  # GL backends fail with backend-specific exception types
        pytest.skip(f"no OpenGL backend for offscreen rendering: {exc!r}")
    yield cams
    cams.close()
