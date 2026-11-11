import math

import numpy as np
import pytest

from agvlab.layout import Layout, Marker, build_path
from agvlab.sensor import SensorConfig, SensorSim


def straight_layout(length=5.0, markers=()):
    """One tape along +x from the origin; markers are (x, y, role)."""
    tape = build_path((0.0, 0.0), 0.0, [("line", length)])
    return Layout(
        tapes=[tape], routes={"straight": tape, "branch": tape},
        markers=[Marker(x, y, "straight", role) for x, y, role in markers],
        start_pose=(0.0, 0.0, 0.0), fork_s=1e9, branch_side="right",
        post_fork={"straight": tape, "branch": tape})


def pose_at(x_sensor, lateral_mm=0.0, heading_deg=0.0, forward=0.25):
    """Robot pose such that the sensor centre sits at x_sensor, `lateral_mm` right of the tape (y < 0)."""
    th = math.radians(heading_deg)
    sx, sy = x_sensor, -lateral_mm / 1000.0
    return (sx - forward * math.cos(th), sy - forward * math.sin(th), th)


def noise_free(**kw):
    return SensorConfig(pos_sigma_mm=0.0, angle_sigma_deg=0.0, marker_sigma_mm=0.0, **kw)


@pytest.fixture
def make_sensor():
    def _make(layout=None, cfg=None, seed=0):
        return SensorSim(layout or straight_layout(), cfg or noise_free(), np.random.default_rng(seed))
    return _make
