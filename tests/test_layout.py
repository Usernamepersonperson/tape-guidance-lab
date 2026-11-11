import math

import numpy as np
import pytest

from agvlab.layout import build_path, cumulative_length, default_layout, nearest_on_polyline, point_at


def test_line_length():
    p = build_path((0, 0), 0.0, [("line", 2.5)])
    assert cumulative_length(p)[-1] == pytest.approx(2.5, abs=1e-6)


def test_arc_length_and_heading():
    p = build_path((0, 0), 0.0, [("arc", 1.2, 90.0)])
    assert cumulative_length(p)[-1] == pytest.approx(1.2 * math.pi / 2, rel=1e-3)
    assert p[-1] == pytest.approx([1.2, 1.2], abs=2e-3)  # left turn ends north-east of the start
    _, th = point_at(p, cumulative_length(p)[-1])
    assert th == pytest.approx(math.pi / 2, abs=0.02)


def test_negative_arc_turns_right():
    p = build_path((0, 0), 0.0, [("arc", 1.0, -90.0)])
    assert p[-1][1] < -0.9


def test_unknown_segment_rejected():
    with pytest.raises(ValueError):
        build_path((0, 0), 0.0, [("spiral", 1.0)])


def test_nearest_signed_left_positive():
    p = build_path((0, 0), 0.0, [("line", 2.0)])
    d, signed, s = nearest_on_polyline(p, np.array([1.0, 0.03]))
    assert d == pytest.approx(0.03) and signed == pytest.approx(0.03) and s == pytest.approx(1.0, abs=0.01)
    assert nearest_on_polyline(p, np.array([1.0, -0.02]))[1] == pytest.approx(-0.02)


def test_default_layout_routes_share_the_spine():
    lay = default_layout()
    assert set(lay.routes) == {"straight", "branch"}
    n = int(lay.fork_s / 0.005) - 5
    assert np.allclose(lay.routes["straight"][:n], lay.routes["branch"][:n])
    assert not np.allclose(lay.routes["straight"][-1], lay.routes["branch"][-1])


def test_markers_sit_55_mm_left_of_their_tape():
    lay = default_layout()
    for m in lay.markers:
        d, signed, _ = nearest_on_polyline(lay.routes[m.route], np.array([m.x, m.y]))
        assert d == pytest.approx(0.055, abs=2e-3) and signed > 0  # left of travel direction


def test_marker_roles_and_lookup():
    lay = default_layout()
    assert sorted(m.role for m in lay.markers if m.route == "branch") == ["slow", "stop"]
    assert lay.marker_for("branch").role == "stop"


def test_branch_side_mirror():
    right, left = default_layout("right"), default_layout("left")
    assert right.routes["branch"][-1][0] > right.routes["straight"][-1][0]
    assert left.routes["branch"][-1][0] < left.routes["straight"][-1][0]
    assert right.route_side("branch") == "right" and right.route_side("straight") == "left"
    assert left.route_side("branch") == "left" and left.route_side("straight") == "right"


def test_branch_separates_beyond_sensor_width_before_the_markers():
    lay = default_layout()
    m = lay.marker_for("branch")
    d_other = nearest_on_polyline(lay.post_fork["straight"], np.array([m.x, m.y]))[0]
    assert d_other > 0.16  # the other tape is well outside the 160 mm sensing width at the stop marker


def test_bad_branch_side():
    with pytest.raises(ValueError):
        default_layout("up")
