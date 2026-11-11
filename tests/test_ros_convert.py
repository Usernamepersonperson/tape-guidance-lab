"""The ROS-free message conversions of the driver package (ros2/src/tape_guide_driver)."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ros2" / "src" / "tape_guide_driver"))

from tape_guide_driver import convert  # noqa: E402

from agvlab import frames  # noqa: E402


def test_track_fields_use_si_units_and_keep_signs():
    f = frames.TrackFrame.from_physical(-12.3, 8.0, -2.5, 1.0, 80.0, fork=True)
    d = convert.track_fields(f)
    assert d["offset_left_m"] == pytest.approx(-0.0123) and d["offset_right_m"] == pytest.approx(0.008)
    assert d["heading_left_rad"] == pytest.approx(math.radians(-2.5))
    assert d["quality"] == pytest.approx(0.8) and d["fork"] and d["detected"] and d["two_tracks"]


def test_no_tape_gives_zero_offsets_and_not_detected():
    d = convert.track_fields(frames.TrackFrame())
    assert d["detected"] is False and d["offset_left_m"] == 0.0 and d["quality"] == 0.0


def test_track_roundtrip_through_message_fields():
    f = frames.TrackFrame.from_physical(5.5, -3.1, 4.0, -1.5, 60.0, merge=True, marker_left=True)
    back = convert.frame_from_track(SimpleNamespace(**convert.track_fields(f)))
    assert back == f


def test_track_roundtrip_without_tape():
    f = frames.TrackFrame(marker_right=True)
    assert convert.frame_from_track(SimpleNamespace(**convert.track_fields(f))) == f


def test_markers_fields_flag_empty_slots():
    d = convert.markers_fields(frames.MarkerFrame(left_along_raw=400, left_lateral_raw=1100))
    assert d["left_valid"] and d["left_along_m"] == pytest.approx(0.02)
    assert d["left_lateral_m"] == pytest.approx(0.055)
    assert not d["right_valid"] and d["right_along_m"] == 0.0


def test_markers_roundtrip():
    f = frames.MarkerFrame(-300, 1100, frames.NO_DATA, frames.NO_DATA)
    assert convert.frame_from_markers(SimpleNamespace(**convert.markers_fields(f))) == f


def test_follower_accepts_converted_frames():
    from agvlab.follower import Follower
    fol = Follower()
    f = frames.TrackFrame.from_physical(20.0)
    fol.on_track(0.0, convert.frame_from_track(SimpleNamespace(**convert.track_fields(f))))
    v, w = fol.update(0.0)
    assert v > 0 and w > 0  # tape on the left: steer counter-clockwise
