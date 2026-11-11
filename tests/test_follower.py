import math

import pytest

from agvlab import frames
from agvlab.follower import Follower, FollowerConfig


def track(pos_l, pos_r=None, ang=0, tape=True, **flags):
    """TPDO1 payload; offsets in mm, left positive. tape=False: the sensor sees nothing."""
    if not tape:
        return frames.encode_tpdo1(frames.TrackFrame(**flags))
    return frames.encode_tpdo1(frames.TrackFrame.from_physical(pos_l, pos_r, ang, ang, 100.0, **flags))


def feed(fol, t, data, can_id=0x195):
    fol.on_frame(t, can_id, data)


def test_steers_toward_tape_on_the_right():
    f = Follower()
    feed(f, 0.0, track(-20))
    v, w = f.update(0.0)
    assert v == pytest.approx(0.30) and w < 0


def test_steers_toward_tape_on_the_left():
    f = Follower()
    feed(f, 0.0, track(20))
    assert f.update(0.0)[1] > 0


def test_centred_tape_goes_straight():
    f = Follower()
    feed(f, 0.0, track(0))
    v, w = f.update(0.0)
    assert v > 0 and w == pytest.approx(0.0, abs=1e-9)


def test_heading_term_steers_counter_clockwise_for_positive_heading():
    f = Follower()
    feed(f, 0.0, track(0, ang=5))
    assert f.update(0.0)[1] > 0


def test_omega_is_limited():
    f = Follower(FollowerConfig(w_max=0.7))
    feed(f, 0.0, track(-70))
    assert f.update(0.0)[1] == pytest.approx(-0.7)


def test_baseline_law_numbers():
    f = Follower(FollowerConfig.baseline())
    feed(f, 0.0, track(10, ang=4))
    v, w = f.update(0.0)
    assert v == pytest.approx(0.15)
    assert w == pytest.approx(5.0 * 0.010 + 1.0 * math.radians(4))


def test_pure_pursuit_curvature():
    f = Follower(FollowerConfig(controller="pure_pursuit", lookahead_m=0.25))
    feed(f, 0.0, track(10))
    v, w = f.update(0.0)
    yt = 0.010
    assert w == pytest.approx(v * 2 * yt / (0.25 ** 2 + yt ** 2))


def test_unknown_controller_rejected():
    f = Follower(FollowerConfig(controller="magic"))
    feed(f, 0.0, track(0))
    with pytest.raises(ValueError):
        f.update(0.0)


def test_side_selects_the_branch_at_a_fork():
    right = Follower(FollowerConfig(side="right"))
    left = Follower(FollowerConfig(side="left"))
    data = track(40, -40)
    for fol in (right, left):
        feed(fol, 0.0, data)
    assert right.update(0.0)[1] < 0  # tape on the right: steer right
    assert left.update(0.0)[1] > 0


def test_invert_position_mirrors_the_steering():
    f = Follower(FollowerConfig(invert_position=True))
    feed(f, 0.0, track(-20))
    assert f.update(0.0)[1] > 0


def test_stale_frame_holds_then_stops():
    f = Follower(FollowerConfig(stale_s=0.2, hold_s=0.15))
    feed(f, 0.0, track(10))
    v0, w0 = f.update(0.0)
    assert v0 > 0 and f.update(0.1)[0] == pytest.approx(v0)  # still fresh
    v1, w1 = f.update(0.25)  # stale: coast on the last command at half speed
    assert v1 == pytest.approx(0.5 * 0.30) and w1 == pytest.approx(w0)
    assert f.update(0.6) == (0.0, 0.0)
    assert f.lost_events == 1


def test_no_tape_frame_means_stop_after_hold():
    f = Follower()
    feed(f, 0.0, track(0))
    f.update(0.0)
    feed(f, 0.01, track(0, tape=False))
    f.update(0.02)
    assert f.update(0.5) == (0.0, 0.0)


def test_recovers_when_tape_returns():
    f = Follower()
    feed(f, 0.0, track(0))
    f.update(0.0)
    f.update(1.0)
    feed(f, 1.1, track(5))
    assert f.update(1.1)[0] > 0
    assert f.lost_events == 1


def test_no_data_yet_is_not_counted_as_lost():
    f = Follower()
    assert f.update(0.0) == (0.0, 0.0) and f.lost_events == 0


def test_bad_length_and_foreign_frames():
    f = Follower()
    feed(f, 0.0, b"\x00\x01")
    feed(f, 0.0, track(0), can_id=0x196)  # other node
    assert f.decode_errors == 1 and f.frames_rx == 0


def test_derivative_term_reacts_to_motion():
    f = Follower()
    for i in range(10):
        feed(f, i * 0.01, track(i))  # tape moving to the left at 0.1 m/s
    assert f._dpos > 0.05


def marker_pair(t, flag, y_raw=600):
    return track(0, marker_left=flag), frames.encode_tpdo2(frames.MarkerFrame(y_raw, 1100)), t


def pass_marker(fol, t0, n=10):
    for i in range(n):
        t1, t2, t = marker_pair(t0 + i * 0.02, True)
        feed(fol, t, t1)
        feed(fol, t, t2, 0x295)
    feed(fol, t0 + n * 0.02, track(0))
    return t0 + n * 0.02


def drive(fol, t0, seconds):
    """Follow a centred tape for a while so the follower's odometry advances."""
    t = t0
    while t < t0 + seconds:
        feed(fol, t, track(0))
        fol.update(t)
        t += 0.02
    return t


def test_marker_edges_are_counted_with_debounce():
    f = Follower()
    t = pass_marker(f, 1.0)
    assert f.markers_seen == 1
    feed(f, t + 0.1, track(0, marker_left=True))  # re-flag 0.1 s later: same marker, not a new one
    assert f.markers_seen == 1
    feed(f, t + 0.12, track(0))
    t = drive(f, t + 0.2, 4.0)  # 0.4 m of creep to the next marker
    pass_marker(f, t)
    assert f.markers_seen == 2


def test_marker_reappearing_after_a_sensor_dropout_is_not_a_new_marker():
    f = Follower()
    t = pass_marker(f, 1.0)
    f.update(t)
    # the sensor loses the marker for 0.4 s (longer than the 0.25 s gap) but the robot barely moved
    feed(f, t + 0.4, track(0, marker_left=True))
    assert f.markers_seen == 1


def test_first_marker_switches_to_creep_speed():
    f = Follower()
    t = pass_marker(f, 1.0)
    feed(f, t + 0.01, track(0))
    assert f.update(t + 0.02)[0] == pytest.approx(0.10)


def test_stops_on_second_marker_with_small_error():
    f = Follower()
    t = pass_marker(f, 1.0)
    t = drive(f, t + 0.1, 4.0)
    # second marker: offset 30 mm ahead of the sensor centre, then dead-reckoning only
    feed(f, t, track(0, marker_left=True))
    feed(f, t, frames.encode_tpdo2(frames.MarkerFrame(600, 1100)), 0x295)
    feed(f, t + 0.01, track(0))
    travelled = 0.0
    for k in range(400):
        tk = t + 0.02 + k * 0.02
        feed(f, tk, track(0))
        v, _ = f.update(tk)
        travelled += v * 0.02
        if f.stopped:
            break
    assert f.stopped and f.markers_seen == 2
    assert 0.0 <= f.stop_marker_y_mm < 10.0  # commanded stop leads by the plant lag (~v*0.1 s)
    assert 0.0 < travelled <= 0.03


def test_stop_disabled_keeps_cruising_over_markers():
    f = Follower(FollowerConfig(stop_enabled=False))
    t = pass_marker(f, 1.0)
    feed(f, t + 0.01, track(0))
    assert f.update(t + 0.02)[0] == pytest.approx(0.30)


def test_stale_marker_offset_is_forgotten_on_new_marker():
    f = Follower()
    t = pass_marker(f, 1.0)
    assert f._marker_y_m is not None
    t = drive(f, t + 0.1, 4.0)
    feed(f, t, track(0, marker_left=True))  # new marker edge before any TPDO2 for it
    assert f._marker_y_m is None
