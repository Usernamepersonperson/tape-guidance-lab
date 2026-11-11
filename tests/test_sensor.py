import math

import numpy as np
import pytest
from conftest import noise_free, pose_at, straight_layout

from agvlab import frames
from agvlab.layout import Layout, Marker, build_path
from agvlab.sensor import SensorConfig, SensorSim, cluster, crossings, quality_for


def read(sensor, pose, t=0.0):
    out = sensor.step(t, pose)
    by_id = {f.can_id: f for f in out}
    return {k: by_id.get(v) for k, v in sensor.ids.items()}


def tpdo1(sensor, pose, t=0.0):
    return frames.decode_tpdo1(read(sensor, pose, t)["tpdo1"].data)


def custom_layout(tapes, markers=()):
    return Layout(tapes=tapes, routes={"straight": tapes[0], "branch": tapes[0]},
                  markers=[Marker(x, y, "straight", "stop") for x, y in markers],
                  start_pose=(0, 0, 0), fork_s=1e9, branch_side="right")


def test_first_step_emits_all_four_frames_with_node_ids(make_sensor):
    got = read(make_sensor(), pose_at(1.0))
    assert all(v is not None for v in got.values())
    assert [got[k].can_id for k in ("tpdo1", "tpdo2", "tpdo3", "heartbeat")] == [0x195, 0x295, 0x395, 0x715]


def test_centred_tape_reads_zero_full_quality(make_sensor):
    f = tpdo1(make_sensor(), pose_at(1.0))
    assert (f.left_offset_mm, f.right_offset_mm, f.left_heading_deg) == (0.0, 0.0, 0.0)
    assert f.quality_pct == 100.0 and f.single_track and f.tape_detected


@pytest.mark.parametrize("lateral,expected", [(10, 10.0), (-10, -10.0), (40, 40.0), (-57.3, -57.3)])
def test_offset_is_left_positive(make_sensor, lateral, expected):
    # lateral = how far the sensor sits to the right of the tape; the tape then lies on its left (positive)
    assert tpdo1(make_sensor(), pose_at(1.0, lateral_mm=lateral)).left_offset_mm == pytest.approx(expected, abs=0.051)


def test_heading_is_counter_clockwise_positive(make_sensor):
    # robot yawed counter-clockwise: the tape leans clockwise relative to the sensor -> negative heading
    f = tpdo1(make_sensor(), pose_at(1.0, heading_deg=5.0))
    assert f.left_heading_deg == pytest.approx(-5.0)
    assert tpdo1(make_sensor(), pose_at(1.0, heading_deg=-5.0)).left_heading_deg == pytest.approx(5.0)


@pytest.mark.parametrize("lateral,quality", [(0, 100.0), (35, 100.0), (52.5, 60.0), (69, 22.3)])
def test_quality_ramp(make_sensor, lateral, quality):
    f = tpdo1(make_sensor(), pose_at(1.0, lateral_mm=lateral))
    assert f.quality_pct == pytest.approx(quality, abs=0.6)


def test_tape_beyond_detection_limit_reads_as_no_tape(make_sensor):
    f = tpdo1(make_sensor(), pose_at(1.0, lateral_mm=72))
    assert f == frames.TrackFrame() and not f.tape_detected


def test_quality_for_edges():
    cfg = SensorConfig()
    assert [quality_for(x, cfg) for x in (0, 35, 70, 70.1)] == [100.0, 100.0, 20.0, 0.0]
    assert quality_for(-52.5, cfg) == quality_for(52.5, cfg)


def test_no_tape_under_sensor(make_sensor):
    f = tpdo1(make_sensor(), pose_at(6.0))  # beyond the end of the tape
    assert f == frames.TrackFrame()


def test_two_tapes_report_left_and_right(make_sensor):
    lay = custom_layout([build_path((0, 0.04), 0.0, [("line", 3)]), build_path((0, -0.04), 0.0, [("line", 3)])])
    f = tpdo1(make_sensor(lay), pose_at(1.0))
    assert (f.left_offset_mm, f.right_offset_mm) == (pytest.approx(40.0), pytest.approx(-40.0))
    assert not f.single_track


def test_tapes_closer_than_merge_distance_read_as_one(make_sensor):
    lay = custom_layout([build_path((0, 0.005), 0.0, [("line", 3)]), build_path((0, -0.005), 0.0, [("line", 3)])])
    f = tpdo1(make_sensor(lay), pose_at(1.0))
    assert f.single_track and f.left_offset_mm == 0.0


def test_cluster_merges_and_sorts():
    assert cluster([(30.0, 0.0), (-30.0, 0.1), (31.0, 0.0)], 25.0) == [(-30.0, 0.1), (30.5, 0.0)]
    assert cluster([], 25.0) == []


def test_fork_flag_when_tapes_diverge(make_sensor):
    straight = build_path((0, -0.01), 0.0, [("line", 3)])
    left_branch = build_path((0, 0.01 - math.tan(math.radians(20))), math.radians(20), [("line", 3)])
    f = tpdo1(make_sensor(custom_layout([straight, left_branch])), pose_at(1.0))
    assert f.fork and not f.merge and f.left_offset_mm == pytest.approx(10.0)
    assert f.right_offset_mm == pytest.approx(-10.0)


def test_merge_flag_when_tapes_converge(make_sensor):
    straight = build_path((0, -0.01), 0.0, [("line", 3)])
    left_branch = build_path((0, 0.01 + math.tan(math.radians(20))), -math.radians(20), [("line", 3)])
    f = tpdo1(make_sensor(custom_layout([straight, left_branch])), pose_at(1.0))
    assert f.merge and not f.fork


def test_fork_flag_needs_tracks_within_one_tape_width(make_sensor):
    straight = build_path((0, -0.05), 0.0, [("line", 3)])
    left_branch = build_path((0, 0.05 - math.tan(math.radians(20))), math.radians(20), [("line", 3)])
    f = tpdo1(make_sensor(custom_layout([straight, left_branch])), pose_at(1.0))
    assert not f.fork and not f.single_track  # 100 mm apart: two tracks, but no fork flag


def test_steep_crossing_sets_junction_and_zeroes_headings(make_sensor):
    steep = build_path((1.0 - 0.5 / math.tan(math.radians(75)), -0.5), math.radians(75), [("line", 1.5)])
    f = tpdo1(make_sensor(custom_layout([steep])), pose_at(1.0))
    assert f.junction and (f.left_heading_deg, f.right_heading_deg) == (0.0, 0.0)


def test_marker_left_of_tape_sets_left_slot_and_flag(make_sensor):
    lay = straight_layout(markers=[(1.02, 0.055, "stop")])
    got = read(make_sensor(lay), pose_at(1.0))
    f, m = frames.decode_tpdo1(got["tpdo1"].data), frames.decode_tpdo2(got["tpdo2"].data)
    assert f.marker_left and not f.marker_right
    # 20 mm ahead, 55 mm to the left of the sensor centre (left positive)
    assert (m.left_along_mm, m.left_lateral_mm) == (pytest.approx(20.0), pytest.approx(55.0))
    assert m.right_along_mm is None and m.right_lateral_mm is None


def test_marker_positions_are_quantised_to_the_resolution(make_sensor):
    lay = straight_layout(markers=[(1.0123, 0.0557, "stop")])
    m = frames.decode_tpdo2(read(make_sensor(lay), pose_at(1.0))["tpdo2"].data)
    assert m.left_lateral_mm == pytest.approx(55.75) and m.left_along_mm == pytest.approx(12.25)


def test_marker_right_of_tape_sets_right_slot(make_sensor):
    lay = straight_layout(markers=[(0.97, -0.05, "stop")])
    got = read(make_sensor(lay), pose_at(1.0))
    f, m = frames.decode_tpdo1(got["tpdo1"].data), frames.decode_tpdo2(got["tpdo2"].data)
    assert f.marker_right and not f.marker_left
    assert (m.right_along_mm, m.right_lateral_mm) == (pytest.approx(-30.0), pytest.approx(-50.0))


def test_marker_outside_longitudinal_window_not_reported(make_sensor):
    lay = straight_layout(markers=[(1.0 + 0.06, 0.055, "stop")])
    got = read(make_sensor(lay), pose_at(1.0))
    assert not frames.decode_tpdo1(got["tpdo1"].data).marker_left
    assert frames.decode_tpdo2(got["tpdo2"].data) == frames.MarkerFrame()


def test_lone_marker_reported_on_both_sides(make_sensor):
    lay = straight_layout(markers=[(6.0, 0.03, "stop")])
    got = read(make_sensor(lay), pose_at(6.0))
    f, m = frames.decode_tpdo1(got["tpdo1"].data), frames.decode_tpdo2(got["tpdo2"].data)
    assert not f.tape_detected
    assert m.left_lateral_mm == m.right_lateral_mm == pytest.approx(30.0) and m.left_along_mm == 0.0


def test_frame_periods_100_50_20_1_hz(make_sensor):
    s = make_sensor()
    times = {k: [] for k in s.ids.values()}
    for i in range(0, 201):
        t = i * 0.005
        for fr in s.step(t, pose_at(1.0)):
            times[fr.can_id].append(fr.t)
    for key, spacing in (("tpdo1", 0.010), ("tpdo2", 0.020), ("tpdo3", 0.050), ("heartbeat", 1.0)):
        ts = times[s.ids[key]]
        assert np.allclose(np.diff(ts), spacing, atol=1e-9)
    assert 100 <= len(times[s.ids["tpdo1"]]) <= 101


def test_all_emitted_payloads_have_the_profile_lengths(make_sensor):
    lens = {0x195: 8, 0x295: 8, 0x395: 3, 0x715: 1}
    s = make_sensor()
    for i in range(100):
        for fr in s.step(i * 0.01, pose_at(1.0)):
            assert len(fr.data) == lens[fr.can_id]


def test_heartbeat_reports_operational(make_sensor):
    hb = read(make_sensor(), pose_at(1.0))["heartbeat"]
    assert frames.decode_frame(hb.can_id, hb.data) == frames.HeartbeatFrame(frames.NMT_OPERATIONAL)


def test_sample_counter_in_tpdo3_counts_tpdo1_frames(make_sensor):
    s = make_sensor()
    last = None
    for i in range(30):
        for fr in s.step(i * 0.01, pose_at(1.0)):
            if fr.can_id == 0x395:
                last = frames.decode_tpdo3(fr.data)
    # TPDO3 goes out at t = 0.00, 0.05, ...; at t = 0.25 the TPDO1 samples at 0.00..0.25 (26) have been sent
    assert last.sample_counter == 26 and last.error_register == 0


def test_invert_position_flips_sign(make_sensor):
    cfg = noise_free(invert_position=True)
    assert tpdo1(make_sensor(cfg=cfg), pose_at(1.0, lateral_mm=10)).left_offset_mm == pytest.approx(-10.0)


def test_noise_has_configured_spread_and_is_seeded():
    cfg = SensorConfig(pos_sigma_mm=3.0)
    a = SensorSim(straight_layout(), cfg, np.random.default_rng(5))
    b = SensorSim(straight_layout(), cfg, np.random.default_rng(5))
    pos_a = [tpdo1(a, pose_at(1.0), i * 0.01).left_offset_mm for i in range(1500)]
    pos_b = [tpdo1(b, pose_at(1.0), i * 0.01).left_offset_mm for i in range(1500)]
    assert pos_a == pos_b
    assert 2.6 < np.std(pos_a) < 3.5 and abs(np.mean(pos_a)) < 0.3


def test_nominal_noise_is_close_to_configured_sigma():
    s = SensorSim(straight_layout(), SensorConfig(), np.random.default_rng(1))
    pos = [tpdo1(s, pose_at(1.0), i * 0.01).left_offset_mm for i in range(1500)]
    assert 0.25 < np.std(pos) < 0.36  # 0.3 mm rms, 0.1 mm wire LSB


def test_bursts_blank_the_sensor():
    cfg = noise_free(burst_hz=1000.0, burst_ms=500.0)
    s = SensorSim(straight_layout(), cfg, np.random.default_rng(0))
    f = [tpdo1(s, pose_at(1.0), 0.01 + i * 0.01) for i in range(20)]
    assert s.bursts >= 1 and all(not x.tape_detected for x in f[2:])


def test_crossings_ignores_tape_outside_sensor_width():
    tape = build_path((0, 0.2), 0.0, [("line", 3)])
    assert crossings(pose_at(1.0), [tape], 0.25, 75.0) == []


def test_tape_beyond_detection_limit_is_ignored_and_the_other_track_keeps_its_quality(make_sensor):
    # left tape 72 mm away (inside the 150 mm width but too faint to see), right tape centred
    lay = custom_layout([build_path((0, 0.072), 0.0, [("line", 3)]), build_path((0, 0.0), 0.0, [("line", 3)])])
    f = tpdo1(make_sensor(lay), pose_at(1.0))
    assert f.single_track and f.left_offset_mm == 0.0 and f.quality_pct == 100.0
