import math
from dataclasses import replace

import numpy as np
import pytest

from agvlab import frames
from agvlab.follower import FollowerConfig
from agvlab.lab import RunConfig, World, run_closed_loop
from agvlab.layout import default_layout
from agvlab.plant import ChannelConfig
from agvlab.sweep import SWEEPS, aggregate, find_break, make_config, run_sweep


@pytest.fixture(scope="module")
def nominal_branch():
    return run_closed_loop(RunConfig(route="branch"), keep_frames=True)


@pytest.fixture(scope="module")
def nominal_straight():
    return run_closed_loop(RunConfig(route="straight", seed=1))


def test_nominal_branch_run_succeeds(nominal_branch):
    m = nominal_branch.metrics
    assert m["success"] == 1.0 and m["junction_ok"] == 1.0 and m["stopped"] == 1.0
    assert m["markers_seen"] == 2


def test_nominal_straight_run_succeeds(nominal_straight):
    m = nominal_straight.metrics
    assert m["success"] == 1.0 and m["junction_ok"] == 1.0


def test_nominal_tracking_error_is_small(nominal_branch, nominal_straight):
    for r in (nominal_branch, nominal_straight):
        assert r.metrics["rms_mm"] < 3.0 and r.metrics["max_mm"] < 20.0


def test_nominal_stop_accuracy(nominal_branch, nominal_straight):
    for r in (nominal_branch, nominal_straight):
        assert abs(r.metrics["stop_error_mm"]) < 5.0
        assert abs(r.metrics["stop_lateral_mm"]) < 60.0  # marker sits 55 mm beside the tape


def test_frame_rate_is_100_hz(nominal_branch):
    assert nominal_branch.metrics["tpdo1_rate_hz"] == pytest.approx(100.0, abs=1.0)


def test_simulated_arrival_jitter_follows_the_assumed_channel(nominal_branch):
    # assumed channel: 1 ms half-normal jitter, bus order kept -> interval sd of about 1 ms
    ts = [t for t, cid, _ in nominal_branch.frames_log if cid == 0x195]
    sd_ms = np.std(np.diff(ts)) * 1000.0
    assert 0.5 < sd_ms < 1.6


def test_every_logged_frame_decodes(nominal_branch):
    for _, cid, data in nominal_branch.frames_log:
        assert frames.decode_frame(cid, data, 21) is not None


def test_initial_offset_is_recovered_quickly(nominal_branch):
    t, e = nominal_branch.t, nominal_branch.err_mm
    assert abs(e[0]) > 15.0  # starts 20 mm off
    assert np.max(np.abs(e[(t > 2.0) & (t < 8.0)])) < 3.0


def test_runs_are_deterministic_per_seed():
    cfg = RunConfig(route="straight", seed=3, max_time_s=8.0)
    a, b = run_closed_loop(cfg), run_closed_loop(cfg)
    assert np.array_equal(a.err_mm, b.err_mm)
    c = run_closed_loop(replace(cfg, seed=4))
    assert not np.array_equal(a.err_mm, c.err_mm)


def test_error_sign_matches_start_displacement():
    r = run_closed_loop(RunConfig(start_offset_mm=20.0, max_time_s=0.2))
    assert r.err_mm[0] > 15.0  # displaced left of the route = positive
    r2 = run_closed_loop(RunConfig(start_offset_mm=-20.0, max_time_s=0.2))
    assert r2.err_mm[0] < -8.0  # axle 20 mm right, but yawed 2 deg left with the sensor 0.25 m ahead


def test_mirrored_layout_also_works():
    r = run_closed_loop(RunConfig(route="branch", layout_branch_side="left"))
    assert r.metrics["success"] == 1.0


def test_pure_pursuit_follows_but_with_curve_offset(nominal_branch):
    r = run_closed_loop(RunConfig(follower=FollowerConfig(controller="pure_pursuit")))
    assert r.metrics["junction_ok"] == 1.0 and r.metrics["stopped"] == 1.0
    assert r.metrics["rms_mm"] > nominal_branch.metrics["rms_mm"]


def test_baseline_p_law_tracks_but_cannot_stop_or_slow_down():
    r = run_closed_loop(RunConfig(follower=FollowerConfig.baseline(), route="straight"))
    assert r.metrics["derailed"] == 0.0 and r.metrics["stopped"] == 0.0
    assert r.metrics["rms_mm"] > 5.0  # P-only law at 0.15 m/s leaves a steady offset in the curve


def test_survives_heavy_frame_loss():
    r = run_closed_loop(RunConfig(channel=ChannelConfig(latency_ms=2.0, jitter_ms=1.0, drop_prob=0.5)))
    assert r.metrics["success"] == 1.0
    assert r.metrics["tpdo1_rate_hz"] == pytest.approx(50.0, abs=5.0)
    assert r.metrics["frames_dropped"] > 500


def test_survives_ten_times_nominal_noise():
    base = RunConfig()
    s = replace(base.sensor, pos_sigma_mm=3.0, angle_sigma_deg=10.0, marker_sigma_mm=1.0)
    r = run_closed_loop(replace(base, sensor=s))
    assert r.metrics["success"] == 1.0 and r.metrics["rms_mm"] < 6.0


def test_extreme_latency_breaks_the_run():
    base = RunConfig()
    r = run_closed_loop(replace(base, channel=replace(base.channel, latency_ms=400.0)))
    assert r.metrics["success"] == 0.0


def test_latency_degrades_stop_accuracy_monotonically():
    errs = []
    for lat in (0.0, 100.0, 200.0):
        base = RunConfig()
        r = run_closed_loop(replace(base, channel=replace(base.channel, latency_ms=lat)))
        errs.append(abs(r.metrics["stop_error_mm"]))
    assert errs[0] < errs[1] < errs[2]


def test_world_emits_frames_and_moves():
    w = World(default_layout(), seed=0)
    n = 0
    for _ in range(100):
        n += len(w.step(0.3, 0.0, 0.01))
    assert n > 150 and w.plant.x > 0.1


def test_sweep_definitions_and_config_builder():
    assert set(SWEEPS) == {"noise", "latency", "drop", "burst", "speed"}
    cfg = make_config("latency", 150.0, seed=3)
    assert cfg.channel.latency_ms == 150.0 and cfg.route == "straight" and cfg.seed == 3
    assert make_config("noise", 10, 0).sensor.pos_sigma_mm == pytest.approx(3.0)
    assert make_config("drop", 0.2, 0).channel.drop_prob == 0.2
    assert make_config("burst", 100, 0).sensor.burst_hz > 0
    assert make_config("speed", 0.5, 0).follower.v_cruise == 0.5


def test_aggregate_and_find_break():
    ok = {"success": 1.0, "junction_ok": 1.0, "stopped": 1.0, "derailed": 0.0, "rms_mm": 2.0, "max_mm": 9.0,
          "stop_error_mm": -1.0, "tpdo1_rate_hz": 100.0}
    bad = dict(ok, success=0.0, rms_mm=9.0, stop_error_mm=math.nan)
    rows = [aggregate(1, [ok, ok]), aggregate(2, [ok, bad])]
    assert rows[0]["success_rate"] == 1.0 and rows[1]["success_rate"] == 0.5
    assert rows[1]["rms_mm_worst"] == 9.0 and rows[1]["stop_abs_mm_worst"] == 1.0
    assert find_break(rows) == {"first_failure": 2, "first_rms_over_limit": 2}  # median of 2 and 9 mm is 5.5 mm
    assert find_break([aggregate(1, [bad])])["first_rms_over_limit"] == 1


def test_run_sweep_serial_smoke():
    rows = run_sweep("noise", [1, 30], seeds=1, workers=1)
    assert [r["value"] for r in rows] == [1, 30]
    assert rows[1]["rms_mm_median"] > rows[0]["rms_mm_median"]
