import math

import numpy as np
import pytest

from agvlab.plant import Channel, ChannelConfig, DiffDrive, PlantConfig
from agvlab.sensor import CanFrame


def fr(t, i=0):
    return CanFrame(t, 0x195, bytes([i]))


def test_channel_adds_fixed_latency():
    ch = Channel(ChannelConfig(latency_ms=50.0))
    ch.send(fr(1.0))
    assert ch.deliver(1.049) == []
    got = ch.deliver(1.051)
    assert len(got) == 1 and got[0].t == pytest.approx(1.05)


def test_channel_keeps_bus_order_with_jitter():
    ch = Channel(ChannelConfig(latency_ms=1.0, jitter_ms=20.0), np.random.default_rng(3))
    for i in range(200):
        ch.send(fr(i * 0.01, i % 256))
    out = ch.deliver(10.0)
    assert [f.data[0] for f in out] == [i % 256 for i in range(200)]
    assert all(b.t >= a.t for a, b in zip(out, out[1:], strict=False))


def test_channel_drop_rate():
    ch = Channel(ChannelConfig(drop_prob=0.3), np.random.default_rng(1))
    for i in range(5000):
        ch.send(fr(i * 0.01))
    assert 0.27 < ch.dropped / ch.sent < 0.33
    assert len(ch.deliver(1e6)) == ch.sent - ch.dropped


def test_channel_ideal_is_transparent():
    ch = Channel(ChannelConfig())
    ch.send(fr(0.5))
    assert ch.deliver(0.5)[0].t == 0.5
    assert ch.deliver(0.5) == []  # delivered only once


def test_plant_straight_line_speed_converges():
    p = DiffDrive()
    for _ in range(300):
        p.step(0.3, 0.0, 0.01)
    assert p.v == pytest.approx(0.3, abs=1e-3) and p.y == pytest.approx(0.0, abs=1e-9)
    assert p.x == pytest.approx(0.3 * 3.0 - 0.3 * 0.1, abs=0.02)  # distance minus the lag


def test_plant_turns_left_with_positive_omega():
    p = DiffDrive()
    for _ in range(200):
        p.step(0.2, 0.5, 0.01)
    assert p.th > 0.5 and p.y > 0


def test_plant_first_order_lag():
    p = DiffDrive(cfg=PlantConfig(tau_v_s=0.1, a_max=100.0))
    for _ in range(10):  # 0.1 s = one time constant
        p.step(1.0, 0.0, 0.01)
    assert p.v == pytest.approx(1 - math.exp(-1), abs=0.02)


def test_plant_limits_acceleration_and_speed():
    p = DiffDrive(cfg=PlantConfig(a_max=1.0, v_max=0.5))
    p.step(10.0, 0.0, 0.1)
    assert p.v <= 0.1 + 1e-9
    for _ in range(500):
        p.step(10.0, 0.0, 0.01)
    assert p.v <= 0.5 + 1e-9
