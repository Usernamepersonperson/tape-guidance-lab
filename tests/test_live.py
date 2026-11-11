import importlib.util
import sys
import time
import uuid

import can
import pytest

from agvlab import frames
from agvlab.lab import RunConfig
from agvlab.live import open_pair, run_live, to_message, vcan_available
from agvlab.sensor import CanFrame


def _timer_ok() -> bool:
    """True when sleep(5 ms) really takes about 5 ms (a busy or coarse-timer host cannot hold 100 Hz)."""
    t = time.perf_counter()
    for _ in range(20):
        time.sleep(0.005)
    return (time.perf_counter() - t) / 20 < 0.008


needs_timer = pytest.mark.skipif(not _timer_ok(), reason="host timer too coarse or busy for real-time rates")


def test_to_message_is_standard_11_bit():
    m = to_message(CanFrame(0.0, 0x195, bytes.fromhex("500085ff02fb07c8")))
    assert m.arbitration_id == 0x195 and not m.is_extended_id and bytes(m.data).hex() == "500085ff02fb07c8"


def test_frames_cross_a_python_can_virtual_bus():
    ch = "t-" + uuid.uuid4().hex[:6]
    tx, rx = open_pair("virtual", ch)
    try:
        tx.send(to_message(CanFrame(0.0, 0x195, frames.encode_tpdo1(frames.TrackFrame.from_physical(3.0)))))
        msg = rx.recv(timeout=1.0)
        assert msg is not None and frames.decode_frame(msg.arbitration_id, bytes(msg.data), 21).left_offset_mm == 3.0
    finally:
        tx.shutdown()
        rx.shutdown()


@pytest.mark.slow
@needs_timer
def test_realtime_loop_over_virtual_bus_holds_100_hz():
    tx, rx = open_pair("virtual", "live-" + uuid.uuid4().hex[:6])
    try:
        res = run_live(tx, rx, RunConfig(), duration_s=3.0, stop_on_stopped=False)
    finally:
        tx.shutdown()
        rx.shutdown()
    assert 90.0 <= res.metrics["tpdo1_rate_hz"] <= 110.0
    assert res.n_tpdo1 > 270
    assert res.metrics["lost_events"] == 0 and res.metrics["derailed"] == 0
    assert res.metrics["max_mm"] < 25.0  # starts 20 mm off, then follows the straight tape
    assert res.rx_interval_ms["mean"] == pytest.approx(10.0, abs=1.5)


@pytest.mark.slow
@needs_timer
@pytest.mark.skipif(not vcan_available("vcan0"), reason="no SocketCAN vcan0 on this host")
def test_realtime_loop_over_socketcan_vcan0():
    tx, rx = open_pair("socketcan", "vcan0")
    try:
        res = run_live(tx, rx, RunConfig(), duration_s=3.0, stop_on_stopped=False)
    finally:
        tx.shutdown()
        rx.shutdown()
    assert 90.0 <= res.metrics["tpdo1_rate_hz"] <= 110.0


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("msgpack") is None or sys.platform == "win32",
                    reason="udp_multicast needs msgpack and a multicast-capable host")
def test_realtime_loop_over_udp_multicast():
    group = "239.74.163.%d" % (20 + uuid.uuid4().int % 200)
    try:
        tx, rx = open_pair("udp_multicast", group)
    except can.CanError as exc:  # pragma: no cover - sandbox without multicast
        pytest.skip(f"udp_multicast unavailable: {exc}")
    try:
        res = run_live(tx, rx, RunConfig(), duration_s=2.5, stop_on_stopped=False)
    finally:
        tx.shutdown()
        rx.shutdown()
    assert res.n_tpdo1 > 150
