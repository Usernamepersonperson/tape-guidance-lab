"""Real-time lab over python-can: sensor simulator -> bus -> follower, in wall-clock time.

The sensor side and the follower side use separate `can.Bus` objects on the same channel,
so the frames travel through the real python-can backend (`virtual` in tests, `socketcan`
on vcan0 / a real CAN adapter, `udp_multicast` across processes or containers).
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import can
import numpy as np

from .follower import Follower
from .lab import RunConfig, World, score
from .layout import default_layout, nearest_on_polyline


def to_message(fr, timestamp: Optional[float] = None) -> can.Message:
    return can.Message(arbitration_id=fr.can_id, data=fr.data, is_extended_id=False,
                       timestamp=timestamp if timestamp is not None else 0.0)


def open_pair(interface: str, channel: str) -> tuple:
    """Two independent bus handles on one channel: (sensor side, follower side)."""
    kw = {"interface": interface, "channel": channel}
    return can.Bus(**kw), can.Bus(**kw)


def vcan_available(name: str = "vcan0") -> bool:
    """True when a SocketCAN interface `name` exists and python-can can open it."""
    import sys
    if not sys.platform.startswith("linux"):
        return False
    try:
        bus = can.Bus(interface="socketcan", channel=name)
    except Exception:
        return False
    bus.shutdown()
    return True


@dataclass
class LiveResult:
    metrics: Dict[str, float]
    rx_interval_ms: Dict[str, float]
    n_tpdo1: int


def run_live(tx_bus: can.BusABC, rx_bus: can.BusABC, cfg: Optional[RunConfig] = None,
             duration_s: float = 3.0, stop_on_stopped: bool = True) -> LiveResult:
    """Run the closed loop in real time. Returns tracking metrics and the achieved frame rate."""
    cfg = cfg or RunConfig()
    layout = default_layout(cfg.layout_branch_side)
    world = World(layout, cfg.sensor, cfg.plant, cfg.seed, cfg.start_offset_mm, cfg.start_heading_deg)
    fcfg = replace(cfg.follower, side=layout.route_side(cfg.route), node_id=cfg.sensor.node_id)
    follower = Follower(fcfg)
    route_path = layout.routes[cfg.route]

    cmd = [0.0, 0.0]
    lock = threading.Lock()
    stop = threading.Event()
    err_log: List[tuple] = []
    sent = [0]

    def sensor_thread() -> None:
        t_prev = time.perf_counter()
        t0 = t_prev
        next_tick = t_prev
        while not stop.is_set():
            next_tick += min(cfg.dt, 0.005)
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            with lock:
                v, w = cmd
            for fr in world.step(v, w, dt):
                tx_bus.send(to_message(fr))
                sent[0] += 1
            d, signed, _ = nearest_on_polyline(route_path, world.sensor_point())
            err_log.append((now - t0, signed * 1000.0, world.plant.v))

    th = threading.Thread(target=sensor_thread, name="agvlab-sensor", daemon=True)
    t_start = time.perf_counter()
    th.start()
    rx_times: List[float] = []
    next_ctl = t_start
    try:
        while time.perf_counter() - t_start < duration_s:
            msg = rx_bus.recv(timeout=0.002)
            now = time.perf_counter()
            if msg is not None:
                follower.on_frame(now - t_start, msg.arbitration_id, bytes(msg.data))
                if (msg.arbitration_id & 0x780) == 0x180:
                    rx_times.append(now - t_start)
            if now >= next_ctl:
                next_ctl += 1.0 / cfg.control_hz
                v, w = follower.update(now - t_start)
                with lock:
                    cmd[0], cmd[1] = v, w
                if stop_on_stopped and follower.stopped and world.plant.v < 1e-3:
                    break
    finally:
        stop.set()
        th.join(timeout=1.0)
    dur = time.perf_counter() - t_start
    t_arr = [e[0] for e in err_log]
    metrics = score(world, follower, layout, cfg.route, t_arr, [e[1] for e in err_log],
                    [e[2] for e in err_log], len(rx_times), dur, cfg.settle_s, False)
    metrics["frames_sent"] = float(sent[0])
    iv = np.diff(np.array(rx_times)) * 1000.0 if len(rx_times) > 2 else np.array([math.nan])
    stats = {"mean": float(np.mean(iv)), "sd": float(np.std(iv)), "max": float(np.max(iv)),
             "p99": float(np.percentile(iv, 99))}
    return LiveResult(metrics, stats, len(rx_times))
