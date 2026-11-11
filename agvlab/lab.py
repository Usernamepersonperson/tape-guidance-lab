"""Closed-loop lab: World (plant + sensor) and the simulated-time runner that scores a run."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from .follower import Follower, FollowerConfig
from .layout import Layout, cumulative_length, default_layout, nearest_on_polyline
from .plant import Channel, ChannelConfig, DiffDrive, PlantConfig
from .sensor import CanFrame, SensorConfig, SensorSim, _frame_vectors


class World:
    """Plant + sensor + layout. `step` advances physics and returns the frames the sensor emitted."""

    def __init__(self, layout: Layout, sensor_cfg: Optional[SensorConfig] = None,
                 plant_cfg: Optional[PlantConfig] = None, seed: int = 0,
                 start_offset_mm: float = 20.0, start_heading_deg: float = 2.0):
        self.layout = layout
        x0, y0, th0 = layout.start_pose
        # positive offset = robot displaced to the left of the tape
        y0 += start_offset_mm / 1000.0 * math.cos(th0)
        x0 += -start_offset_mm / 1000.0 * math.sin(th0)
        self.plant = DiffDrive((x0, y0, th0 + math.radians(start_heading_deg)), plant_cfg)
        self.sensor = SensorSim(layout, sensor_cfg, np.random.default_rng(seed))
        self.t = 0.0

    def step(self, v_cmd: float, w_cmd: float, dt: float) -> List[CanFrame]:
        self.plant.step(v_cmd, w_cmd, dt)
        self.t += dt
        return self.sensor.step(self.t, self.plant.pose)

    def sensor_point(self) -> np.ndarray:
        c, _, _ = _frame_vectors(self.plant.pose, self.sensor.cfg.mount_forward_m)
        return c

    def marker_in_sensor_frame(self, marker) -> tuple:
        """(lateral u mm right-positive, longitudinal y mm forward-positive) of a marker."""
        c, f, r = _frame_vectors(self.plant.pose, self.sensor.cfg.mount_forward_m)
        v = np.array([marker.x, marker.y]) - c
        return float(v @ r) * 1000.0, float(v @ f) * 1000.0


@dataclass
class RunConfig:
    route: str = "branch"  # "branch" or "straight" (junction choice)
    seed: int = 0
    dt: float = 0.01
    control_hz: float = 50.0
    max_time_s: Optional[float] = None  # None: route length / speed + 20 s
    start_offset_mm: float = 20.0
    start_heading_deg: float = 2.0
    settle_s: float = 1.0  # excluded from RMS (initial offset transient)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    # nominal channel: ASSUMED 2 ms latency and 1 ms half-normal jitter (a USB-CAN adapter plus host
    # scheduling); not measured on hardware, see README
    channel: ChannelConfig = field(default_factory=lambda: ChannelConfig(latency_ms=2.0, jitter_ms=1.0))
    plant: PlantConfig = field(default_factory=PlantConfig)
    follower: FollowerConfig = field(default_factory=FollowerConfig)
    layout_branch_side: str = "right"


@dataclass
class RunResult:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    th: np.ndarray
    v: np.ndarray
    err_mm: np.ndarray  # signed cross-track error of the sensor centre vs the intended route
    layout: Layout
    route: str
    metrics: Dict[str, float]
    cfg: RunConfig
    frames_log: List[tuple] = field(default_factory=list)  # (t_rx, id, data) as delivered


def score(world: World, follower: Follower, layout: Layout, route: str, t_arr, err_arr, v_arr,
          rx_tpdo1: int, duration: float, settle_s: float, derailed: bool) -> Dict[str, float]:
    t = np.asarray(t_arr)
    err = np.asarray(err_arr)
    v = np.asarray(v_arr)
    mask = (t >= settle_s) & (v > 0.02)
    if mask.sum() < 5:
        mask = t >= settle_s
    rms = float(np.sqrt(np.mean(err[mask] ** 2))) if mask.any() else float("nan")
    mx = float(np.max(np.abs(err[mask]))) if mask.any() else float("nan")

    # junction choice: final sensor point must sit on the intended post-fork tape, not the other one
    p = world.sensor_point()
    dists = {name: nearest_on_polyline(tape, p)[0] for name, tape in layout.post_fork.items()}
    other = "straight" if route == "branch" else "branch"
    _, _, s_final = nearest_on_polyline(layout.routes[route], p)
    past_fork = s_final >= layout.fork_s + 1.0
    junction_ok = bool(past_fork and dists[route] < 0.10 and dists[route] < dists[other])

    stop_err = float("nan")
    lat_err = float("nan")
    if follower.stopped and world.plant.v < 0.01:
        u, y = world.marker_in_sensor_frame(layout.marker_for(route))
        stop_err, lat_err = y, u
    return {
        "rms_mm": rms,
        "max_mm": mx,
        "junction_ok": float(junction_ok),
        "stop_error_mm": stop_err,  # +: marker still ahead of the sensor centre (short of target)
        "stop_lateral_mm": lat_err,
        "stopped": float(follower.stopped),
        "derailed": float(derailed),
        "tpdo1_rate_hz": rx_tpdo1 / duration if duration > 0 else 0.0,
        "lost_events": float(follower.lost_events),
        "duration_s": float(duration),
        "markers_seen": float(follower.markers_seen),
    }


def run_closed_loop(cfg: Optional[RunConfig] = None, layout: Optional[Layout] = None,
                    keep_frames: bool = False) -> RunResult:
    """Drive the whole loop in simulated time: sensor -> CAN bytes -> channel -> follower -> plant."""
    cfg = cfg or RunConfig()
    layout = layout or default_layout(cfg.layout_branch_side)
    world = World(layout, cfg.sensor, cfg.plant, cfg.seed, cfg.start_offset_mm, cfg.start_heading_deg)
    channel = Channel(cfg.channel, np.random.default_rng(cfg.seed + 7919))
    fcfg = replace(cfg.follower, side=layout.route_side(cfg.route), node_id=cfg.sensor.node_id)
    follower = Follower(fcfg)
    route_path = layout.routes[cfg.route]

    ctl_every = max(1, int(round(1.0 / cfg.control_hz / cfg.dt)))
    err_every = 1
    v_cmd = w_cmd = 0.0
    ts, xs, ys, ths, vs, errs = [], [], [], [], [], []
    log: List[tuple] = []
    rx1 = 0
    derailed = False
    stopped_at: Optional[float] = None
    route_len = float(cumulative_length(route_path)[-1])
    max_time = cfg.max_time_s
    if max_time is None:
        max_time = route_len / max(0.8 * fcfg.v_cruise, 0.05) + 20.0
    n = int(max_time / cfg.dt)
    for k in range(n):
        for fr in world.step(v_cmd, w_cmd, cfg.dt):
            channel.send(fr)
        for fr in channel.deliver(world.t):
            follower.on_frame(fr.t, fr.can_id, fr.data)
            if (fr.can_id & 0x780) == 0x180:
                rx1 += 1
            if keep_frames:
                log.append((fr.t, fr.can_id, fr.data))
        if k % ctl_every == 0:
            v_cmd, w_cmd = follower.update(world.t)
        if k % err_every == 0:
            d, signed, _ = nearest_on_polyline(route_path, world.sensor_point())
            ts.append(world.t)
            xs.append(world.plant.x)
            ys.append(world.plant.y)
            ths.append(world.plant.th)
            vs.append(world.plant.v)
            errs.append(signed * 1000.0)
            if d > 0.30:
                derailed = True
                break
        if follower.stopped and world.plant.v < 1e-3:
            stopped_at = stopped_at if stopped_at is not None else world.t
            if world.t - stopped_at > 0.3:
                break
        if follower._lost_since is not None and follower._seen_tape and \
                world.t - follower._lost_since > 3.0:
            break  # gave up: tape lost for 3 s
    metrics = score(world, follower, layout, cfg.route, ts, errs, vs, rx1, world.t, cfg.settle_s, derailed)
    metrics["frames_dropped"] = float(channel.dropped)
    metrics["success"] = float(
        metrics["junction_ok"] == 1.0 and metrics["stopped"] == 1.0 and not derailed
        and abs(metrics["stop_error_mm"]) <= 20.0)
    return RunResult(np.array(ts), np.array(xs), np.array(ys), np.array(ths), np.array(vs),
                     np.array(errs), layout, cfg.route, metrics, cfg, log)
