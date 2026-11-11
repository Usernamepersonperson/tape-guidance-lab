"""Kinematic differential-drive plant with first-order velocity response, and the CAN channel model."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .sensor import CanFrame


@dataclass
class PlantConfig:
    track_width_m: float = 0.40
    tau_v_s: float = 0.10  # first-order lag of linear velocity
    tau_w_s: float = 0.10  # first-order lag of yaw rate
    v_max: float = 1.0
    w_max: float = 2.0
    a_max: float = 1.5  # m/s^2 linear acceleration limit


class DiffDrive:
    """Unicycle model (v, omega) with lagged response; pose is the drive-axle centre."""

    def __init__(self, pose=(0.0, 0.0, 0.0), cfg: Optional[PlantConfig] = None):
        self.cfg = cfg or PlantConfig()
        self.x, self.y, self.th = pose
        self.v = 0.0
        self.w = 0.0

    @property
    def pose(self):
        return (self.x, self.y, self.th)

    def step(self, v_cmd: float, w_cmd: float, dt: float) -> None:
        c = self.cfg
        v_cmd = max(-c.v_max, min(c.v_max, v_cmd))
        w_cmd = max(-c.w_max, min(c.w_max, w_cmd))
        dv = (v_cmd - self.v) * (1 - math.exp(-dt / c.tau_v_s))
        dv = max(-c.a_max * dt, min(c.a_max * dt, dv))
        self.v += dv
        self.w += (w_cmd - self.w) * (1 - math.exp(-dt / c.tau_w_s))
        mid = self.th + 0.5 * self.w * dt
        self.x += self.v * math.cos(mid) * dt
        self.y += self.v * math.sin(mid) * dt
        self.th += self.w * dt


@dataclass
class ChannelConfig:
    latency_ms: float = 0.0  # fixed one-way delay sensor -> follower
    jitter_ms: float = 0.0  # |N(0, jitter)| added per frame (bus-order preserved)
    drop_prob: float = 0.0  # independent frame loss


class Channel:
    """Delays and drops frames in simulation time. Delivery order equals send order, like a CAN bus."""

    def __init__(self, cfg: Optional[ChannelConfig] = None, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg or ChannelConfig()
        self.rng = rng if rng is not None else np.random.default_rng(1)
        self._heap: list = []
        self._seq = 0
        self._last_delivery = 0.0
        self.sent = 0
        self.dropped = 0

    def send(self, frame: CanFrame) -> None:
        self.sent += 1
        if self.cfg.drop_prob > 0 and self.rng.random() < self.cfg.drop_prob:
            self.dropped += 1
            return
        delay = self.cfg.latency_ms / 1000.0
        if self.cfg.jitter_ms > 0:
            delay += abs(float(self.rng.normal(0.0, self.cfg.jitter_ms / 1000.0)))
        when = max(frame.t + delay, self._last_delivery)
        self._last_delivery = when
        self._seq += 1
        heapq.heappush(self._heap, (when, self._seq, frame))

    def deliver(self, t: float) -> List[CanFrame]:
        """Frames whose delivery time is <= t, as (delivery time, id, data)."""
        out = []
        while self._heap and self._heap[0][0] <= t + 1e-12:
            when, _, f = heapq.heappop(self._heap)
            out.append(CanFrame(when, f.can_id, f.data))
        return out
