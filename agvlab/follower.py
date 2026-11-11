"""ROS-free tape follower: decoded profile frames in, (v, omega) out.

Sign convention (ROS REP-103): the lateral offset is LEFT-positive and headings are counter-clockwise
positive, so omega = +(...) steers toward the tape.

Controllers
  pd            omega = kp*pos + kd*d(pos)/dt + (v/L)*ka*heading   (L = sensor distance ahead of the axle)
  pure_pursuit  aim at the tape point on the sensing line: curvature 2*y/(L^2+y^2), omega = v*curvature
  p_only        naive baseline omega = 5.0*pos + 1.0*heading at 0.15 m/s, no marker handling, for comparison

Markers: the first marker edge switches to creep speed (deceleration marker), the second is the
stop marker: the speed profile v = sqrt(2*a*d) drives the marker's longitudinal offset to zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from . import frames


@dataclass
class FollowerConfig:
    controller: str = "pd"  # pd | pure_pursuit | p_only
    side: str = "left"  # which reported track to follow ("left" | "right"); picks the fork branch
    v_cruise: float = 0.30
    v_creep: float = 0.10
    kp: float = 14.0  # rad/s per metre (pd)
    kd: float = 0.9  # rad/s per (m/s) (pd)
    ka: float = 1.0  # scales the curvature feed-forward v*angle/L (1.0 = geometric)
    deriv_tau_s: float = 0.08
    lookahead_m: float = 0.25  # sensor distance ahead of the axle (pure pursuit)
    w_max: float = 1.5
    marker_min_spacing_m: float = 0.3  # marker edges closer than this (commanded travel) are one marker
    stale_s: float = 0.2
    hold_s: float = 0.15  # keep last omega at half speed this long after the tape is lost
    stop_enabled: bool = True
    a_stop: float = 0.8
    lag_comp_s: float = 0.10  # plant lag assumed by the stop profile
    node_id: int = frames.DEFAULT_NODE_ID
    invert_position: bool = False  # mirrored mounting
    invert_angle: bool = False

    @classmethod
    def baseline(cls, side: str = "left") -> "FollowerConfig":
        return cls(controller="p_only", side=side, v_cruise=0.15, w_max=1.0, hold_s=0.0,
                   stop_enabled=False)


class Follower:
    def __init__(self, cfg: Optional[FollowerConfig] = None):
        self.cfg = cfg or FollowerConfig()
        self._track: Optional[frames.TrackFrame] = None
        self._track_t = -math.inf
        self._markers: Optional[frames.MarkerFrame] = None
        self._marker_flag_t = -math.inf
        self._marker_present_prev = False
        self._marker_gap_start = -math.inf
        self._odo = 0.0  # travelled distance from the commanded speed (m)
        self._odo_at_marker = -math.inf
        self.markers_seen = 0
        self._marker_y_m: Optional[float] = None
        self._prev_pos_mm: Optional[float] = None
        self._prev_pos_t = 0.0
        self._dpos = 0.0  # m/s, filtered
        self._last_w = 0.0
        self._lost_since: Optional[float] = None
        self._last_update_t: Optional[float] = None
        self._last_v = 0.0
        self.stopped = False
        self.lost_events = 0
        self._seen_tape = False
        self.frames_rx = 0
        self.tpdo1_rx = 0
        self.decode_errors = 0
        self.stop_marker_y_mm: Optional[float] = None

    # ---------------------------------------------------------- inputs
    def on_frame(self, t: float, can_id: int, data: bytes) -> None:
        """Feed one received CAN frame (t = receive time in seconds)."""
        try:
            dec = frames.decode_frame(can_id, data, self.cfg.node_id)
        except frames.DecodeError:
            self.decode_errors += 1
            return
        if dec is None:
            return
        self.frames_rx += 1
        if isinstance(dec, frames.TrackFrame):
            self.on_track(t, dec)
        elif isinstance(dec, frames.MarkerFrame):
            self.on_markers(t, dec)

    def on_track(self, t: float, frame: frames.TrackFrame) -> None:
        """Feed a decoded TPDO1 (the ROS node builds this from a `guide_msgs/Track` message)."""
        self.tpdo1_rx += 1
        self._on_track(t, frame)

    def on_markers(self, t: float, frame: frames.MarkerFrame) -> None:
        """Feed a decoded TPDO2 (from a `guide_msgs/Markers` message in the ROS node)."""
        self._markers = frame
        self._update_marker_y(t)

    def _on_track(self, t: float, f: frames.TrackFrame) -> None:
        self._track = f
        self._track_t = t
        c = self.cfg
        if f.tape_detected:
            pos = self._pick(f)[0]
            if self._prev_pos_mm is not None and t > self._prev_pos_t + 1e-4:
                dt = t - self._prev_pos_t
                raw = (pos - self._prev_pos_mm) / 1000.0 / dt
                if abs(pos - self._prev_pos_mm) > 15:  # branch hand-over jump: do not differentiate
                    raw = self._dpos
                a = 1 - math.exp(-dt / c.deriv_tau_s)
                self._dpos += a * (raw - self._dpos)
            self._prev_pos_mm, self._prev_pos_t = pos, t
        else:
            self._prev_pos_mm = None
        present = f.marker_left or f.marker_right
        if present:
            if (not self._marker_present_prev and t - self._marker_gap_start >= 0.25
                    and self._odo - self._odo_at_marker >= c.marker_min_spacing_m):
                self.markers_seen += 1
                self._odo_at_marker = self._odo
                self._marker_y_m = None  # forget the previous marker's offset
                self._markers = None
            self._marker_flag_t = t
        elif self._marker_present_prev:
            self._marker_gap_start = t
        self._marker_present_prev = present
        self._update_marker_y(t)

    def _pick(self, f: frames.TrackFrame) -> Tuple[float, float]:
        sp = -1.0 if self.cfg.invert_position else 1.0
        sa = -1.0 if self.cfg.invert_angle else 1.0
        if self.cfg.side == "right":
            return sp * (f.right_offset_mm or 0.0), sa * f.right_heading_deg
        return sp * (f.left_offset_mm or 0.0), sa * f.left_heading_deg

    def _update_marker_y(self, t: float) -> None:
        """Take the marker's longitudinal offset from TPDO2 while TPDO1 flags a marker."""
        f, m = self._track, self._markers
        if f is None or m is None or t - self._marker_flag_t > 0.05:
            return
        if f.marker_left:
            along = m.left_along_mm
        elif f.marker_right:
            along = m.right_along_mm
        else:
            return
        if along is None:
            return  # empty slot
        self._marker_y_m = along / 1000.0

    # --------------------------------------------------------- control
    def update(self, t: float) -> Tuple[float, float]:
        """Return (v m/s, omega rad/s). Call at the control rate (e.g. 50 Hz)."""
        c = self.cfg
        dt = 0.02 if self._last_update_t is None else max(1e-4, t - self._last_update_t)
        self._last_update_t = t
        self._odo += self._last_v * dt
        if self.stopped:
            return 0.0, 0.0

        f = self._track
        fresh = f is not None and (t - self._track_t) <= c.stale_s and f.tape_detected
        if fresh:
            self._seen_tape = True
        if not fresh:
            if self._lost_since is None:
                self._lost_since = t
                if self._seen_tape:
                    self.lost_events += 1
            if self._seen_tape and t - self._lost_since <= c.hold_s and c.hold_s > 0:
                self._last_v = 0.5 * c.v_cruise
                return self._last_v, self._last_w
            self._last_w = 0.0
            self._last_v = 0.0
            return 0.0, 0.0
        self._lost_since = None

        pos_mm, ang_deg = self._pick(f)
        pos_m = pos_mm / 1000.0
        ang = math.radians(ang_deg)
        v = c.v_cruise
        if self.markers_seen >= 1 and c.stop_enabled:
            v = c.v_creep

        if c.controller == "p_only":
            w = 5.0 * pos_m + 1.0 * ang
        elif c.controller == "pure_pursuit":
            yt = pos_m  # left-positive lateral target at distance L
            kappa = 2.0 * yt / (c.lookahead_m ** 2 + yt ** 2)
            w = v * kappa
        elif c.controller == "pd":
            # on a circle of curvature k the tape crosses a sensing line L ahead of the axle at an
            # angle of about k*L, so v*angle/L is the yaw rate (v*k) the curve needs
            w = c.kp * pos_m + c.kd * self._dpos + c.ka * v / c.lookahead_m * ang
        else:
            raise ValueError(f"unknown controller {c.controller!r}")
        w = max(-c.w_max, min(c.w_max, w))

        if c.stop_enabled and self.markers_seen >= 2 and self._marker_y_m is not None:
            # dead-reckon between TPDO2 frames with the last commanded speed
            self._marker_y_m -= self._last_v * dt
            d = self._marker_y_m - self._last_v * c.lag_comp_s
            if d <= 0.0005:
                self.stopped = True
                self.stop_marker_y_mm = self._marker_y_m * 1000.0
                return 0.0, 0.0
            v = min(v, math.sqrt(2.0 * c.a_stop * d))
        self._last_v = v
        self._last_w = w
        return v, w
