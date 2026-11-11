"""Tape-guide sensor model: robot pose + tape layout -> CAN frames of the open profile.

The sensing line is `2 * half_width_mm` wide, perpendicular to the robot heading, `mount_forward_m`
ahead of the drive axle. Where a tape centreline crosses it gives a lateral offset (reported left
positive) and a heading angle (counter-clockwise positive).

Everything numeric about the sensor is an ASSUMPTION of this lab, not a measurement of hardware:
75 mm half width, 0.3 mm rms offset noise, 1 deg rms heading noise, 0.1 mm / 0.5 deg wire resolution,
quality ramp, 12 mm tape merge distance, 40 mm marker window. See README "What is not verified yet".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from . import frames
from .layout import Layout

Pose = Tuple[float, float, float]


class CanFrame(NamedTuple):
    t: float
    can_id: int
    data: bytes


@dataclass
class SensorConfig:
    node_id: int = frames.DEFAULT_NODE_ID
    tpdo1_ms: float = 10.0
    tpdo2_ms: float = 20.0
    tpdo3_ms: float = 50.0
    heartbeat_ms: float = 1000.0
    mount_forward_m: float = 0.25
    half_width_mm: float = 75.0
    pos_sigma_mm: float = 0.3  # assumed rms offset noise
    angle_sigma_deg: float = 1.0
    marker_sigma_mm: float = 0.1
    merge_mm: float = 12.0  # two tape centrelines closer than this read as one track (assumed)
    tape_width_mm: float = 25.0  # narrow tape; the fork flag needs the tracks within one tape width
    marker_resolution_mm: float = 0.25  # assumed marker position resolution
    full_quality_mm: float = 35.0  # |offset| up to here: quality 100 %, then a linear ramp (assumed)
    detect_mm: float = 70.0  # beyond this the tape is not seen: quality 20 % at the limit, 0 outside
    min_quality_pct: float = 20.0
    marker_window_mm: float = 40.0  # longitudinal window in which a marker is reported (assumed)
    burst_hz: float = 0.0  # tape-lost bursts per second (magnetic glitches)
    burst_ms: float = 80.0
    invert_position: bool = False
    invert_angle: bool = False


@dataclass
class Reading:
    """Noise-free geometric truth for one sensing instant."""

    tracks: List[Tuple[float, float]]  # [(u_mm, phi_rad)] clustered, sorted left -> right
    intersection: bool
    markers: List[Tuple[float, float]]  # [(u_mm, y_mm)] inside the window


def _frame_vectors(pose: Pose, forward_m: float):
    x, y, th = pose
    f = np.array([math.cos(th), math.sin(th)])
    r = np.array([math.sin(th), -math.cos(th)])  # right of heading
    c = np.array([x, y]) + f * forward_m
    return c, f, r


def crossings(pose: Pose, tapes: List[np.ndarray], forward_m: float,
              half_width_mm: float) -> List[Tuple[float, float]]:
    """All tape crossings of the sensing line as (u_mm right-positive, phi_rad ccw tangent angle)."""
    c, f, r = _frame_vectors(pose, forward_m)
    hw = half_width_mm / 1000.0
    a = c - r * hw
    ab = 2 * hw * r
    out: List[Tuple[float, float]] = []
    for path in tapes:
        p0 = path[:-1]
        d = path[1:] - p0
        # solve p0 + t d = a + s ab
        den = d[:, 0] * ab[1] - d[:, 1] * ab[0]
        ok = np.abs(den) > 1e-12
        den_safe = np.where(ok, den, 1.0)
        ap = a - p0
        t = (ap[:, 0] * ab[1] - ap[:, 1] * ab[0]) / den_safe
        s = (ap[:, 0] * d[:, 1] - ap[:, 1] * d[:, 0]) / den_safe
        hit = ok & (t >= 0.0) & (t < 1.0) & (s >= 0.0) & (s <= 1.0)
        for i in np.nonzero(hit)[0]:
            di = d[i] / math.hypot(d[i, 0], d[i, 1])
            if di @ f < 0:
                di = -di
            phi = math.atan2(f[0] * di[1] - f[1] * di[0], f @ di)
            out.append(((s[i] - 0.5) * 2 * half_width_mm, phi))
    return out


def cluster(cross: List[Tuple[float, float]], merge_mm: float) -> List[Tuple[float, float]]:
    """Merge crossings closer than merge_mm (tape width / joint overlap). Sorted left -> right."""
    if not cross:
        return []
    cross = sorted(cross)
    groups: List[List[Tuple[float, float]]] = [[cross[0]]]
    for item in cross[1:]:
        if item[0] - groups[-1][-1][0] < merge_mm:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [(sum(g[0] for g in grp) / len(grp), sum(g[1] for g in grp) / len(grp)) for grp in groups]


def quality_for(offset_mm: float, cfg: SensorConfig) -> float:
    """Signal quality in percent: 100 near the centre, linear ramp down to `min_quality_pct` at the limit."""
    a = abs(offset_mm)
    if a > cfg.detect_mm:
        return 0.0
    if a <= cfg.full_quality_mm:
        return 100.0
    frac = (a - cfg.full_quality_mm) / (cfg.detect_mm - cfg.full_quality_mm)
    return 100.0 - frac * (100.0 - cfg.min_quality_pct)


class SensorSim:
    """Emits TPDO1/2/3 and heartbeat frames on their own timers as the pose changes."""

    def __init__(self, layout: Layout, cfg: Optional[SensorConfig] = None,
                 rng: Optional[np.random.Generator] = None):
        self.layout = layout
        self.cfg = cfg or SensorConfig()
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.ids = frames.cob_ids(self.cfg.node_id)
        self._next = {"tpdo1": 0.0, "tpdo2": 0.0, "tpdo3": 0.0, "heartbeat": 0.0}
        self._sample = 0  # TPDO1 sample counter (reported in TPDO3)
        self._burst_until = -1.0
        self._next_burst = self._draw_burst_gap(0.0)
        self.last_reading: Optional[Reading] = None
        self.bursts = 0

    # -------------------------------------------------------------- truth
    def measure(self, pose: Pose) -> Reading:
        cfg = self.cfg
        cr = crossings(pose, self.layout.tapes, cfg.mount_forward_m, cfg.half_width_mm)
        # a tape beyond the detection limit is not seen at all, even inside the sensing width
        cr = [c for c in cr if abs(c[0]) <= cfg.detect_mm]
        tracks = cluster(cr, cfg.merge_mm)
        intersection = any(abs(phi) > math.radians(60.0) for _, phi in tracks)
        c, f, r = _frame_vectors(pose, cfg.mount_forward_m)
        marks = []
        for m in self.layout.markers:
            v = np.array([m.x, m.y]) - c
            u_mm = float(v @ r) * 1000.0
            y_mm = float(v @ f) * 1000.0
            if abs(u_mm) <= cfg.half_width_mm and abs(y_mm) <= cfg.marker_window_mm:
                marks.append((u_mm, y_mm))
        return Reading(tracks, intersection, marks)

    # ------------------------------------------------------------- bursts
    def _draw_burst_gap(self, t: float) -> float:
        if self.cfg.burst_hz <= 0:
            return math.inf
        return t + float(self.rng.exponential(1.0 / self.cfg.burst_hz))

    def _in_burst(self, t: float) -> bool:
        if t >= self._next_burst and t >= self._burst_until:
            self._burst_until = t + self.cfg.burst_ms / 1000.0
            self._next_burst = self._draw_burst_gap(self._burst_until)
            self.bursts += 1
        return t < self._burst_until

    # ----------------------------------------------------------- encoding
    def _q(self, v: float) -> int:
        return int(math.floor(v + 0.5))

    def encode_track(self, rd: Reading, burst: bool) -> frames.TrackFrame:
        cfg = self.cfg
        if burst or not rd.tracks:
            return frames.TrackFrame()
        left, right = rd.tracks[0], rd.tracks[-1]
        single = len(rd.tracks) == 1
        n = self.rng.normal
        # internal u is right-positive, phi counter-clockwise: the profile reports offset left-positive
        lo = -left[0] + n(0.0, cfg.pos_sigma_mm)
        lh = math.degrees(left[1]) + n(0.0, cfg.angle_sigma_deg)
        if single:
            ro, rh = lo, lh
        else:
            ro = -right[0] + n(0.0, cfg.pos_sigma_mm)
            rh = math.degrees(right[1]) + n(0.0, cfg.angle_sigma_deg)
        if rd.intersection:
            lh = rh = 0.0
        q = max(quality_for(left[0], cfg), quality_for(right[0], cfg))
        if q <= 0.0:
            return frames.TrackFrame()
        so = -1 if cfg.invert_position else 1
        sh = -1 if cfg.invert_angle else 1
        # fork / merge: the two tracks lie within one tape width and their headings differ by 10..45 deg
        diff = math.degrees(left[1] - right[1]) if not single else 0.0
        near = (not single) and abs(right[0] - left[0]) <= cfg.tape_width_mm
        fork = near and 10.0 <= diff < 45.0
        merge = near and -45.0 < diff <= -10.0
        ml = mr = False
        if not rd.intersection:
            for u, _y in rd.markers:
                if u < (left[0] + right[0]) / 2:
                    ml = True
                else:
                    mr = True
        return frames.TrackFrame.from_physical(
            so * lo, so * ro, sh * lh, sh * rh, q, two_tracks=not single, fork=fork, merge=merge,
            junction=rd.intersection, marker_left=ml, marker_right=mr)

    def encode_markers(self, rd: Reading, burst: bool) -> frames.MarkerFrame:
        cfg = self.cfg
        if burst or not rd.markers or rd.intersection:
            return frames.MarkerFrame()
        sx = -1.0 if cfg.invert_position else 1.0
        slots = {"l": None, "r": None}
        ref = (rd.tracks[0][0] + rd.tracks[-1][0]) / 2 if rd.tracks else None
        for u, y in rd.markers:
            u += self.rng.normal(0.0, cfg.marker_sigma_mm)
            if ref is None:
                slots["l"] = slots["r"] = (u, y)  # lone marker: reported on both sides
            elif u < ref:
                slots["l"] = (u, y)
            else:
                slots["r"] = (u, y)
        res = cfg.marker_resolution_mm

        def raw_of(mm: float) -> int:  # quantise to the marker resolution, then to wire LSBs
            return int(round(self._q(mm / res) * res / frames.MARKER_LSB_MM))

        raw = []
        for k in ("l", "r"):
            v = slots[k]
            # along first, then lateral (left positive: internal u is right-positive)
            raw += [frames.NO_DATA, frames.NO_DATA] if v is None else [raw_of(v[1]), raw_of(-sx * v[0])]
        return frames.MarkerFrame(*raw)

    # --------------------------------------------------------------- step
    def step(self, t: float, pose: Pose) -> List[CanFrame]:
        """Emit every frame whose timer expired at or before `t`."""
        cfg = self.cfg
        out: List[CanFrame] = []
        due = [k for k, v in self._next.items() if v <= t + 1e-9]
        if not due:
            return out
        burst = self._in_burst(t)
        rd = self.measure(pose) if ("tpdo1" in due or "tpdo2" in due) else self.last_reading
        self.last_reading = rd
        periods = {"tpdo1": cfg.tpdo1_ms, "tpdo2": cfg.tpdo2_ms, "tpdo3": cfg.tpdo3_ms,
                   "heartbeat": cfg.heartbeat_ms}
        for k in due:
            if k == "tpdo1":
                data = frames.encode_tpdo1(self.encode_track(rd, burst))
                self._sample = (self._sample + 1) & 0xFFFF
            elif k == "tpdo2":
                data = frames.encode_tpdo2(self.encode_markers(rd, burst))
            elif k == "tpdo3":
                data = frames.encode_tpdo3(frames.DiagFrame(self._sample, 0))
            else:
                data = bytes([frames.NMT_OPERATIONAL])
            out.append(CanFrame(self._next[k], self.ids[k], data))
            while self._next[k] <= t + 1e-9:
                self._next[k] += periods[k] / 1000.0
        return out
