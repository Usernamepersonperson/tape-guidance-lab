"""2D tape layouts: polylines for the tape, route polylines for the intended path, stop markers.

World frame: x east, y north, metres, heading counter-clockwise from +x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

STEP_M = 0.005  # polyline sampling


def build_path(start_xy: Tuple[float, float], heading: float,
               segments: Sequence[Tuple], step: float = STEP_M) -> np.ndarray:
    """Build a polyline from ("line", length_m) and ("arc", radius_m, angle_deg) segments.

    Positive arc angle turns left (counter-clockwise). Returns an (N, 2) array.
    """
    pts = [np.array(start_xy, dtype=float)]
    x, y = float(start_xy[0]), float(start_xy[1])
    th = float(heading)
    for seg in segments:
        if seg[0] == "line":
            n = max(1, int(round(seg[1] / step)))
            for _ in range(n):
                x += math.cos(th) * seg[1] / n
                y += math.sin(th) * seg[1] / n
                pts.append(np.array([x, y]))
        elif seg[0] == "arc":
            r, ang = float(seg[1]), math.radians(float(seg[2]))
            arc_len = abs(r * ang)
            n = max(1, int(round(arc_len / step)))
            dth = ang / n
            for _ in range(n):
                # exact chord step along a circle of radius r
                chord = 2 * r * math.sin(abs(dth) / 2)
                mid = th + dth / 2
                x += chord * math.cos(mid)
                y += chord * math.sin(mid)
                th += dth
                pts.append(np.array([x, y]))
        else:
            raise ValueError(f"unknown segment {seg[0]!r}")
    return np.array(pts)


def cumulative_length(path: np.ndarray) -> np.ndarray:
    d = np.hypot(*np.diff(path, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def point_at(path: np.ndarray, s: float) -> Tuple[np.ndarray, float]:
    """Point and heading at arc length s along path."""
    cl = cumulative_length(path)
    s = float(np.clip(s, 0.0, cl[-1]))
    i = int(np.clip(np.searchsorted(cl, s) - 1, 0, len(path) - 2))
    seg = path[i + 1] - path[i]
    seg_len = max(cl[i + 1] - cl[i], 1e-12)
    p = path[i] + seg * ((s - cl[i]) / seg_len)
    return p, math.atan2(seg[1], seg[0])


@dataclass(frozen=True)
class Marker:
    """A point magnet beside the tape (stop marker)."""

    x: float
    y: float
    route: str
    role: str = "stop"  # "slow" (deceleration marker) or "stop"


@dataclass
class Layout:
    """Tape layout. `tapes` are what the sensor sees; `routes` are intended paths for scoring."""

    tapes: List[np.ndarray]
    routes: Dict[str, np.ndarray]
    markers: List[Marker]
    start_pose: Tuple[float, float, float]
    fork_s: float  # arc length along the spine where the branch leaves
    branch_side: str  # "right" or "left"
    post_fork: Dict[str, np.ndarray] = field(default_factory=dict)  # tape beyond the fork per route
    meta: dict = field(default_factory=dict)

    def route_side(self, route: str) -> str:
        """Which reported track ('left'/'right') leads onto the given route after the fork."""
        if route == "branch":
            return self.branch_side
        return "left" if self.branch_side == "right" else "right"

    def marker_for(self, route: str) -> Marker:
        return next(m for m in self.markers if m.route == route and m.role == "stop")


def default_layout(branch_side: str = "right", marker_offset_mm: float = 55.0,
                   branch_angle_deg: float = 25.0, curve_radius_m: float = 1.2) -> Layout:
    """straight 2.5 m -> 90 deg left curve (R 1.2 m) -> straight 1.5 m -> fork -> stop marker.

    The spine continues straight through the fork; the branch leaves it on `branch_side`
    (arc R 2.0 m through `branch_angle_deg`, then straight). Each route carries a deceleration
    marker and a stop marker, placed `marker_offset_mm` to the left of the tape.
    """
    if branch_side not in ("left", "right"):
        raise ValueError("branch_side must be 'left' or 'right'")
    sgn = 1.0 if branch_side == "left" else -1.0
    pre = [("line", 2.5), ("arc", curve_radius_m, 90.0), ("line", 1.5)]
    after_fork_len = 3.0
    spine = build_path((0.0, 0.0), 0.0, pre + [("line", after_fork_len)])
    pre_path = build_path((0.0, 0.0), 0.0, pre)
    fork_s = float(cumulative_length(pre_path)[-1])
    fork_pt, fork_th = point_at(spine, fork_s)
    branch = build_path(tuple(fork_pt), fork_th,
                        [("arc", 2.0, sgn * branch_angle_deg), ("line", after_fork_len - 0.5)])

    idx = int(np.searchsorted(cumulative_length(spine), fork_s))
    routes = {
        "straight": spine,
        "branch": np.vstack([spine[: idx + 1], branch[1:]]),
    }
    post = {"straight": spine[idx:], "branch": branch}

    markers = []
    # deceleration marker 1.6 m and stop marker 2.2 m past the fork point on each route
    stop_s_after_fork = 2.2
    slow_s_after_fork = 1.6
    for name, route in routes.items():
        for role, s_after in (("slow", slow_s_after_fork), ("stop", stop_s_after_fork)):
            p, th = point_at(route, fork_s + s_after)
            left = np.array([-math.sin(th), math.cos(th)])
            mp = p + left * (marker_offset_mm / 1000.0)
            markers.append(Marker(float(mp[0]), float(mp[1]), name, role))

    return Layout(
        tapes=[spine, branch],
        routes=routes,
        markers=markers,
        start_pose=(0.0, 0.0, 0.0),
        fork_s=fork_s,
        branch_side=branch_side,
        post_fork=post,
        meta={"stop_s_after_fork": stop_s_after_fork, "marker_offset_mm": marker_offset_mm},
    )


def nearest_on_polyline(path: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    """Return (distance, signed lateral offset, arc length) of the nearest point on `path`.

    The offset is positive when `p` is to the left of the path direction.
    """
    a = path[:-1]
    b = path[1:]
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(axis=1), 1e-18)
    t = np.clip(((p - a) * ab).sum(axis=1) / ab2, 0.0, 1.0)
    proj = a + ab * t[:, None]
    d = np.hypot(*(p - proj).T)
    i = int(np.argmin(d))
    cross = ab[i, 0] * (p[1] - a[i, 1]) - ab[i, 1] * (p[0] - a[i, 0])
    seg_len = math.sqrt(float(ab2[i]))
    cl_before = float(np.hypot(*np.diff(path[: i + 1], axis=0).T).sum()) if i > 0 else 0.0
    return float(d[i]), float(math.copysign(d[i], cross)), cl_before + float(t[i]) * seg_len
