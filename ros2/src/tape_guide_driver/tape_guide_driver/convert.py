"""Pure conversions between decoded profile frames and the field values of the ROS messages.

No ROS imports, so it is unit-tested in the plain-Python suite (tests/test_ros_convert.py).
Wire units (mm, degrees, percent) become SI units (m, rad, 0..1).
"""

import math

from agvlab import frames


def track_fields(f: frames.TrackFrame) -> dict:
    """Field values for guide_msgs/Track. Absent offsets become 0.0 (check `detected`)."""
    lo = f.left_offset_mm
    ro = f.right_offset_mm
    return {
        "detected": f.tape_detected, "two_tracks": f.two_tracks, "fork": f.fork, "merge": f.merge,
        "junction": f.junction, "marker_left": f.marker_left, "marker_right": f.marker_right,
        "fault": f.fault,
        "offset_left_m": 0.0 if lo is None else lo / 1000.0,
        "offset_right_m": 0.0 if ro is None else ro / 1000.0,
        "heading_left_rad": math.radians(f.left_heading_deg),
        "heading_right_rad": math.radians(f.right_heading_deg),
        "quality": f.quality_pct / 100.0,
    }


def markers_fields(f: frames.MarkerFrame) -> dict:
    """Field values for guide_msgs/Markers."""
    def mm(v):
        return 0.0 if v is None else v / 1000.0

    return {
        "left_valid": f.left_along_raw != frames.NO_DATA and f.left_lateral_raw != frames.NO_DATA,
        "left_along_m": mm(f.left_along_mm), "left_lateral_m": mm(f.left_lateral_mm),
        "right_valid": f.right_along_raw != frames.NO_DATA and f.right_lateral_raw != frames.NO_DATA,
        "right_along_m": mm(f.right_along_mm), "right_lateral_m": mm(f.right_lateral_mm),
    }


def frame_from_track(m) -> frames.TrackFrame:
    """Rebuild a TrackFrame from a guide_msgs/Track (or anything with the same attributes)."""
    flags = dict(two_tracks=bool(m.two_tracks), fork=bool(m.fork), merge=bool(m.merge),
                 junction=bool(m.junction), marker_left=bool(m.marker_left),
                 marker_right=bool(m.marker_right), fault=bool(m.fault))
    if not m.detected:
        return frames.TrackFrame(**flags)
    return frames.TrackFrame.from_physical(
        m.offset_left_m * 1000.0, m.offset_right_m * 1000.0,
        math.degrees(m.heading_left_rad), math.degrees(m.heading_right_rad), m.quality * 100.0, **flags)


def frame_from_markers(m) -> frames.MarkerFrame:
    """Rebuild a MarkerFrame from a guide_msgs/Markers."""
    def raw(valid, v):
        return round(v * 1000.0 / frames.MARKER_LSB_MM) if valid else frames.NO_DATA

    return frames.MarkerFrame(
        raw(m.left_valid, m.left_along_m), raw(m.left_valid, m.left_lateral_m),
        raw(m.right_valid, m.right_along_m), raw(m.right_valid, m.right_lateral_m))
