"""Frame layouts of the open tape-guide sensor profile (encoder + decoder + object dictionary).

Defined in docs/PROTOCOL.md; this file is the reference implementation. CANopen (CiA 301)
conventions: 11-bit COB-IDs = function code + node ID, little-endian PDOs, heartbeat producer.

    TPDO1 "Track"    0x180+node  8 B  i16 offset_left, i16 offset_right (0.1 mm, left +),
                                      i8 heading_left, i8 heading_right (0.5 deg, ccw +),
                                      u8 flags, u8 quality (0.5 %)
    TPDO2 "Markers"  0x280+node  8 B  i16 left_along, left_lateral, right_along, right_lateral (0.05 mm)
    TPDO3 "Diag"     0x380+node  3 B  u16 sample counter, u8 error register
    Heartbeat        0x700+node  1 B  NMT state

    flags bit0..7: track | two_tracks | fork | merge | junction | marker_left | marker_right | fault
    0x8000 in an i16 field means "no data".
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

FC_TPDO1 = 0x180
FC_TPDO2 = 0x280
FC_TPDO3 = 0x380
FC_HEARTBEAT = 0x700
DEFAULT_NODE_ID = 21

NMT_BOOTUP = 0x00
NMT_STOPPED = 0x04
NMT_OPERATIONAL = 0x05
NMT_PREOPERATIONAL = 0x7F

NO_DATA = -32768  # 0x8000 as int16

# flag bits of TPDO1 byte 6
FL_TRACK = 0x01
FL_TWO_TRACKS = 0x02
FL_FORK = 0x04
FL_MERGE = 0x08
FL_JUNCTION = 0x10
FL_MARKER_LEFT = 0x20
FL_MARKER_RIGHT = 0x40
FL_FAULT = 0x80

OFFSET_LSB_MM = 0.1
HEADING_LSB_DEG = 0.5
MARKER_LSB_MM = 0.05
QUALITY_LSB_PCT = 0.5

_T1 = struct.Struct("<hhbbBB")
_T2 = struct.Struct("<hhhh")
_T3 = struct.Struct("<HB")


class DecodeError(ValueError):
    """Frame has the wrong payload length for its PDO."""


def _opt_mm(raw: int, lsb: float) -> Optional[float]:
    return None if raw == NO_DATA else raw * lsb


@dataclass(frozen=True)
class TrackFrame:
    """Decoded TPDO1. Raw integers as on the wire; properties give mm / degrees / percent.

    Offsets are left-positive (ROS REP-103 y axis), headings counter-clockwise positive.
    On a single tape both slots carry the same values and `two_tracks` is False.
    """

    left_offset_raw: int = NO_DATA
    right_offset_raw: int = NO_DATA
    left_heading_raw: int = 0
    right_heading_raw: int = 0
    quality_raw: int = 0
    detected: bool = False
    two_tracks: bool = False
    fork: bool = False
    merge: bool = False
    junction: bool = False
    marker_left: bool = False
    marker_right: bool = False
    fault: bool = False

    @classmethod
    def from_physical(cls, left_mm: Optional[float], right_mm: Optional[float] = None,
                      left_deg: float = 0.0, right_deg: Optional[float] = None,
                      quality_pct: float = 100.0, **flags) -> "TrackFrame":
        """Build from mm / degrees (rounded to the wire resolution). right_* default to the left values."""
        if left_mm is None:
            return cls(**flags)
        right_mm = left_mm if right_mm is None else right_mm
        right_deg = left_deg if right_deg is None else right_deg
        flags.setdefault("detected", True)
        flags.setdefault("two_tracks", right_mm != left_mm or right_deg != left_deg)
        return cls(round(left_mm / OFFSET_LSB_MM), round(right_mm / OFFSET_LSB_MM),
                   round(left_deg / HEADING_LSB_DEG), round(right_deg / HEADING_LSB_DEG),
                   round(quality_pct / QUALITY_LSB_PCT), **flags)

    @property
    def left_offset_mm(self) -> Optional[float]:
        return _opt_mm(self.left_offset_raw, OFFSET_LSB_MM)

    @property
    def right_offset_mm(self) -> Optional[float]:
        return _opt_mm(self.right_offset_raw, OFFSET_LSB_MM)

    @property
    def left_heading_deg(self) -> float:
        return self.left_heading_raw * HEADING_LSB_DEG

    @property
    def right_heading_deg(self) -> float:
        return self.right_heading_raw * HEADING_LSB_DEG

    @property
    def quality_pct(self) -> float:
        return self.quality_raw * QUALITY_LSB_PCT

    @property
    def tape_detected(self) -> bool:
        return self.detected and self.left_offset_raw != NO_DATA

    @property
    def single_track(self) -> bool:
        return not self.two_tracks


@dataclass(frozen=True)
class MarkerFrame:
    """Decoded TPDO2 (raw int16, 0.05 mm). `along` = forward of the sensing line, `lateral` = left positive."""

    left_along_raw: int = NO_DATA
    left_lateral_raw: int = NO_DATA
    right_along_raw: int = NO_DATA
    right_lateral_raw: int = NO_DATA

    @property
    def left_along_mm(self) -> Optional[float]:
        return _opt_mm(self.left_along_raw, MARKER_LSB_MM)

    @property
    def left_lateral_mm(self) -> Optional[float]:
        return _opt_mm(self.left_lateral_raw, MARKER_LSB_MM)

    @property
    def right_along_mm(self) -> Optional[float]:
        return _opt_mm(self.right_along_raw, MARKER_LSB_MM)

    @property
    def right_lateral_mm(self) -> Optional[float]:
        return _opt_mm(self.right_lateral_raw, MARKER_LSB_MM)


@dataclass(frozen=True)
class DiagFrame:
    sample_counter: int  # increments with every TPDO1 sample, wraps at 65536
    error_register: int  # CiA 301 object 1001h, 0 = no error


@dataclass(frozen=True)
class HeartbeatFrame:
    state: int


Decoded = Union[TrackFrame, MarkerFrame, DiagFrame, HeartbeatFrame]


def cob_ids(node_id: int = DEFAULT_NODE_ID) -> dict:
    if not 1 <= node_id <= 127:
        raise ValueError(f"node_id must be 1..127, got {node_id}")
    return {
        "tpdo1": FC_TPDO1 + node_id,
        "tpdo2": FC_TPDO2 + node_id,
        "tpdo3": FC_TPDO3 + node_id,
        "heartbeat": FC_HEARTBEAT + node_id,
    }


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def encode_flags(f: TrackFrame) -> int:
    bits = ((FL_TRACK if f.detected else 0) | (FL_TWO_TRACKS if f.two_tracks else 0)
            | (FL_FORK if f.fork else 0) | (FL_MERGE if f.merge else 0)
            | (FL_JUNCTION if f.junction else 0) | (FL_MARKER_LEFT if f.marker_left else 0)
            | (FL_MARKER_RIGHT if f.marker_right else 0) | (FL_FAULT if f.fault else 0))
    return bits


def encode_tpdo1(f: TrackFrame) -> bytes:
    """Encode a TrackFrame; fields saturate instead of wrapping (-32768 stays reserved for no data)."""
    def off(v: int) -> int:
        return NO_DATA if v == NO_DATA else _clamp(v, -32767, 32767)

    return _T1.pack(off(f.left_offset_raw), off(f.right_offset_raw),
                    _clamp(f.left_heading_raw, -128, 127), _clamp(f.right_heading_raw, -128, 127),
                    encode_flags(f), _clamp(f.quality_raw, 0, 200))


def decode_tpdo1(data: bytes) -> TrackFrame:
    if len(data) != 8:
        raise DecodeError(f"TPDO1 needs 8 bytes, got {len(data)}")
    lo, ro, lh, rh, fl, q = _T1.unpack(bytes(data))
    return TrackFrame(
        lo, ro, lh, rh, q,
        detected=bool(fl & FL_TRACK), two_tracks=bool(fl & FL_TWO_TRACKS), fork=bool(fl & FL_FORK),
        merge=bool(fl & FL_MERGE), junction=bool(fl & FL_JUNCTION),
        marker_left=bool(fl & FL_MARKER_LEFT), marker_right=bool(fl & FL_MARKER_RIGHT),
        fault=bool(fl & FL_FAULT))


def encode_tpdo2(f: MarkerFrame) -> bytes:
    def c(v: int) -> int:
        return v if v == NO_DATA else _clamp(v, -32767, 32767)

    return _T2.pack(c(f.left_along_raw), c(f.left_lateral_raw), c(f.right_along_raw), c(f.right_lateral_raw))


def decode_tpdo2(data: bytes) -> MarkerFrame:
    if len(data) != 8:
        raise DecodeError(f"TPDO2 needs 8 bytes, got {len(data)}")
    return MarkerFrame(*_T2.unpack(bytes(data)))


def encode_tpdo3(f: DiagFrame) -> bytes:
    return _T3.pack(f.sample_counter & 0xFFFF, f.error_register & 0xFF)


def decode_tpdo3(data: bytes) -> DiagFrame:
    if len(data) != 3:
        raise DecodeError(f"TPDO3 needs 3 bytes, got {len(data)}")
    return DiagFrame(*_T3.unpack(bytes(data)))


def decode_frame(can_id: int, data: bytes, node_id: int = DEFAULT_NODE_ID) -> Optional[Decoded]:
    """Decode one frame; None for other nodes or unknown function codes.

    Filter rule: `(id & 0x7F) == node_id`, dispatch on `id & 0x780`.
    """
    if can_id > 0x7FF or (can_id & 0x7F) != node_id:
        return None
    fc = can_id & 0x780
    if fc == FC_TPDO1:
        return decode_tpdo1(data)
    if fc == FC_TPDO2:
        return decode_tpdo2(data)
    if fc == FC_TPDO3:
        return decode_tpdo3(data)
    if fc == FC_HEARTBEAT:
        if len(data) != 1:
            raise DecodeError(f"heartbeat needs 1 byte, got {len(data)}")
        return HeartbeatFrame(data[0])
    return None


# ------------------------------------------------------------------ object dictionary
# (index, sub, name, type, bits, access, default). Application objects live in the manufacturer-
# specific area 2000h..2FFFh because no CiA device profile covers tape guidance.
OBJECTS: List[Tuple[int, int, str, str, int, str, str]] = [
    (0x1000, 0, "Device type", "U32", 32, "ro", "0x00000000"),
    (0x1001, 0, "Error register", "U8", 8, "ro", "0"),
    (0x1008, 0, "Manufacturer device name", "VISIBLE_STRING", 0, "const", "TGS-1 tape guide sensor"),
    (0x1017, 0, "Producer heartbeat time (ms)", "U16", 16, "rw", "1000"),
    (0x1018, 1, "Identity: vendor ID", "U32", 32, "ro", "0 (unassigned)"),
    (0x1018, 2, "Identity: product code", "U32", 32, "ro", "0x00000001"),
    (0x1018, 3, "Identity: revision", "U32", 32, "ro", "0x00010000"),
    (0x1800, 1, "TPDO1 COB-ID", "U32", 32, "ro", "$NODEID+0x180"),
    (0x1800, 2, "TPDO1 transmission type", "U8", 8, "ro", "254"),
    (0x1800, 5, "TPDO1 event timer (ms)", "U16", 16, "rw", "10"),
    (0x1801, 1, "TPDO2 COB-ID", "U32", 32, "ro", "$NODEID+0x280"),
    (0x1801, 2, "TPDO2 transmission type", "U8", 8, "ro", "254"),
    (0x1801, 5, "TPDO2 event timer (ms)", "U16", 16, "rw", "20"),
    (0x1802, 1, "TPDO3 COB-ID", "U32", 32, "ro", "$NODEID+0x380"),
    (0x1802, 2, "TPDO3 transmission type", "U8", 8, "ro", "254"),
    (0x1802, 5, "TPDO3 event timer (ms)", "U16", 16, "rw", "50"),
    (0x2000, 1, "Track: offset left (0.1 mm, left +)", "I16", 16, "ro", "-32768 = none"),
    (0x2000, 2, "Track: offset right (0.1 mm, left +)", "I16", 16, "ro", "-32768 = none"),
    (0x2000, 3, "Track: heading left (0.5 deg, ccw +)", "I8", 8, "ro", "0"),
    (0x2000, 4, "Track: heading right (0.5 deg, ccw +)", "I8", 8, "ro", "0"),
    (0x2000, 5, "Track: flags", "U8", 8, "ro", "0"),
    (0x2000, 6, "Track: quality (0.5 %)", "U8", 8, "ro", "0"),
    (0x2001, 1, "Marker: left along (0.05 mm, forward +)", "I16", 16, "ro", "-32768 = none"),
    (0x2001, 2, "Marker: left lateral (0.05 mm, left +)", "I16", 16, "ro", "-32768 = none"),
    (0x2001, 3, "Marker: right along (0.05 mm)", "I16", 16, "ro", "-32768 = none"),
    (0x2001, 4, "Marker: right lateral (0.05 mm)", "I16", 16, "ro", "-32768 = none"),
    (0x2002, 0, "Sample counter", "U16", 16, "ro", "0"),
    (0x2100, 0, "Sensing width (mm)", "U16", 16, "ro", "150"),
    (0x2101, 0, "Invert offset sign", "U8", 8, "rw", "0"),
    (0x2102, 0, "Invert heading sign", "U8", 8, "rw", "0"),
]

# PDO mapping: lists of (index, sub, bits), in wire order.
PDO_MAPPING: Dict[str, List[Tuple[int, int, int]]] = {
    "tpdo1": [(0x2000, 1, 16), (0x2000, 2, 16), (0x2000, 3, 8), (0x2000, 4, 8), (0x2000, 5, 8), (0x2000, 6, 8)],
    "tpdo2": [(0x2001, 1, 16), (0x2001, 2, 16), (0x2001, 3, 16), (0x2001, 4, 16)],
    "tpdo3": [(0x2002, 0, 16), (0x1001, 0, 8)],
}
PDO_DLC = {"tpdo1": 8, "tpdo2": 8, "tpdo3": 3}
PDO_PERIOD_MS = {"tpdo1": 10, "tpdo2": 20, "tpdo3": 50}


def mapping_entry(index: int, sub: int, bits: int) -> int:
    """CiA 301 PDO mapping parameter value: index << 16 | sub << 8 | length in bits."""
    return (index << 16) | (sub << 8) | bits


def eds_text() -> str:
    """The object dictionary as an EDS-style INI text (written to docs/tape-guide-sensor.eds)."""
    types = {"U8": "0x0005", "U16": "0x0006", "U32": "0x0007", "I8": "0x0002", "I16": "0x0003",
             "VISIBLE_STRING": "0x0009"}
    lines = ["[FileInfo]", "FileName=tape-guide-sensor.eds", "FileVersion=1",
             "Description=Open tape-guide sensor profile", "", "[DeviceInfo]",
             "VendorName=open profile (no vendor)", "ProductName=TGS-1 tape guide sensor",
             "NrOfRXPDO=0", "NrOfTXPDO=3", ""]

    def var(sect: str, name: str, typ: str, acc: str, default: str, mappable: bool) -> None:
        lines.extend([f"[{sect}]", f"ParameterName={name}", "ObjectType=0x7", f"DataType={types[typ]}",
                      f"AccessType={acc}", f"DefaultValue={default}", f"PDOMapping={int(mappable)}", ""])

    indices = sorted({o[0] for o in OBJECTS})
    for idx in indices:
        objs = [o for o in OBJECTS if o[0] == idx]
        mappable = idx in (0x2000, 0x2001, 0x2002, 0x1001)
        if len(objs) == 1 and objs[0][1] == 0:
            _, _, name, typ, _, acc, default = objs[0]
            var(f"{idx:X}", name, typ, acc, default, mappable)
            continue
        lines.extend([f"[{idx:X}]", f"ParameterName=Record {idx:X}h", "ObjectType=0x9",
                      f"SubNumber={len(objs) + 1}", ""])
        var(f"{idx:X}sub0", "Highest sub-index supported", "U8", "const", str(max(o[1] for o in objs)), False)
        for _, sub, name, typ, _, acc, default in objs:
            var(f"{idx:X}sub{sub:X}", name, typ, acc, default, mappable)
    for k, entries in enumerate(PDO_MAPPING.values()):
        idx = 0x1A00 + k
        lines.extend([f"[{idx:X}]", f"ParameterName=TPDO{k + 1} mapping", "ObjectType=0x9",
                      f"SubNumber={len(entries) + 1}", ""])
        var(f"{idx:X}sub0", "Number of mapped objects", "U8", "ro", str(len(entries)), False)
        for j, e in enumerate(entries, start=1):
            var(f"{idx:X}sub{j:X}", f"Mapped object {j}", "U32", "ro", f"0x{mapping_entry(*e):08X}", False)
    return "\n".join(lines)
