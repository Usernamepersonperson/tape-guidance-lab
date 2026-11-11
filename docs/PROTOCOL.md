# Open tape-guide sensor profile (TGS profile, version 1)

An open CANopen profile for a magnetic-tape guide sensor on an AGV: where the tape is relative to
the sensor, whether the track forks or merges, and where position markers lie. It follows general
CANopen practice (CiA 301: 11-bit COB-IDs of function code + node ID, little-endian PDOs, event-timer
TPDOs, heartbeat producer, an object dictionary with PDO mapping). It is **a design of this
repository, not a standard and not the layout of any existing product**. No CiA device profile
covers tape guidance, so the application objects sit in the manufacturer-specific area `2000h..2FFFh`.

Reference implementation: `agvlab/frames.py` (encoder, decoder, object dictionary). Sensor
simulator: `agvlab/sensor.py`. The object table below is the one in `frames.OBJECTS`; the tests
check that the PDO mappings add up to each frame's length. `python -m agvlab eds` writes the same
table as `docs/tape-guide-sensor.eds` (an EDS-style INI file; not validated with a CANopen tool).

## Conventions

| Quantity | Convention |
|---|---|
| Frame | CAN 2.0A, 11-bit identifier, classic 8-byte data, 125 kbit/s to 1 Mbit/s |
| Byte order | little-endian, two's complement |
| Node ID | 1..127, default **21** |
| Lateral offset | signed, **left positive** (ROS REP-103 y axis), measured at the sensing line, relative to the sensor centre |
| Heading | signed, **counter-clockwise positive**, tape direction relative to the sensor's forward axis |
| Along (markers) | forward of the sensing line positive |
| No data | `0x8000` (-32768) in an `int16` field |

## COB-IDs

| Message | COB-ID | DLC | Default period | Direction |
|---|---|---|---|---|
| TPDO1 Track | `0x180 + node` | 8 | 10 ms | sensor -> host |
| TPDO2 Markers | `0x280 + node` | 8 | 20 ms | sensor -> host |
| TPDO3 Diagnostics | `0x380 + node` | 3 | 50 ms | sensor -> host |
| Heartbeat | `0x700 + node` | 1 | 1000 ms | sensor -> host |
| NMT command (start/stop) | `0x000` | 2 | on demand | host -> sensor (documented; the simulator is always operational) |

With node 21 the IDs are `0x195`, `0x295`, `0x395`, `0x715`. A receiver accepts a frame when
`(id & 0x7F) == node` and dispatches on `id & 0x780`.

## TPDO1 Track (8 bytes)

| Byte | Field | Type | Scale | Notes |
|---|---|---|---|---|
| 0-1 | offset_left | int16 | 0.1 mm | leftmost tape under the sensing line; `0x8000` = no tape |
| 2-3 | offset_right | int16 | 0.1 mm | rightmost tape; equal to offset_left on a single tape |
| 4 | heading_left | int8 | 0.5 deg | range +/-63.5 deg |
| 5 | heading_right | int8 | 0.5 deg | |
| 6 | flags | uint8 | bit field | below |
| 7 | quality | uint8 | 0.5 % | 0..200 = 0..100 % signal quality, 0 when no tape |

Flags (byte 6):

| Bit | Name | Meaning |
|---|---|---|
| 0 | track | a tape is detected (offsets and headings are valid) |
| 1 | two_tracks | two separate tapes are under the sensing line (otherwise both slots carry the same values) |
| 2 | fork | the two tracks lie within one tape width and diverge (heading difference 10..45 deg, left minus right) |
| 3 | merge | the two tracks lie within one tape width and converge (difference -45..-10 deg) |
| 4 | junction | a tape crosses the sensing line at a steep angle (more than 60 deg); headings are reported as 0 |
| 5 | marker_left | a marker is under the sensing window, left of the tape; its position is in TPDO2 |
| 6 | marker_right | same, right of the tape |
| 7 | fault | sensor fault (reserved; see error register in TPDO3) |

A follower picks the left or the right slot to choose the branch at a fork.

## TPDO2 Markers (8 bytes)

Four `int16` fields at 0.05 mm per bit, in this order: `left_along`, `left_lateral`, `right_along`,
`right_lateral`. An empty slot is `0x8000` in both of its fields. A marker seen with no tape under
the sensor is reported in both slots. `along` is the distance of the marker ahead of the sensing line
(negative once passed); a stop controller drives it to zero.

## TPDO3 Diagnostics (3 bytes)

| Byte | Field | Type | Notes |
|---|---|---|---|
| 0-1 | sample_counter | uint16 | counts TPDO1 transmissions, wraps at 65536: a receiver detects lost TPDO1 frames from gaps |
| 2 | error_register | uint8 | object 1001h; 0 = no error |

## Heartbeat

One byte, NMT state: `0x00` boot-up, `0x04` stopped, `0x05` operational, `0x7F` pre-operational.
Producer time is object 1017h (default 1000 ms). A consumer should flag the sensor lost after about
three missed heartbeats (the ROS driver does so at 3 s).

## Object dictionary (EDS-like)

| Index | Sub | Name | Type | Access | Default |
|---|---|---|---|---|---|
| 1000h | 0 | Device type | U32 | ro | 0x00000000 (no device profile) |
| 1001h | 0 | Error register | U8 | ro | 0 |
| 1008h | 0 | Manufacturer device name | VISIBLE_STRING | const | `TGS-1 tape guide sensor` |
| 1017h | 0 | Producer heartbeat time (ms) | U16 | rw | 1000 |
| 1018h | 1 | Identity: vendor ID | U32 | ro | 0 (unassigned) |
| 1018h | 2 | Identity: product code | U32 | ro | 0x00000001 |
| 1018h | 3 | Identity: revision | U32 | ro | 0x00010000 |
| 1800h | 1 | TPDO1 COB-ID | U32 | ro | $NODEID+0x180 |
| 1800h | 2 | TPDO1 transmission type | U8 | ro | 254 (event driven, timer) |
| 1800h | 5 | TPDO1 event timer (ms) | U16 | rw | 10 |
| 1801h | 1 | TPDO2 COB-ID | U32 | ro | $NODEID+0x280 |
| 1801h | 2 | TPDO2 transmission type | U8 | ro | 254 |
| 1801h | 5 | TPDO2 event timer (ms) | U16 | rw | 20 |
| 1802h | 1 | TPDO3 COB-ID | U32 | ro | $NODEID+0x380 |
| 1802h | 2 | TPDO3 transmission type | U8 | ro | 254 |
| 1802h | 5 | TPDO3 event timer (ms) | U16 | rw | 50 |
| 1A00h | 1..6 | TPDO1 mapping | U32 | ro | `2000h:1..6` (see below) |
| 1A01h | 1..4 | TPDO2 mapping | U32 | ro | `2001h:1..4` |
| 1A02h | 1..2 | TPDO3 mapping | U32 | ro | `2002h:0`, `1001h:0` |
| 2000h | 1 | Track: offset left (0.1 mm, left +) | I16 | ro | -32768 = none |
| 2000h | 2 | Track: offset right | I16 | ro | -32768 = none |
| 2000h | 3 | Track: heading left (0.5 deg, ccw +) | I8 | ro | 0 |
| 2000h | 4 | Track: heading right | I8 | ro | 0 |
| 2000h | 5 | Track: flags | U8 | ro | 0 |
| 2000h | 6 | Track: quality (0.5 %) | U8 | ro | 0 |
| 2001h | 1 | Marker: left along (0.05 mm, forward +) | I16 | ro | -32768 = none |
| 2001h | 2 | Marker: left lateral (0.05 mm, left +) | I16 | ro | -32768 = none |
| 2001h | 3 | Marker: right along | I16 | ro | -32768 = none |
| 2001h | 4 | Marker: right lateral | I16 | ro | -32768 = none |
| 2002h | 0 | Sample counter | U16 | ro | 0 |
| 2100h | 0 | Sensing width (mm) | U16 | ro | 150 |
| 2101h | 0 | Invert offset sign (mounting) | U8 | rw | 0 |
| 2102h | 0 | Invert heading sign (mounting) | U8 | rw | 0 |

PDO mapping entries are `index << 16 | sub << 8 | length_in_bits`; TPDO1 maps
`0x20000110, 0x20000210, 0x20000308, 0x20000408, 0x20000508, 0x20000608`.

Objects 2100h..2102h are listed for completeness; the simulator takes the same settings from
`SensorConfig` and does not implement SDO access.

## Worked example

Two tapes under the sensor just after a fork: the left one 8.0 mm left of the sensor centre, the right
one 12.3 mm right of it, headings +1.0 deg and -2.5 deg, full signal quality:

```
0x195  50 00 85 FF 02 FB 07 C8
       |     |     |  |  |  quality 200 = 100 %
       |     |     |  |  flags 0x07 = track | two_tracks | fork
       |     |     |  heading_right -2.5 deg (0xFB = -5)
       |     |     heading_left +1.0 deg
       |     offset_right -12.3 mm (0xFF85 = -123)
       offset_left +8.0 mm (0x0050 = 80)
```

`tests/test_frames.py` pins this vector.

## Not implemented / not verified

- SDO server (reading or writing the dictionary), NMT command handling, SYNC and EMCY messages are
  documented as intended practice but **not** implemented by the simulator.
- No hardware implements this profile. The scalings are chosen for a 150 mm sensing width and 25 mm
  tape; they have not been checked against any real sensor array.
- Fork and merge flag semantics are defined here; a follower should not rely on them (the lab's follower
  chooses a branch from the left or right track).
