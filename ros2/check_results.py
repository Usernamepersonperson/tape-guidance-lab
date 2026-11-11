#!/usr/bin/env python3
"""Assert the ROS closed-loop results (written by sim_node.py and follower_node.py) are healthy."""

import json
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/agvlab"
sim = json.load(open(f"{d}/sim.json"))
fol = json.load(open(f"{d}/follower.json"))
print(json.dumps({"sim": sim, "follower": fol}, indent=1))

checks = {
    "ran to the stop (no timeout)": not sim["timeout"],
    "right branch taken": sim["junction_ok"] == 1.0,
    "tracking RMS < 10 mm": sim["rms_mm"] is not None and sim["rms_mm"] < 10.0,
    "stopped within 40 mm of the stop marker": sim["stop_error_mm"] is not None
    and abs(sim["stop_error_mm"]) < 40.0,
    "driver /guide/track rate 80..120 Hz": 80.0 <= fol["track_rate_hz"] <= 120.0,
    "follower saw both markers": fol["markers_seen"] == 2,
}
bad = [k for k, ok in checks.items() if not ok]
for k, ok in checks.items():
    print(("PASS " if ok else "FAIL ") + k)
sys.exit(1 if bad else 0)
