"""Command line: run one scenario, sweep, candump-style frame log, real-time bus run, figures."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from typing import List, Optional

from . import __version__
from .follower import FollowerConfig
from .lab import RunConfig, run_closed_loop
from .sweep import QUICK_VALUES, SWEEPS, find_break, run_sweep


def _cfg_from_args(a: argparse.Namespace) -> RunConfig:
    base = RunConfig(route=a.route, seed=a.seed)
    sensor = replace(base.sensor, pos_sigma_mm=0.3 * a.noise_x, angle_sigma_deg=1.0 * a.noise_x,
                     marker_sigma_mm=0.1 * a.noise_x, burst_hz=a.burst_hz, burst_ms=a.burst_ms)
    channel = replace(base.channel, latency_ms=a.latency_ms, jitter_ms=a.jitter_ms, drop_prob=a.drop)
    if a.controller == "p_only":
        follower = FollowerConfig.baseline()
    else:
        follower = FollowerConfig(controller=a.controller, v_cruise=a.speed)
    return replace(base, sensor=sensor, channel=channel, follower=follower)


def _add_scenario_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--route", choices=["branch", "straight"], default="branch", help="junction choice")
    p.add_argument("--controller", choices=["pd", "pure_pursuit", "p_only"], default="pd")
    p.add_argument("--speed", type=float, default=0.30, help="cruise speed m/s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-x", type=float, default=1.0, help="multiples of nominal 0.3 mm / 1 deg noise")
    p.add_argument("--latency-ms", type=float, default=2.0)
    p.add_argument("--jitter-ms", type=float, default=1.0)
    p.add_argument("--drop", type=float, default=0.0, help="frame loss probability")
    p.add_argument("--burst-hz", type=float, default=0.0, help="tape-lost bursts per second")
    p.add_argument("--burst-ms", type=float, default=80.0)


def cmd_run(a: argparse.Namespace) -> int:
    res = run_closed_loop(_cfg_from_args(a))
    m = res.metrics
    print(f"route {a.route}, controller {a.controller}, {m['duration_s']:.1f} s simulated")
    print(f"  tracking RMS error    {m['rms_mm']:.2f} mm   (max {m['max_mm']:.1f} mm)")
    print(f"  junction choice       {'correct' if m['junction_ok'] else 'WRONG'}")
    if math.isfinite(m["stop_error_mm"]):
        print(f"  stop error at marker  {m['stop_error_mm']:+.1f} mm (+ = short of marker)")
    else:
        print("  stop error at marker  not stopped")
    print(f"  TPDO1 rate achieved   {m['tpdo1_rate_hz']:.1f} Hz   frames dropped {int(m['frames_dropped'])}")
    return 0 if m["success"] else 1


def cmd_candump(a: argparse.Namespace) -> int:
    """Print the simulator's frames for a straight-ahead drive in `candump -l` style."""
    res = run_closed_loop(replace(_cfg_from_args(a), max_time_s=a.duration), keep_frames=True)
    for t, cid, data in res.frames_log:
        if t > a.duration:
            break
        print(f"({t:.6f}) {a.iface} {cid:03X}#{data.hex().upper()}")
    return 0


def cmd_sweep(a: argparse.Namespace) -> int:
    names = a.names or list(SWEEPS)
    unknown = [n for n in names if n not in SWEEPS]
    if unknown:
        print(f"unknown sweep(s): {', '.join(unknown)}; choose from {', '.join(SWEEPS)}", file=sys.stderr)
        return 2
    out = {}
    for n in names:
        vals = QUICK_VALUES[n] if a.quick else None
        rows = run_sweep(n, vals, seeds=a.seeds, workers=a.workers, progress=print)
        out[n] = {"label": SWEEPS[n]["label"], "rows": rows, "break": find_break(rows)}
        print(f"{n}: break {out[n]['break']}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"version": __version__, "seeds": a.seeds, "sweeps": out}, fh, indent=1)
    return 0


def cmd_figure(a: argparse.Namespace) -> int:
    from .plots import make_figure, make_sweep_figure
    metrics = make_figure(a.out)
    if a.post:
        from .plots import make_post_figure
        make_post_figure(a.post)
        print(f"wrote {a.post}")
    print(f"wrote {a.out}")
    for k, m in metrics.items():
        print(f"  {k}: rms {m['rms_mm']:.2f} mm max {m['max_mm']:.1f} mm stop {m['stop_error_mm']:+.1f} mm "
              f"junction_ok {bool(m['junction_ok'])}")
    if a.sweeps:
        with open(a.sweeps, encoding="utf-8") as fh:
            data = json.load(fh)["sweeps"]
        make_sweep_figure(a.sweep_out, {k: v["rows"] for k, v in data.items()})
        print(f"wrote {a.sweep_out}")
    return 0


def cmd_eds(a: argparse.Namespace) -> int:
    from .frames import eds_text
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(eds_text())
    print(f"wrote {a.out}")
    return 0


def cmd_live(a: argparse.Namespace) -> int:
    from .live import open_pair, run_live
    tx, rx = open_pair(a.interface, a.channel)
    try:
        res = run_live(tx, rx, replace(_cfg_from_args(a), dt=0.005), duration_s=a.duration)
    finally:
        tx.shutdown()
        rx.shutdown()
    m = res.metrics
    print(f"{a.interface}/{a.channel}: {res.n_tpdo1} TPDO1 frames in {m['duration_s']:.2f} s "
          f"= {m['tpdo1_rate_hz']:.1f} Hz; interval mean {res.rx_interval_ms['mean']:.2f} ms, "
          f"sd {res.rx_interval_ms['sd']:.2f}, p99 {res.rx_interval_ms['p99']:.2f}, "
          f"max {res.rx_interval_ms['max']:.2f}")
    print(f"tracking RMS {m['rms_mm']:.2f} mm, max {m['max_mm']:.1f} mm")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="agvlab", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="one closed-loop scenario in simulated time")
    _add_scenario_args(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("candump", help="print simulated sensor frames, candump -l style")
    _add_scenario_args(p)
    p.add_argument("--duration", type=float, default=0.5)
    p.add_argument("--iface", default="vcan0")
    p.set_defaults(fn=cmd_candump)

    p = sub.add_parser("sweep", help="noise / latency / loss / burst / speed sweeps")
    p.add_argument("names", nargs="*", help="any of: " + ", ".join(SWEEPS) + " (default: all)")
    p.add_argument("--seeds", type=int, default=6)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="two values per sweep (CI smoke)")
    p.add_argument("--out", default="docs/results.json")
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("figure", help="render docs/figure.png (+ sweep figure)")
    p.add_argument("--out", default="docs/figure.png")
    p.add_argument("--post", default=None, help="also write the 960x700 LinkedIn image to this path")
    p.add_argument("--sweeps", default=None, help="results.json to plot")
    p.add_argument("--sweep-out", default="docs/sweep.png")
    p.set_defaults(fn=cmd_figure)

    p = sub.add_parser("eds", help="write the sensor profile's object dictionary as an EDS-style file")
    p.add_argument("--out", default="docs/tape-guide-sensor.eds")
    p.set_defaults(fn=cmd_eds)

    p = sub.add_parser("live", help="real-time run over a python-can bus (virtual, socketcan/vcan0, ...)")
    _add_scenario_args(p)
    p.add_argument("--interface", default="virtual")
    p.add_argument("--channel", default="agvlab")
    p.add_argument("--duration", type=float, default=5.0)
    p.set_defaults(fn=cmd_live)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
