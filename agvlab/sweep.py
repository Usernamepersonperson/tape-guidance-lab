"""Parameter sweeps: noise, latency, frame loss, tape-lost bursts, speed. Shows where guidance breaks."""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .lab import RunConfig, run_closed_loop

ROUTES = ("branch", "straight")  # seeds alternate so both junction choices are exercised


def _noise(cfg: RunConfig, k: float) -> RunConfig:
    s = replace(cfg.sensor, pos_sigma_mm=0.3 * k, angle_sigma_deg=1.0 * k, marker_sigma_mm=0.1 * k)
    return replace(cfg, sensor=s)


def _latency(cfg: RunConfig, ms: float) -> RunConfig:
    return replace(cfg, channel=replace(cfg.channel, latency_ms=ms))


def _drop(cfg: RunConfig, p: float) -> RunConfig:
    return replace(cfg, channel=replace(cfg.channel, drop_prob=p))


def _burst(cfg: RunConfig, ms: float) -> RunConfig:
    # one tape-lost burst every ~2 s on average, `ms` long
    s = replace(cfg.sensor, burst_hz=0.5 if ms > 0 else 0.0, burst_ms=ms)
    return replace(cfg, sensor=s)


def _speed(cfg: RunConfig, v: float) -> RunConfig:
    return replace(cfg, follower=replace(cfg.follower, v_cruise=v))


SWEEPS: Dict[str, dict] = {
    "noise": dict(label="sensor noise (x nominal 0.3 mm / 1 deg)", values=[0, 1, 3, 5, 10, 20, 30, 40],
                  apply=_noise),
    "latency": dict(label="one-way latency (ms)", values=[0, 20, 50, 100, 150, 200, 300, 400],
                    apply=_latency),
    "drop": dict(label="frame loss probability", values=[0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95],
                 apply=_drop),
    "burst": dict(label="tape-lost burst length (ms), 0.5 Hz", values=[0, 50, 100, 150, 200, 300, 500],
                  apply=_burst),
    "speed": dict(label="cruise speed (m/s)", values=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0], apply=_speed),
}
QUICK_VALUES = {"noise": [1, 10], "latency": [0, 200], "drop": [0.0, 0.5], "burst": [0, 200],
                "speed": [0.3, 1.0]}


def make_config(name: str, value: float, seed: int, base: Optional[RunConfig] = None) -> RunConfig:
    base = base or RunConfig()
    cfg = SWEEPS[name]["apply"](base, value)
    return replace(cfg, seed=seed, route=ROUTES[seed % 2])


def _run_one(args) -> Dict[str, float]:
    name, value, seed = args
    return run_closed_loop(make_config(name, value, seed)).metrics


def _finite(xs: Sequence[float]) -> List[float]:
    return [x for x in xs if x is not None and math.isfinite(x)]


def aggregate(value: float, runs: List[Dict[str, float]]) -> Dict[str, float]:
    rms = _finite([r["rms_mm"] for r in runs])
    stop = _finite([abs(r["stop_error_mm"]) for r in runs])
    return {
        "value": value,
        "n": len(runs),
        "success_rate": float(np.mean([r["success"] for r in runs])),
        "junction_rate": float(np.mean([r["junction_ok"] for r in runs])),
        "stopped_rate": float(np.mean([r["stopped"] for r in runs])),
        "derailed_rate": float(np.mean([r["derailed"] for r in runs])),
        "rms_mm_median": float(np.median(rms)) if rms else math.nan,
        "rms_mm_worst": float(np.max(rms)) if rms else math.nan,
        "max_mm_worst": float(np.max(_finite([r["max_mm"] for r in runs]) or [math.nan])),
        "stop_abs_mm_median": float(np.median(stop)) if stop else math.nan,
        "stop_abs_mm_worst": float(np.max(stop)) if stop else math.nan,
        "tpdo1_rate_hz": float(np.mean([r["tpdo1_rate_hz"] for r in runs])),
    }


def run_sweep(name: str, values: Optional[Sequence[float]] = None, seeds: int = 6,
              workers: Optional[int] = None, progress: Optional[Callable[[str], None]] = None) -> List[dict]:
    values = list(values if values is not None else SWEEPS[name]["values"])
    jobs = [(name, v, s) for v in values for s in range(seeds)]
    workers = workers if workers is not None else max(1, min(os.cpu_count() or 1, 8))
    if workers == 1:
        results = [_run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_run_one, jobs, chunksize=1))
    rows = []
    for i, v in enumerate(values):
        rows.append(aggregate(v, results[i * seeds:(i + 1) * seeds]))
        if progress:
            progress(f"{name}={v}: success {rows[-1]['success_rate']:.0%}, "
                     f"rms median {rows[-1]['rms_mm_median']:.1f} mm")
    return rows


def find_break(rows: List[dict], rms_limit_mm: float = 5.0) -> Dict[str, Optional[float]]:
    """First sweep value where (a) any run fails, (b) median RMS exceeds the limit."""
    first_fail = next((r["value"] for r in rows if r["success_rate"] < 1.0), None)
    first_rms = next((r["value"] for r in rows if r["rms_mm_median"] > rms_limit_mm), None)
    return {"first_failure": first_fail, "first_rms_over_limit": first_rms}
