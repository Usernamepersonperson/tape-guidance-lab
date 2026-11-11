"""Figures: layout + driven path + lateral error over time, and the parameter sweeps."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .lab import RunConfig, RunResult, run_closed_loop  # noqa: E402
from .layout import nearest_on_polyline  # noqa: E402
from .sweep import SWEEPS  # noqa: E402

C_BRANCH = "#1f77b4"
C_STRAIGHT = "#ff7f0e"
C_HARSH = "#d62728"


def _event_times(res: RunResult) -> Dict[str, float]:
    """Times at which the sensor point passes the fork and the two markers (from ground truth)."""
    route = res.layout.routes[res.route]
    ev = {}
    fork = res.layout.fork_s
    marks = {m.role: m for m in res.layout.markers if m.route == res.route}
    targets = {"fork": fork}
    for role, m in marks.items():
        targets[role] = nearest_on_polyline(route, np.array([m.x, m.y]))[2]
    th = res.th
    for i in range(len(res.t)):
        p = np.array([res.x[i] + 0.25 * np.cos(th[i]), res.y[i] + 0.25 * np.sin(th[i])])
        s = nearest_on_polyline(route, p)[2]
        for k, tgt in list(targets.items()):
            if s >= tgt:
                ev[k] = float(res.t[i])
                del targets[k]
        if not targets:
            break
    return ev


def make_figure(path: str, runs: Optional[Dict[str, RunResult]] = None) -> Dict[str, dict]:
    """Layout with driven paths (colour = |error|) and error/speed over time. Returns the run metrics."""
    if runs is None:
        base = RunConfig()
        harsh = replace(
            base, route="branch",
            sensor=replace(base.sensor, pos_sigma_mm=3.0, angle_sigma_deg=10.0),
            channel=replace(base.channel, latency_ms=100.0, drop_prob=0.2))
        runs = {
            "branch": run_closed_loop(replace(base, route="branch")),
            "straight": run_closed_loop(replace(base, route="straight", seed=1)),
            "harsh": run_closed_loop(harsh),
        }
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13})
    fig = plt.figure(figsize=(16, 9), dpi=100)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.35, 1.0],
                          left=0.05, right=0.985, top=0.93, bottom=0.085, wspace=0.17, hspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    ax_e = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1], sharex=ax_e)

    layout = runs["branch"].layout
    for tape in layout.tapes:
        ax.plot(tape[:, 0], tape[:, 1], color="#555555", lw=7, alpha=0.35, solid_capstyle="butt", zorder=1)
    ax.plot([], [], color="#555555", lw=7, alpha=0.35, label="magnetic tape")
    for name, col in (("branch", C_BRANCH), ("straight", C_STRAIGHT)):
        r = runs[name]
        pts = np.column_stack([r.x + 0.25 * np.cos(r.th), r.y + 0.25 * np.sin(r.th)])
        ax.plot(pts[:, 0], pts[:, 1], color=col, lw=1.8, zorder=3,
                label=f"driven, choose {name} (RMS {r.metrics['rms_mm']:.1f} mm)")
    r = runs["harsh"]
    pts = np.column_stack([r.x + 0.25 * np.cos(r.th), r.y + 0.25 * np.sin(r.th)])
    ax.plot(pts[:, 0], pts[:, 1], color=C_HARSH, lw=1.2, ls="--", zorder=2,
            label=f"harsh run, branch (RMS {r.metrics['rms_mm']:.1f} mm)")
    for m in layout.markers:
        ax.scatter([m.x], [m.y], marker="s" if m.role == "stop" else "D", s=70 if m.role == "stop" else 45,
                   color="#c00000" if m.role == "stop" else "#e0a000", zorder=4)
    ax.scatter([], [], marker="D", color="#e0a000", s=45, label="slow marker")
    ax.scatter([], [], marker="s", color="#c00000", s=70, label="stop marker")
    sx, sy, _ = layout.start_pose
    ax.annotate("start", (sx, sy), xytext=(sx + 0.1, sy + 0.45), arrowprops=dict(arrowstyle="->"))
    fp = layout.post_fork["straight"][0]
    ax.annotate("fork", fp, xytext=(fp[0] - 1.3, fp[1] - 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 5.6)
    ax.set_ylim(-0.7, 6.7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Tape layout and driven path (sensor centre)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)

    for name, col, lw, ls in (("branch", C_BRANCH, 1.6, "-"), ("straight", C_STRAIGHT, 1.6, "-"),
                              ("harsh", C_HARSH, 1.0, "--")):
        r = runs[name]
        label = name if name != "harsh" else "harsh (noise x10, 100 ms, 20% loss)"
        ax_e.plot(r.t, r.err_mm, color=col, lw=lw, ls=ls, label=label)
        ax_v.plot(r.t, r.v, color=col, lw=lw, ls=ls)
    ev = _event_times(runs["branch"])
    ax_e.set_ylim(-42, 42)
    for k, text in (("fork", "fork"), ("slow", "slow marker"), ("stop", "stop marker")):
        if k in ev:
            for a in (ax_e, ax_v):
                a.axvline(ev[k], color="k", ls=":", lw=1)
            low = k == "stop"  # the slow and stop labels would collide on one row
            ax_e.text(ev[k] - 0.25 if low else ev[k] + 0.25, -38 if low else 31, text, fontsize=11,
                      va="center", ha="right" if low else "left")
    ax_e.axhline(0, color="k", lw=0.6)
    ax_e.set_ylabel("lateral error (mm)")
    ax_e.set_title("Cross-track error of the sensor centre vs the intended route (ground truth)")
    ax_e.grid(alpha=0.25)
    ax_e.legend(loc="lower left", fontsize=10.5)
    m = runs["branch"].metrics
    ax_e.text(0.01, 0.97, f"nominal branch run: RMS {m['rms_mm']:.2f} mm, max {m['max_mm']:.1f} mm "
              f"(fork), stop {m['stop_error_mm']:+.1f} mm", transform=ax_e.transAxes, fontsize=11, va="top",
              bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax_v.set_ylabel("speed (m/s)")
    ax_v.set_xlabel("time (s)")
    ax_v.grid(alpha=0.25)
    ax_v.set_title("Speed: cruise, creep after the slow marker, stop at the stop marker")
    fig.suptitle("tape-guidance-lab: simulated guide-sensor CAN frames -> decoder -> follower -> diff-drive "
                 "(simulated; assumed 2 ms latency, 1 ms jitter, 0.3 mm noise)", fontsize=14, y=0.985)
    fig.savefig(path)
    plt.close(fig)
    return {k: v.metrics for k, v in runs.items()}


def make_sweep_figure(path: str, sweeps: Dict[str, List[dict]], rms_limit_mm: float = 5.0) -> None:
    n_seeds = next(iter(sweeps.values()))[0]["n"]
    plt.rcParams.update({"font.size": 12})
    names = [n for n in SWEEPS if n in sweeps]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=100)
    for ax, n in zip(axes.flat, names, strict=False):
        rows = sweeps[n]
        x = [r["value"] for r in rows]
        rms = [r["rms_mm_median"] for r in rows]
        worst = [r["rms_mm_worst"] for r in rows]
        ax.plot(x, rms, "o-", color=C_BRANCH, label="RMS error, median")
        ax.plot(x, worst, "o--", color=C_BRANCH, alpha=0.5, label="RMS error, worst run")
        ax.axhline(rms_limit_mm, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_ylabel("lateral RMS (mm, log)")
        ax.set_xlabel(SWEEPS[n]["label"])
        ax2 = ax.twinx()
        ax2.plot(x, [100 * r["success_rate"] for r in rows], "s-", color=C_HARSH, label="success rate")
        ax2.set_ylim(-5, 105)
        ax2.set_ylabel("runs succeeded (%)")
        ax.grid(alpha=0.25)
        ax.set_title(n)
        if n == names[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9.5)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    fig.suptitle(f"Where it breaks: {n_seeds} seeds per point, success = right branch and stopped within 20 mm",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ LinkedIn image
P_BG, P_GRID, P_TEXT, P_MUTED = "#16213A", "#2A3750", "#F8FAFC", "#A9B6CB"
P_ACCENT, P_GREEN, P_AMBER, P_RED, P_GREY, P_VIOLET = "#38BDF8", "#34D399", "#FBBF24", "#F87171", "#64748B", "#A78BFA"


def make_post_figure(path: str, runs: Optional[Dict[str, RunResult]] = None) -> Dict[str, dict]:
    """960x700 dark image: tape layout and the driven paths, with a large legend. Returns run metrics."""
    if runs is None:
        base = RunConfig()
        harsh = replace(base, route="branch",
                        sensor=replace(base.sensor, pos_sigma_mm=3.0, angle_sigma_deg=10.0),
                        channel=replace(base.channel, latency_ms=100.0, drop_prob=0.2))
        runs = {"branch": run_closed_loop(replace(base, route="branch")),
                "straight": run_closed_loop(replace(base, route="straight", seed=1)),
                "harsh": run_closed_loop(harsh)}
    with plt.rc_context({"font.family": ["Segoe UI", "DejaVu Sans"], "text.color": P_TEXT}):
        fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=P_BG)
        ax = fig.add_axes([0.05, 0.07, 0.43, 0.76], facecolor=P_BG)
        layout = runs["branch"].layout
        for tape in layout.tapes:
            ax.plot(tape[:, 0], tape[:, 1], color=P_GREY, lw=9, alpha=0.8, solid_capstyle="butt", zorder=1)
        for name, col, lw, ls in (("harsh", P_RED, 1.6, "--"), ("straight", P_AMBER, 2.4, "-"),
                                  ("branch", P_ACCENT, 2.4, "-")):
            r = runs[name]
            pts = np.column_stack([r.x + 0.25 * np.cos(r.th), r.y + 0.25 * np.sin(r.th)])
            ax.plot(pts[:, 0], pts[:, 1], color=col, lw=lw, ls=ls, zorder=3)
        for m in layout.markers:
            ax.scatter([m.x], [m.y], marker="s" if m.role == "stop" else "D", s=110 if m.role == "stop" else 70,
                       color=P_GREEN if m.role == "stop" else P_VIOLET, zorder=4, edgecolors=P_BG)
        ax.set_aspect("equal")
        ax.set_xlim(-0.3, 5.6)
        ax.set_ylim(-0.7, 6.7)
        ax.grid(color=P_GRID, lw=0.8)
        ax.tick_params(colors=P_MUTED, labelsize=12)
        for sp in ax.spines.values():
            sp.set_color(P_GRID)
        ax.set_xlabel("x (m)", color=P_MUTED, fontsize=12)
        ax.set_ylabel("y (m)", color=P_MUTED, fontsize=12)
        mb, ms, mh = runs["branch"].metrics, runs["straight"].metrics, runs["harsh"].metrics
        fig.text(0.05, 0.945, "Tape-following AGV: tape layout and the path it drove", fontsize=19,
                 fontweight="bold", va="center")
        fig.text(0.05, 0.895, "Simulated sensor, controller and robot. Tape 25 mm, cruise 0.3 m/s.",
                 fontsize=13, color=P_MUTED, va="center")
        y = 0.80
        step = 0.098
        entries = [
            ("line", P_GREY, 9, "-", "Magnetic tape", "fork, curve, straights"),
            ("line", P_ACCENT, 3, "-", "Drove the branch",
             f"RMS {mb['rms_mm']:.1f} mm, stop {mb['stop_error_mm']:+.1f} mm"),
            ("line", P_AMBER, 3, "-", "Drove the straight",
             f"RMS {ms['rms_mm']:.1f} mm, stop {ms['stop_error_mm']:+.1f} mm"),
            ("line", P_RED, 2, "--", "Harsh run (noise x10,",
             f"100 ms delay, 20% loss): RMS {mh['rms_mm']:.0f} mm"),
            ("D", P_VIOLET, 8, "", "Slow marker", "drops to creep speed"),
            ("s", P_GREEN, 10, "", "Stop marker", "brake to the mark"),
        ]
        for kind, col, size, ls, head, sub in entries:
            lax = fig.add_axes([0.52, y - 0.025, 0.06, 0.05], facecolor=P_BG)
            lax.axis("off")
            lax.set_xlim(0, 1)
            lax.set_ylim(0, 1)
            if kind == "line":
                lax.plot([0.05, 0.95], [0.5, 0.5], color=col, lw=size, ls=ls, solid_capstyle="butt")
            else:
                lax.plot([0.5], [0.5], marker=kind, color=col, markersize=size)
            fig.text(0.60, y + 0.012, head, fontsize=14, fontweight="bold", va="center")
            fig.text(0.60, y - 0.020, sub, fontsize=12, color=P_MUTED, va="center")
            y -= step
        fig.text(0.52, 0.17, "Junction choice correct in both runs.", fontsize=13, color=P_GREEN, va="center")
        fig.text(0.52, 0.12, "Open CANopen sensor profile, no hardware.", fontsize=13, color=P_MUTED, va="center")
        fig.savefig(path, dpi=100, facecolor=P_BG)
        plt.close(fig)
    return {k: v.metrics for k, v in runs.items()}
