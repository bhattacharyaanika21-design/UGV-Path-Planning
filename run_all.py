#!/usr/bin/env python3
"""Run all scenarios (Monte-Carlo over seeds), collect metrics, render videos.

Usage
  python3 run_all.py                 # 6 scenarios x 5 seeds, videos for seed 0
  python3 run_all.py --seeds 10 --jobs 4
  python3 run_all.py --only S5 --seeds 1
  python3 run_all.py --no-video      # metrics only (fast)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from multiprocessing import Pool

import numpy as np

from av.scenarios import SCENARIOS
from av.sim import run

OUT = "results"


def _job(args):
    key, seed, video = args
    t0 = time.time()
    res = run(SCENARIOS[key](seed), seed=seed, record=video)
    m = res["metrics"]
    m["seed"] = seed
    m["wall_s"] = round(time.time() - t0, 1)
    if video:
        from av.viz import render
        os.makedirs(f"{OUT}/videos", exist_ok=True)
        path = f"{OUT}/videos/{m['scenario']}.mp4"
        render(res, path, fps=10, dpi=80)
        m["video"] = path
        # time-series for the report
        np.savez_compressed(f"{OUT}/logs_{m['scenario']}.npz",
                            **{k: np.array(v, dtype=object if k == "state" else float)
                               for k, v in res["log"].items()},
                            plan_ms=np.array([p["ms"] for p in res["plans"]]),
                            plan_t=np.array([p["t"] for p in res["plans"]]))
        _export_traj(res, f"{OUT}/traj_{m['scenario']}.csv")
    print(f"[{key} seed {seed}] completed={m['completed']} collision={m['collision']} "
          f"t={m['time_s']:.1f}s  replan {m['replan_latency_ms_mean']:.0f} ms  "
          f"({m['wall_s']} s wall)", flush=True)
    return m


def _export_traj(res, path):
    """Ego trajectory as CSV (for MATLAB / RoadRunner playback)."""
    L = res["log"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw", "v", "a", "steer", "state"])
        for i in range(len(L["t"])):
            w.writerow([f"{L['t'][i]:.2f}", f"{L['x'][i]:.3f}", f"{L['y'][i]:.3f}",
                        f"{L['yaw'][i]:.4f}", f"{L['v'][i]:.3f}", f"{L['a'][i]:.3f}",
                        f"{L['delta'][i]:.4f}", L["state"][i]])


def summarise(rows):
    by = {}
    for r in rows:
        by.setdefault(r["scenario"], []).append(r)
    summary = []
    for key, rs in by.items():
        lat = [r["reaction_latency_s_mean"] for r in rs if r["reaction_latency_s_mean"] is not None]
        summary.append(dict(
            scenario=key, title=rs[0]["title"], runs=len(rs),
            completion_rate=float(np.mean([r["completed"] for r in rs])),
            collisions=int(sum(r["collision"] for r in rs)),
            time_s=float(np.mean([r["time_s"] for r in rs])),
            avg_speed_kmh=float(np.mean([r["avg_speed_mps"] for r in rs]) * 3.6),
            min_clearance_m=float(np.min([r["min_clearance_m"] for r in rs])),
            replan_ms_mean=float(np.mean([r["replan_latency_ms_mean"] for r in rs])),
            replan_ms_p95=float(np.max([r["replan_latency_ms_p95"] for r in rs])),
            replans_per_s=float(np.mean([r["replans"] / max(r["time_s"], 1) for r in rs])),
            event_replans=int(sum(r["replan_reasons"].get("new_object", 0) +
                                  r["replan_reasons"].get("plan_invalid", 0) for r in rs)),
            reaction_s=float(np.mean(lat)) if lat else None,
            rms_long_jerk=float(np.mean([r["rms_long_jerk"] for r in rs])),
            rms_lat_jerk=float(np.mean([r["rms_lat_jerk"] for r in rs])),
            max_lat_acc=float(np.max([r["max_lat_acc"] for r in rs])),
            max_decel=float(np.max([r["max_decel"] for r in rs])),
            emergency_rate=float(np.mean([r["emergency_plans"] / max(r["replans"], 1) for r in rs])),
            reroutes=int(sum(r["reroutes"] for r in rs)),
            potholes_hit=f"{sum(r['potholes_hit'] for r in rs)}/{sum(r['potholes_total'] for r in rs)}",
            track_rmse_m=float(np.nanmean([r["track_rmse_m"] or np.nan for r in rs])),
        ))
    return summary


def plot_summary(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    names = [s["scenario"].split("_")[0] + "\n" + s["scenario"].split("_", 1)[1] for s in summary]
    panels = [
        ("Scenario completion rate", [s["completion_rate"] * 100 for s in summary], "%", blue, (0, 105)),
        ("Mean replanning latency (compute)", [s["replan_ms_mean"] for s in summary], "ms", blue, None),
        ("Minimum clearance to any road user", [s["min_clearance_m"] for s in summary], "m", aqua, None),
        ("RMS longitudinal jerk (smoothness)", [s["rms_long_jerk"] for s in summary], "m/s³", orange, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=130, facecolor=surf)
    for ax, (title, vals, unit, col, ylim) in zip(axes.ravel(), panels):
        ax.set_facecolor(surf)
        bars = ax.bar(names, vals, color=col, width=0.6, zorder=3)
        ax.set_title(title, loc="left", fontsize=11, color=ink, fontweight="bold")
        ax.set_ylabel(unit, color=ink2)
        ax.grid(axis="y", color=grid, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(grid)
        ax.tick_params(colors=ink2, labelsize=8)
        if ylim:
            ax.set_ylim(*ylim)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}" if v < 10 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8, color=ink)
    fig.suptitle("Closed-loop validation summary (Monte-Carlo over seeds)", x=0.01, ha="left",
                 fontsize=13, color=ink, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=surf)
    plt.close(fig)


def title_card(text, sub, path, secs=2.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    fig = plt.figure(figsize=(12.8, 7.2), dpi=80, facecolor="#101418")
    fig.text(0.5, 0.56, text, ha="center", va="center", color="white", fontsize=30, fontweight="bold")
    fig.text(0.5, 0.44, sub, ha="center", va="center", color="#c3c2b7", fontsize=15, wrap=True)
    w = FFMpegWriter(fps=10, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with w.saving(fig, path, dpi=80):
        for _ in range(int(secs * 10)):
            w.grab_frame()
    plt.close(fig)


def make_demo(summary):
    parts = []
    os.makedirs(f"{OUT}/videos/_cards", exist_ok=True)
    intro = f"{OUT}/videos/_cards/00_intro.mp4"
    title_card("Adaptive Path Planning on Unstructured Indian Roads",
               "Perception → Fusion → Prediction → Decision FSM → Lattice planner → Control", intro, 3.5)
    parts.append(intro)
    for s in summary:
        vid = f"{OUT}/videos/{s['scenario']}.mp4"
        if not os.path.exists(vid):
            continue
        card = f"{OUT}/videos/_cards/{s['scenario']}.mp4"
        title_card(s["title"], f"completion {s['completion_rate']*100:.0f}%  ·  "
                   f"collisions {s['collisions']}  ·  replan {s['replan_ms_mean']:.0f} ms", card)
        parts += [card, vid]
    lst = f"{OUT}/videos/_cards/list.txt"
    with open(lst, "w") as f:
        for p in parts:
            f.write(f"file '{os.path.abspath(p)}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", f"{OUT}/demo_all_scenarios.mp4"],
                   check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2)))
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    keys = a.only or list(SCENARIOS)
    jobs = [(k, s, (s == 0 and not a.no_video)) for k in keys for s in range(a.seeds)]
    jobs.sort(key=lambda j: not j[2])  # video jobs first
    with Pool(a.jobs) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    if a.only and os.path.exists(f"{OUT}/metrics_runs.json"):
        # merge with previous runs of the other scenarios
        done = {r["scenario"] for r in rows}
        old = json.load(open(f"{OUT}/metrics_runs.json"))
        rows += [r for r in old if r["scenario"] not in done]
    rows.sort(key=lambda r: (r["scenario"], r["seed"]))
    with open(f"{OUT}/metrics_runs.json", "w") as f:
        json.dump(rows, f, indent=1, default=str)
    summary = summarise(rows)
    with open(f"{OUT}/metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(f"{OUT}/metrics_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    plot_summary(summary, f"{OUT}/summary_dashboard.png")
    if not a.no_video:
        make_demo(summary)
    print("\n=== SUMMARY ===")
    for s in summary:
        print(f"{s['scenario']:14s} completion {s['completion_rate']*100:5.1f}%  collisions {s['collisions']}"
              f"  replan {s['replan_ms_mean']:5.1f} ms (p95 {s['replan_ms_p95']:5.1f})"
              f"  min clr {s['min_clearance_m']:.2f} m  jerk {s['rms_long_jerk']:.2f}")


if __name__ == "__main__":
    main()
