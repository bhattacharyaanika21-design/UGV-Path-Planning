"""Rendering: bird's-eye view videos (MP4/GIF) and snapshot PNGs."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from .world import CLASSES, rect_corners  # noqa: E402

STATE_COLORS = {"CRUISE": "#2ca02c", "FOLLOW": "#1f77b4", "NUDGE": "#9467bd",
                "YIELD": "#ff7f0e", "OCCLUSION_CAUTION": "#bcbd22", "WAIT": "#7f7f7f",
                "EMERGENCY": "#d62728", "REROUTE": "#17becf"}


def draw_road(ax, net, blocked_color="#b33"):
    for e in net.edges.values():
        P = e["pts"]
        w = e["width"]
        for a, b in zip(P[:-1], P[1:]):
            d = b - a
            n = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-9) * w / 2
            poly = np.array([a + n, b + n, b - n, a - n])
            ax.add_patch(Polygon(poly, closed=True, fc="#5b5b5b", ec="none", zorder=1))
        for p in P:
            ax.add_patch(Circle(p, w / 2, fc="#5b5b5b", ec="none", zorder=1))


def render(result, path, fps=10, every=1, dpi=90, title_suffix=""):
    spec = result["spec"]
    frames = result["frames"][::every]
    world = spec.world
    net = world.network
    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi)
    ax = fig.add_axes([0.01, 0.12, 0.70, 0.80])
    axs = fig.add_axes([0.76, 0.58, 0.22, 0.30])
    axi = fig.add_axes([0.74, 0.08, 0.25, 0.44])
    axi.axis("off")
    ax.set_facecolor("#e9e4d8")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for poly, col in world.decor:
        ax.add_patch(Polygon(poly, closed=True, fc=col, ec="none", zorder=0))
    draw_road(ax, net)
    for (px, py, pr) in world.potholes:
        ax.add_patch(Circle((px, py), pr, fc="#3a2f25", ec="#2a2018", zorder=2))
    fig.suptitle(f"{spec.title}{title_suffix}", fontsize=14, fontweight="bold", x=0.36, y=0.97)
    fig.text(0.36, 0.925, spec.description, ha="center", fontsize=8.5, color="#333")

    t_all = np.array(result["log"]["t"])
    v_all = np.array(result["log"]["v"])
    vd_all = np.array(result["log"]["v_des"])
    axs.plot(t_all, v_all * 3.6, color="#1f77b4", lw=1.5, label="speed")
    axs.plot(t_all, vd_all * 3.6, color="#aaa", lw=1, ls="--", label="desired")
    axs.set_xlim(0, max(t_all[-1], 1))
    axs.set_ylim(0, max(v_all.max(), vd_all.max()) * 3.6 * 1.15 + 1)
    axs.set_xlabel("time [s]", fontsize=8)
    axs.set_ylabel("km/h", fontsize=8)
    axs.tick_params(labelsize=7)
    axs.legend(fontsize=7, loc="lower right")
    cursor = axs.axvline(0, color="k", lw=0.8)

    dyn = []
    ref_line, = ax.plot([], [], color="#ffffff", lw=0.8, ls=(0, (4, 4)), zorder=3, alpha=0.7)
    info = axi.text(0, 1, "", va="top", fontsize=9, family="monospace")
    # legend
    leg_items = [("ego", "#00b3b3"), ("pedestrian", CLASSES["pedestrian"]["color"]),
                 ("cattle", CLASSES["cattle"]["color"]), ("auto", CLASSES["auto"]["color"]),
                 ("2-wheeler", CLASSES["twowheeler"]["color"]), ("car", CLASSES["car"]["color"]),
                 ("bus", CLASSES["bus"]["color"]), ("truck", CLASSES["truck"]["color"])]
    for i, (lbl, col) in enumerate(leg_items):
        fig.text(0.02 + (i % 8) * 0.088, 0.06, "■ " + lbl, color=col, fontsize=9, fontweight="bold")
    fig.text(0.02, 0.025, "— planned path   ┄ candidate paths   · predicted (hard)   "
             "· possible (soft)   ○ fused tracks   ● potholes", fontsize=8.5, color="#333")

    def update(k):
        for h in dyn:
            h.remove()
        dyn.clear()
        f = frames[k]
        ex, ey, eyaw, ev, ea, ed = f["ego"]
        if f.get("ref_xy") is not None:
            ref_line.set_data(f["ref_xy"][:, 0], f["ref_xy"][:, 1])
        V = spec.view
        ax.set_xlim(ex - V * 0.75 + 12 * np.cos(eyaw), ex + V * 1.05 + 12 * np.cos(eyaw))
        ax.set_ylim(ey - V * 0.55 + 10 * np.sin(eyaw), ey + V * 0.55 + 10 * np.sin(eyaw))
        for corners, cls in f["agents"]:
            p = Polygon(corners, closed=True, fc=CLASSES[cls]["color"], ec="k", lw=0.6, zorder=6)
            ax.add_patch(p)
            dyn.append(p)
        for alt in f["alt"]:
            l, = ax.plot(alt[:, 0], alt[:, 1], color="#cfe8ff", lw=0.6, alpha=0.6, zorder=4)
            dyn.append(l)
        for tr in f["hard"]:
            l, = ax.plot(tr[:, 0], tr[:, 1], ".", color="#ff5a5a", ms=2.5, zorder=5)
            dyn.append(l)
        for tr in f["soft"]:
            l, = ax.plot(tr[:, 0], tr[:, 1], ".", color="#ffc04d", ms=1.8, alpha=0.7, zorder=5)
            dyn.append(l)
        for (pos, cls, tid, vel) in f["tracks"]:
            c = Circle(pos, 0.9, fc="none", ec="#ffffff", lw=0.9, zorder=7)
            ax.add_patch(c)
            dyn.append(c)
            if cls is not None:
                tx = ax.text(pos[0], pos[1] + 1.4, cls[:5], fontsize=6, color="white",
                             ha="center", zorder=8)
                dyn.append(tx)
        pl = f["plan"]
        col = "#ff2020" if f["emergency"] else "#39ff14"
        l, = ax.plot(pl[:, 0], pl[:, 1], color=col, lw=2.2, zorder=9)
        dyn.append(l)
        eg = Polygon(rect_corners(ex, ey, eyaw, 4.4, 1.8), closed=True, fc="#00b3b3",
                     ec="k", lw=1.2, zorder=10)
        ax.add_patch(eg)
        dyn.append(eg)
        # sensor footprints
        for rng_, fov, colr in ((80, np.deg2rad(55), "#ffffff"), (120, np.deg2rad(20), "#ffd000")):
            th = np.linspace(eyaw - fov, eyaw + fov, 20)
            pts = np.vstack([[ex, ey], np.column_stack([ex + rng_ * np.cos(th), ey + rng_ * np.sin(th)]), [ex, ey]])
            l, = ax.plot(pts[:, 0], pts[:, 1], color=colr, lw=0.5, alpha=0.35, zorder=3)
            dyn.append(l)
        c = Circle((ex, ey), 50, fc="none", ec="#88ccff", lw=0.5, alpha=0.35, ls="--", zorder=3)
        ax.add_patch(c)
        dyn.append(c)
        cursor.set_xdata([f["t"], f["t"]])
        st = f["state"]
        caps = ", ".join(f"{k_}≤{v_*3.6:.0f}" for k_, v_ in f["caps"].items()) or "–"
        clear = f["clearance"]
        info.set_text(
            f"t = {f['t']:5.1f} s\n"
            f"speed      {ev*3.6:5.1f} km/h\n"
            f"target     {f['v_des']*3.6:5.1f} km/h\n"
            f"accel      {ea:+5.2f} m/s²\n"
            f"steer      {np.rad2deg(ed):+5.1f}°\n"
            f"tracks     {len(f['tracks']):3d}\n"
            f"clearance  {clear if np.isfinite(clear) else 99:5.2f} m\n"
            f"caps  {caps}\n")
        info.set_color("#222")
        stx = axi.text(0, 0.32, f" {st} ", fontsize=13, fontweight="bold", color="white",
                       bbox=dict(fc=STATE_COLORS.get(st, "#444"), ec="none", boxstyle="round"))
        dyn.append(stx)
        return []

    if path.endswith(".gif"):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=2500, codec="libx264",
                              extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, path, dpi=dpi):
        for k in range(len(frames)):
            update(k)
            writer.grab_frame()
    plt.close(fig)


def snapshot(result, path, times, dpi=110):
    """Grid of bird's-eye snapshots at chosen times (for the report)."""
    import os
    import tempfile
    frames = result["frames"]
    ts = np.array([f["t"] for f in frames])
    idx = [int(np.argmin(np.abs(ts - t))) for t in times]
    sub = dict(result)
    imgs = []
    tmpdir = tempfile.mkdtemp()
    for i in idx:
        sub["frames"] = [frames[i]]
        p = os.path.join(tmpdir, f"f{i}.png")
        _render_single(sub, p, dpi)
        imgs.append(plt.imread(p))
    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(6.4 * n, 3.8), dpi=dpi)
    for a, im in zip(np.atleast_1d(axes), imgs):
        a.imshow(im)
        a.axis("off")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _render_single(result, path, dpi):
    import matplotlib.animation as anim  # noqa
    tmp = path.replace(".png", ".gif")
    render(result, tmp, fps=1, dpi=dpi)
    from PIL import Image
    Image.open(tmp).convert("RGB").save(path)
