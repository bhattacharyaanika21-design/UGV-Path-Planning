"""Short-term, multi-modal motion prediction for non-lane-based agents.

For every confirmed track we generate several hypotheses over a 5 s
horizon:
  * constant velocity (from the fused Kalman state) with class-specific,
    time-growing uncertainty — the *hard* constraint for the planner;
  * sudden stop (auto-rickshaws stopping for passengers, buses);
  * swerve left/right (two-wheelers and autos cutting across);
  * "dart into road" for pedestrians/cattle standing at the roadside;
Low-probability hypotheses become *soft* costs so the vehicle slows and
keeps extra clearance without freezing. A learned predictor (e.g. an LSTM
trained in Deep Learning Toolbox) can replace `predict_track` without
changing the planner interface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT_PRED = 0.2
HORIZON = 5.0
TGRID = np.arange(0.0, HORIZON + 1e-9, DT_PRED)  # K = 26

# class-specific rate of growth of positional uncertainty [m/s] capturing
# non-lane-based, irregular motion (how quickly the agent can deviate)
DEV_RATE = {"pedestrian": 0.25, "cattle": 0.3, "twowheeler": 0.35, "bicycle": 0.25,
            "auto": 0.3, "car": 0.15, "bus": 0.08, "truck": 0.08, "tractor": 0.1,
            "pushcart": 0.1, "bullockcart": 0.1, None: 0.35}
K_SIG = 0.7       # inflation = K_SIG * sigma (the rest is a soft proximity cost)
SIG_CAP = {"bus": 0.35, "truck": 0.35, "tractor": 0.4, "car": 0.6, "bullockcart": 0.4,
           "pushcart": 0.5, None: 0.9}
VRU = ("pedestrian", "cattle")


@dataclass
class Hypothesis:
    track_id: int
    cls: str
    prob: float
    hard: bool
    centers: np.ndarray      # (K, n_circ, 2)
    radius: np.ndarray       # (K,) circle radius incl. uncertainty
    base_radius: float
    label: str
    speed: float = 0.0


def circles_for(L, W):
    n = int(np.clip(np.ceil(2.4 * L / max(W, 0.3)), 1, 10))
    step = L / n
    offs = (np.arange(n) - (n - 1) / 2) * step
    r = float(np.hypot(step / 2, W / 2))
    return offs, r


def _rollout(p0, v0, yaw0, speed_fn, yawrate=0.0):
    pts, yaws = [], []
    p = p0.copy()
    yaw = yaw0
    prev_t = 0.0
    for t in TGRID:
        dt = t - prev_t
        v = speed_fn(t)
        yaw += yawrate * dt
        p = p + v * dt * np.array([np.cos(yaw), np.sin(yaw)])
        pts.append(p.copy())
        yaws.append(yaw)
        prev_t = t
    return np.array(pts), np.array(yaws)


def _stop_gap(tr, statics):
    """Distance along the track's heading to the first stopped vehicle in its
    way (interaction-aware prediction: queued traffic does not drive through
    a stopped bus)."""
    if tr.speed < 0.5:
        return None
    u = tr.vel / tr.speed
    n = np.array([-u[1], u[0]])
    L, W = tr.dims
    best = None
    for o in statics:
        if o is tr:
            continue
        rel = o.pos - tr.pos
        lon, lat = float(rel @ u), float(rel @ n)
        oL, oW = o.dims
        if 0 < lon < 60 and abs(lat) < (W + max(oW, 1.0)) / 2 + 0.3:
            g = lon - (L + max(oL, oW)) / 2 - 1.5
            if best is None or g < best:
                best = max(g, 0.0)
    return best


def predict_track(tr, ref=None, statics=()):
    cls = tr.cls
    L, W = tr.dims
    offs, r0 = circles_for(L, W)
    p0 = tr.pos.copy()
    v = tr.speed
    yaw = tr.yaw if v > 0.4 else tr.yaw
    pos_sig = np.sqrt(max(np.trace(tr.P[:2, :2]) / 2, 0.0))
    vel_sig = np.sqrt(max(np.trace(tr.P[2:, 2:]) / 2, 0.0))
    rate = DEV_RATE.get(cls, 0.35)
    static = v < 0.7 and (tr.t_last - tr.static_since) > 0.8
    if static:
        # parked / standing objects: no drift; vehicles stay put, VRUs keep a
        # growth term (they can start walking) and get the explicit dart mode
        v = 0.0
        vel_sig = 0.0
        rate = 0.03 if cls not in VRU else 0.15
    sig = pos_sig + (min(vel_sig, 0.6) + rate) * TGRID
    cap = SIG_CAP.get(cls, 0.9) if not static else min(SIG_CAP.get(cls, 0.9), 0.6)
    radius = r0 + np.minimum(K_SIG * sig, cap)

    hyps = []

    def make(pts, yaws, prob, hard, label, rad=radius):
        c = pts[:, None, :] + offs[None, :, None] * np.stack(
            [np.cos(yaws), np.sin(yaws)], -1)[:, None, :]
        hyps.append(Hypothesis(tr.id, cls, prob, hard, c, rad, r0, label, float(v)))

    moving = v > 0.4
    # 1) constant velocity — hard (interaction-aware: stops behind a stopped
    #    vehicle that lies in its way)
    gap = _stop_gap(tr, statics) if moving else None
    if gap is not None and gap < v * TGRID[-1]:
        dec = min(max(v * v / (2 * max(gap, 0.3)), 0.5), 6.0)
        pts, yaws = _rollout(p0, v, yaw, lambda t: max(0.0, v - dec * t))
    else:
        pts, yaws = _rollout(p0, v, yaw, lambda t: v)
    p_cv = 1.0
    extra = []
    if moving and cls in ("auto", "car", "bus", "truck", "twowheeler", "tractor"):
        # 2) sudden stop
        dec = 3.0
        extra.append(("stop", 0.2, _rollout(p0, v, yaw, lambda t: max(0.0, v - dec * t))))
    if moving and cls in ("twowheeler", "auto", "bicycle", "pedestrian", "cattle"):
        # 3) swerves (non-lane-based lateral motion)
        yr = 0.35 if cls != "pedestrian" else 0.4
        extra.append(("swerve_l", 0.15, _rollout(p0, v, yaw, lambda t: v, +yr)))
        extra.append(("swerve_r", 0.15, _rollout(p0, v, yaw, lambda t: v, -yr)))
    if cls in VRU and not moving and ref is not None:
        # 4) standing VRU near road may dart across
        s, d = ref.to_frenet(p0[0], p0[1])
        _, _, th, _ = ref.interp(s[0])
        normal = np.array([-np.sin(th), np.cos(th)])
        direction = -np.sign(d[0]) if abs(d[0]) > 0.1 else 1.0
        head = float(np.arctan2(*(normal * direction)[::-1]))
        spd = 1.4 if cls == "pedestrian" else 0.9
        if abs(d[0]) < 12:
            extra.append(("dart", 0.25, _rollout(p0, 0, head,
                                                  lambda t: 0.0 if t < 0.3 else spd)))
    for _, p, _ in extra:
        p_cv -= p
    make(pts, yaws, max(p_cv, 0.3), True, "cv")
    for label, p, (pp, yy) in extra:
        make(pp, yy, p, False, label)
    return hyps


def predict_all(tracks, ref=None):
    statics = [t for t in tracks if t.speed < 0.7 and t.cls not in VRU + ("bicycle",)
               and (t.t_last - t.static_since) > 0.8]
    out = []
    for tr in tracks:
        out.extend(predict_track(tr, ref, statics))
    return out
