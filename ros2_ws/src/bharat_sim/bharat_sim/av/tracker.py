"""Multi-sensor, multi-object tracker (sensor fusion).

Constant-velocity Kalman filter per track, Mahalanobis gating and global
nearest-neighbour (Hungarian) association per sensor, radar radial-velocity
EKF update, camera class-probability fusion, LiDAR extent fusion and
M-of-N track confirmation. This mirrors the role of
`trackerGNN`/`multiObjectTracker` in MATLAB's Automated Driving Toolbox.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from .world import CLASSES, CLASS_NAMES

ACC_NOISE = {"pedestrian": 1.5, "cattle": 1.5, "twowheeler": 3.0, "bicycle": 1.5,
             "auto": 2.5, "car": 2.5, "bus": 1.5, "truck": 1.5, "tractor": 1.2,
             "pushcart": 1.0, "bullockcart": 1.0, None: 3.0}
GATE = 11.8  # chi2(2 dof) 99.7 %


class Track:
    _next = 1

    def __init__(self, z, R, t, det):
        self.id = Track._next
        Track._next += 1
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([R[0, 0] + 0.1, R[1, 1] + 0.1, 16.0, 16.0])
        self.logp = np.zeros(len(CLASS_NAMES))
        self.has_cam = False
        self.size = None
        self.hits = 1
        self.cycles = 1
        self.hit_cycles = 1
        self.confirmed = False
        self.t_first = t
        self.t_last = t
        self.t_confirm = None
        self.static_since = t
        self.yaw = 0.0
        self.true_ids = Counter()
        self.sensors = set()
        self._hit_this_cycle = True
        self.update_meta(det)

    # ------------------------------------------------------------------
    @property
    def pos(self):
        return self.x[:2]

    @property
    def vel(self):
        return self.x[2:]

    @property
    def speed(self):
        return float(np.hypot(*self.x[2:]))

    @property
    def cls(self):
        if self.has_cam:
            return CLASS_NAMES[int(np.argmax(self.logp))]
        if self.size is not None:
            L, W = self.size
            if L < 0.9 and W < 0.9:
                return "pedestrian"
            if L > 8:
                return "truck"
            if L > 3.5:
                return "car"
            if W < 1.0:
                return "twowheeler"
            return "auto"
        return None

    @property
    def dims(self):
        c = self.cls
        if self.size is not None:
            return self.size
        if c is not None:
            return CLASSES[c]["L"], CLASSES[c]["W"]
        return 2.0, 1.2

    @property
    def true_id(self):
        return self.true_ids.most_common(1)[0][0] if self.true_ids else -1

    # ------------------------------------------------------------------
    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = ACC_NOISE.get(self.cls, 3.0) ** 2
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ (q * np.eye(2)) @ G.T

    def innov(self, z, R):
        H = np.zeros((2, 4))
        H[0, 0] = H[1, 1] = 1
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        return y, S, H

    def update(self, det, t):
        y, S, H = self.innov(det.pos, det.R)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        if det.vr is not None and det.sensor_pos is not None:
            rel = self.x[:2] - det.sensor_pos
            u = rel / max(np.hypot(*rel), 1e-6)
            Hv = np.array([[0, 0, u[0], u[1]]])
            yv = det.vr - float((Hv @ self.x)[0])
            Sv = float((Hv @ self.P @ Hv.T)[0, 0]) + det.vr_var
            Kv = (self.P @ Hv.T) / Sv
            self.x = self.x + (Kv * yv).ravel()
            self.P = (np.eye(4) - Kv @ Hv) @ self.P
        self.hits += 1
        self.t_last = t
        self._hit_this_cycle = True
        self.update_meta(det)

    def update_meta(self, det):
        self.sensors.add(det.sensor)
        if det.cls_prob is not None:
            self.has_cam = True
            self.logp += np.log(det.cls_prob + 1e-3)
            self.logp -= self.logp.max()
            self.logp = np.maximum(self.logp, -30)
        if det.yaw is not None:
            # LiDAR box orientation (ambiguous by pi) is far more accurate than
            # the heading of a noisy low-speed velocity estimate
            y = det.yaw
            if self.speed > 0.4:
                vh = np.arctan2(self.x[3], self.x[2])
                if np.cos(y - vh) < 0:
                    y += np.pi
            self.yaw = float(y)
            self.lidar_yaw_t = self.t_last
        if det.size is not None:
            self.size = det.size if self.size is None else tuple(
                0.8 * np.array(self.size) + 0.2 * np.array(det.size))
        self.true_ids[det.true_id] += 1


class MultiObjectTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.t = 0.0
        self.new_confirmed: list[Track] = []

    def step(self, dets, t):
        dt = t - self.t if self.tracks else 0.0
        self.t = t
        for tr in self.tracks:
            if dt > 0:
                tr.predict(dt)
            tr._hit_this_cycle = False
        # sequential per-sensor update (LiDAR first: best position accuracy)
        for sensor in ("lidar", "radar", "camera"):
            batch = [d for d in dets if d.sensor == sensor]
            if not batch:
                continue
            self._associate(batch, t, spawn=True)
        # track management
        self.new_confirmed = []
        keep = []
        for tr in self.tracks:
            tr.cycles += 1
            if tr._hit_this_cycle:
                tr.hit_cycles += 1
            ranged = ("lidar" in tr.sensors) or ("radar" in tr.sensors)
            need = 3 if ranged else 6
            vmax = 5.0 if tr.cls in ("pedestrian", "cattle") else 35.0
            if not tr.confirmed and tr.hit_cycles >= need and tr.hit_cycles >= 0.6 * tr.cycles \
                    and tr.speed < vmax:
                tr.confirmed = True
                tr.t_confirm = t
                self.new_confirmed.append(tr)
            miss = t - tr.t_last
            limit = 2.0 if tr.confirmed else 0.35
            if miss > limit or np.trace(tr.P[:2, :2]) > 60:
                continue
            if tr.speed > 0.4 and t - getattr(tr, "lidar_yaw_t", -1e9) > 0.35:
                tr.yaw = float(np.arctan2(tr.x[3], tr.x[2]))
            if tr.speed > 0.7:
                tr.static_since = t
            keep.append(tr)
        self.tracks = self._merge(keep)

    def _associate(self, batch, t, spawn):
        n, m = len(self.tracks), len(batch)
        if n and m:
            C = np.full((n, m), 1e6)
            for i, tr in enumerate(self.tracks):
                for j, d in enumerate(batch):
                    y, S, _ = tr.innov(d.pos, d.R)
                    md = float(y @ np.linalg.solve(S, y))
                    if md < GATE:
                        C[i, j] = md
            rows, cols = linear_sum_assignment(C)
            used = set()
            for i, j in zip(rows, cols):
                if C[i, j] < GATE:
                    self.tracks[i].update(batch[j], t)
                    used.add(j)
        else:
            used = set()
        if spawn:
            for j, d in enumerate(batch):
                if j in used:
                    continue
                # camera-only births only beyond LiDAR range (range error is large)
                if d.sensor == "camera" and d.sensor_pos is not None and \
                        np.hypot(*(d.pos - d.sensor_pos)) < 48:
                    continue
                # suppress births right on top of an existing track
                if any(np.hypot(*(tr.pos - d.pos)) < 0.8 for tr in self.tracks):
                    continue
                self.tracks.append(Track(d.pos, d.R, t, d))

    def _merge(self, tracks):
        """Merge duplicate tracks that converged onto the same object."""
        tracks = sorted(tracks, key=lambda tr: -tr.hits)
        out = []
        for tr in tracks:
            dup = False
            for o in out:
                if np.hypot(*(tr.pos - o.pos)) < 1.0 and np.hypot(*(tr.vel - o.vel)) < 2.0:
                    dup = True
                    break
            if not dup:
                out.append(tr)
        return out

    def confirmed(self):
        return [t for t in self.tracks if t.confirmed]
