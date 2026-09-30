"""Multi-sensor suite: camera, LiDAR and radar with occlusion and noise.

Each sensor returns world-frame detections with a covariance. Only the
camera classifies; LiDAR gives precise position and extent; radar gives
long range and radial velocity but sees small road users (pedestrians,
cattle) poorly. Occlusion by other road users and roadside structures is
ray-traced, so a pedestrian behind a parked bus is genuinely invisible
until they step out.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import CLASSES, CLASS_NAMES

CONFUSION_GROUPS = [
    ["car", "auto", "tractor"],
    ["bus", "truck"],
    ["twowheeler", "bicycle"],
    ["pedestrian"],
    ["cattle", "pushcart", "bullockcart"],
]


def _group(cls):
    for g in CONFUSION_GROUPS:
        if cls in g:
            return g
    return [cls]


@dataclass
class Detection:
    sensor: str
    pos: np.ndarray
    R: np.ndarray
    cls_prob: np.ndarray | None = None
    vr: float | None = None
    vr_var: float = 0.0
    sensor_pos: np.ndarray | None = None
    size: tuple | None = None
    yaw: float | None = None
    true_id: int = -1


def polar_cov(sensor_pos, target, sig_r, sig_b):
    rel = target - sensor_pos
    r = max(np.hypot(*rel), 0.5)
    b = np.arctan2(rel[1], rel[0])
    J = np.array([[np.cos(b), -r * np.sin(b)], [np.sin(b), r * np.cos(b)]])
    return J @ np.diag([sig_r ** 2, sig_b ** 2]) @ J.T


def _segments_blocked(origins, targets, edges, owners, target_owner):
    """For each ray (origin->target), is any edge not owned by target hit?"""
    if len(edges) == 0:
        return np.zeros(len(targets), bool)
    p = origins[:, None, :]
    r = (targets - origins)[:, None, :]
    q = edges[None, :, 0, :]
    s = (edges[:, 1, :] - edges[:, 0, :])[None]
    rxs = r[..., 0] * s[..., 1] - r[..., 1] * s[..., 0]
    qp = q - p
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qp[..., 0] * s[..., 1] - qp[..., 1] * s[..., 0]) / rxs
        u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / rxs
    hit = (np.abs(rxs) > 1e-9) & (t > 0.0) & (t < 0.98) & (u >= 0) & (u <= 1)
    hit &= owners[None, :] != target_owner[:, None]
    return hit.any(axis=1)


class SensorSuite:
    def __init__(self, seed=0, clutter=True):
        self.rng = np.random.default_rng(seed)
        self.cam = dict(fov=np.deg2rad(55), rng=80.0, mount=1.6)
        self.lidar = dict(rng=50.0, mount=0.0)
        self.radar = dict(fov=np.deg2rad(20), rng=120.0, mount=2.1)
        self.clutter = clutter

    # ------------------------------------------------------------------
    def _visibility(self, world, origin, agents):
        """Fraction-of-rays visibility test for each agent."""
        edges, owners = [], []
        for a in agents:
            c = a.corners()
            for i in range(4):
                edges.append([c[i], c[(i + 1) % 4]])
                owners.append(a.id)
        for k, poly in enumerate(world.occluders):
            P = np.asarray(poly)
            for i in range(len(P)):
                edges.append([P[i], P[(i + 1) % len(P)]])
                owners.append(-1000 - k)
        edges = np.array(edges) if edges else np.zeros((0, 2, 2))
        owners = np.array(owners)
        targets, towners = [], []
        for a in agents:
            rel = a.pos - origin
            n = np.array([-rel[1], rel[0]]) / max(np.hypot(*rel), 1e-6)
            half = 0.4 * max(a.W, min(a.L, 2.0))
            for off in (0.0, half, -half):
                targets.append(a.pos + n * off)
                towners.append(a.id)
        if not targets:
            return {}
        targets = np.array(targets)
        towners = np.array(towners)
        origins = np.repeat(origin[None], len(targets), axis=0)
        blocked = _segments_blocked(origins, targets, edges, owners, towners)
        vis = {}
        for i, a in enumerate(agents):
            vis[a.id] = 1.0 - blocked[3 * i:3 * i + 3].mean()
        return vis

    def sense(self, world, ego):
        agents = world.active_agents()
        c, s = np.cos(ego.yaw), np.sin(ego.yaw)
        fwd = np.array([c, s])
        base = np.array([ego.x, ego.y])
        dets = []
        potholes = []
        if not agents:
            return dets, potholes

        # --- camera --------------------------------------------------------
        cam_pos = base + fwd * self.cam["mount"]
        vis = self._visibility(world, cam_pos, agents)
        for a in agents:
            rel = a.pos - cam_pos
            r = np.hypot(*rel)
            b = np.arctan2(rel[1], rel[0]) - ego.yaw
            b = (b + np.pi) % (2 * np.pi) - np.pi
            if r > self.cam["rng"] or abs(b) > self.cam["fov"]:
                continue
            if vis[a.id] < 0.34:
                continue
            pd = 0.97 - 0.25 * (r / self.cam["rng"]) ** 2
            if a.cls in ("pedestrian", "cattle") and r > 45:
                pd -= 0.15
            if self.rng.random() > pd:
                continue
            sig_r = 0.2 + 0.04 * r
            R = polar_cov(cam_pos, a.pos, sig_r, 0.006)
            z = self.rng.multivariate_normal(a.pos, R)
            probs = np.full(len(CLASS_NAMES), 0.02)
            acc = 0.9 - 0.2 * r / self.cam["rng"]
            g = _group(a.cls)
            label = a.cls
            if self.rng.random() > acc and len(g) > 1:
                label = self.rng.choice([k for k in g if k != a.cls])
            probs[CLASS_NAMES.index(label)] = 0.75
            for k in g:
                if k != label:
                    probs[CLASS_NAMES.index(k)] += 0.1
            probs /= probs.sum()
            dets.append(Detection("camera", z, R, cls_prob=probs, sensor_pos=cam_pos,
                                  true_id=a.id))
        # potholes (camera only, short range)
        for (px, py, pr) in world.potholes:
            rel = np.array([px, py]) - cam_pos
            r = np.hypot(*rel)
            b = np.arctan2(rel[1], rel[0]) - ego.yaw
            b = (b + np.pi) % (2 * np.pi) - np.pi
            if r < 28 and abs(b) < self.cam["fov"] and self.rng.random() < 0.9:
                potholes.append((px + self.rng.normal(0, 0.15), py + self.rng.normal(0, 0.15), pr))
        if self.clutter and self.rng.random() < 0.05:
            rr = self.rng.uniform(10, 50)
            bb = ego.yaw + self.rng.uniform(-0.8, 0.8)
            z = cam_pos + rr * np.array([np.cos(bb), np.sin(bb)])
            probs = np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES))
            dets.append(Detection("camera", z, polar_cov(cam_pos, z, 1.0, 0.01),
                                  cls_prob=probs, sensor_pos=cam_pos, true_id=-2))

        # --- LiDAR ---------------------------------------------------------
        lid_pos = base
        vis = self._visibility(world, lid_pos, agents)
        for a in agents:
            r = np.hypot(*(a.pos - lid_pos))
            if r > self.lidar["rng"] or vis[a.id] < 0.34:
                continue
            pd = 0.98 if a.W > 0.7 else (0.95 if r < 30 else 0.8)
            if self.rng.random() > pd:
                continue
            R = np.eye(2) * 0.12 ** 2 * (1 + r / 50)
            z = self.rng.multivariate_normal(a.pos, R)
            size = (max(0.3, a.L + self.rng.normal(0, 0.2)), max(0.3, a.W + self.rng.normal(0, 0.1)))
            dets.append(Detection("lidar", z, R, sensor_pos=lid_pos, size=size,
                                  yaw=float(a.yaw + self.rng.normal(0, 0.05)), true_id=a.id))

        # --- radar ---------------------------------------------------------
        rad_pos = base + fwd * self.radar["mount"]
        vis = self._visibility(world, rad_pos, agents)
        for a in agents:
            rel = a.pos - rad_pos
            r = np.hypot(*rel)
            b = np.arctan2(rel[1], rel[0]) - ego.yaw
            b = (b + np.pi) % (2 * np.pi) - np.pi
            if r > self.radar["rng"] or abs(b) > self.radar["fov"]:
                continue
            see = vis[a.id] >= 0.34 or self.rng.random() < 0.15  # multipath
            pd = 0.35 + 0.6 * CLASSES[a.cls]["rcs"]
            if not see or self.rng.random() > pd:
                continue
            R = polar_cov(rad_pos, a.pos, 0.3, 0.02)
            z = self.rng.multivariate_normal(a.pos, R)
            u = rel / max(r, 1e-6)
            vr = float(np.dot(a.vel, u) + self.rng.normal(0, 0.2))
            dets.append(Detection("radar", z, R, vr=vr, vr_var=0.2 ** 2, sensor_pos=rad_pos,
                                  true_id=a.id))
        return dets, potholes
