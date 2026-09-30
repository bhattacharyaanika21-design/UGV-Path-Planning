"""Geometry helpers: filleted polylines and a Frenet reference path.

The reference path is the backbone of the planner. On unstructured Indian
roads there are no lane markings, so the "reference" is simply the road
centre line recovered from the map / road-edge estimate, and the planner is
free to use any lateral offset inside the drivable width.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fillet_polyline(pts, radius=8.0, step=0.5):
    """Replace sharp corners of a polyline with circular arcs, then densify."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return densify(pts, step)
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        v1 = p0 - p1
        v2 = p2 - p1
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if l1 < 1e-6 or l2 < 1e-6:
            continue
        u1, u2 = v1 / l1, v2 / l2
        cosang = np.clip(np.dot(u1, u2), -1.0, 1.0)
        ang = np.arccos(cosang)  # interior angle
        if ang > np.pi - 1e-3:  # straight
            out.append(p1)
            continue
        tan_len = radius / np.tan(ang / 2)
        tan_len = min(tan_len, 0.45 * l1, 0.45 * l2)
        r = tan_len * np.tan(ang / 2)
        a = p1 + u1 * tan_len
        b = p1 + u2 * tan_len
        bis = u1 + u2
        bis /= np.linalg.norm(bis)
        c = p1 + bis * (r / np.sin(ang / 2))
        th_a = np.arctan2(*(a - c)[::-1])
        th_b = np.arctan2(*(b - c)[::-1])
        dth = wrap_angle(th_b - th_a)
        n = max(3, int(abs(dth) * r / step))
        for t in np.linspace(0, 1, n):
            th = th_a + dth * t
            out.append(c + r * np.array([np.cos(th), np.sin(th)]))
    out.append(pts[-1])
    return densify(np.array(out), step)


def densify(pts, step=0.5):
    pts = np.asarray(pts, dtype=float)
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    keep = np.concatenate([[True], seglen > 1e-6])
    pts = pts[keep]
    seg = np.diff(pts, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])])


class ReferencePath:
    """Arc-length parameterised centre line with Frenet conversions."""

    def __init__(self, pts, step=0.5, smooth=True):
        pts = densify(pts, step)
        if smooth and len(pts) > 7:
            # light moving-average smoothing that keeps end points fixed
            k = 5
            ker = np.ones(k) / k
            xs = np.convolve(np.pad(pts[:, 0], k // 2, mode="edge"), ker, "valid")
            ys = np.convolve(np.pad(pts[:, 1], k // 2, mode="edge"), ker, "valid")
            pts = np.column_stack([xs, ys])
            pts = densify(pts, step)
        self.xy = pts
        d = np.diff(pts, axis=0)
        seglen = np.hypot(d[:, 0], d[:, 1])
        self.s = np.concatenate([[0.0], np.cumsum(seglen)])
        self.length = float(self.s[-1])
        th = np.arctan2(d[:, 1], d[:, 0])
        th = np.concatenate([th, [th[-1]]])
        self.theta = np.unwrap(th)
        # curvature by finite difference of heading, lightly smoothed
        dth = np.gradient(self.theta, self.s)
        ker = np.ones(7) / 7
        self.kappa = np.convolve(np.pad(dth, 3, mode="edge"), ker, "valid")
        self.tree = cKDTree(pts)

    # --- sampling -----------------------------------------------------
    def interp(self, s):
        s = np.clip(s, 0.0, self.length)
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        th = np.interp(s, self.s, self.theta)
        k = np.interp(s, self.s, self.kappa)
        return x, y, th, k

    def to_cart(self, s, d):
        s = np.asarray(s, dtype=float)
        d = np.asarray(d, dtype=float)
        x, y, th, _ = self.interp(s)
        # linear extrapolation beyond the path end keeps planner well defined
        over = s - self.length
        x = x + np.where(over > 0, over * np.cos(th), 0.0)
        y = y + np.where(over > 0, over * np.sin(th), 0.0)
        return x - d * np.sin(th), y + d * np.cos(th), th

    def to_frenet(self, x, y):
        p = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
        _, idx = self.tree.query(p)
        idx = np.clip(idx, 0, len(self.s) - 2)
        base = self.xy[idx]
        th = self.theta[idx]
        t = np.column_stack([np.cos(th), np.sin(th)])
        rel = p - base
        ds = np.einsum("ij,ij->i", rel, t)
        dd = t[:, 0] * rel[:, 1] - t[:, 1] * rel[:, 0]
        s = self.s[idx] + ds
        return s, dd
