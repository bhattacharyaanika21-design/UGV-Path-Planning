"""Ground-truth world: road network, mixed-traffic agents, potholes, occluders.

Agent behaviours are deliberately *not* lane based. They include lateral
wander, informal merging without signalling, wrong-way driving, abrupt stops,
pedestrians darting out, and cattle that stop or turn back mid-road.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import ReferencePath, fillet_polyline, wrap_angle

# ---------------------------------------------------------------------------
# Road-user classes (length, width, colour, lidar/radar visibility)
# ---------------------------------------------------------------------------
CLASSES = {
    "car":        dict(L=4.2, W=1.8, color="#4c78a8", rcs=1.0),
    "bus":        dict(L=11.0, W=2.6, color="#f58518", rcs=1.0),
    "truck":      dict(L=9.0, W=2.5, color="#b279a2", rcs=1.0),
    "tractor":    dict(L=5.5, W=2.2, color="#9d755d", rcs=1.0),
    "auto":       dict(L=2.8, W=1.4, color="#e7ba52", rcs=0.8),
    "twowheeler": dict(L=2.0, W=0.8, color="#54a24b", rcs=0.6),
    "bicycle":    dict(L=1.8, W=0.6, color="#88d27a", rcs=0.4),
    "pedestrian": dict(L=0.5, W=0.5, color="#e45756", rcs=0.3),
    "cattle":     dict(L=2.2, W=0.9, color="#7f4f24", rcs=0.5),
    "pushcart":   dict(L=2.0, W=1.2, color="#ff9da6", rcs=0.5),
    "bullockcart": dict(L=4.5, W=1.8, color="#a0522d", rcs=0.6),
}
CLASS_NAMES = list(CLASSES.keys())


def rect_corners(x, y, yaw, L, W):
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.array([L / 2, L / 2, -L / 2, -L / 2])
    dy = np.array([W / 2, -W / 2, -W / 2, W / 2])
    return np.column_stack([x + c * dx - s * dy, y + s * dx + c * dy])


def rects_overlap(a, b):
    """Separating-axis test for two convex quads (4x2 arrays)."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa = a @ n
            pb = b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def rect_distance(a, b):
    """Approximate clearance between two quads (0 if overlapping)."""
    if rects_overlap(a, b):
        return 0.0
    best = np.inf
    for P, Q in ((a, b), (b, a)):
        for p in P:
            for i in range(4):
                q0, q1 = Q[i], Q[(i + 1) % 4]
                e = q1 - q0
                t = np.clip(np.dot(p - q0, e) / max(np.dot(e, e), 1e-9), 0, 1)
                best = min(best, np.linalg.norm(p - (q0 + t * e)))
    return best


# ---------------------------------------------------------------------------
# Road network (graph of road segments) — used for drawing, widths, rerouting
# ---------------------------------------------------------------------------
class RoadNetwork:
    def __init__(self):
        self.nodes: dict[str, np.ndarray] = {}
        self.edges: dict[tuple, dict] = {}

    def add_node(self, name, x, y):
        self.nodes[name] = np.array([x, y], dtype=float)

    def add_edge(self, u, v, width=7.0, speed=10.0, shape=None):
        key = tuple(sorted((u, v)))
        pts = [self.nodes[u]] + ([np.asarray(p) for p in shape] if shape else []) + [self.nodes[v]]
        if key[0] != u:
            pts = pts[::-1]
        self.edges[key] = dict(width=width, speed=speed, blocked=False, pts=np.array(pts, float))

    def neighbors(self, n):
        for (a, b) in self.edges:
            if a == n:
                yield b
            elif b == n:
                yield a

    def edge(self, u, v):
        return self.edges[tuple(sorted((u, v)))]

    def edge_pts(self, u, v):
        key = tuple(sorted((u, v)))
        pts = self.edges[key]["pts"]
        return pts if key[0] == u else pts[::-1]

    def route_waypoints(self, route):
        pts = [self.nodes[route[0]]]
        for u, v in zip(route[:-1], route[1:]):
            pts.extend(list(self.edge_pts(u, v)[1:]))
        return np.array(pts)

    def route_path(self, route, fillet=8.0):
        return ReferencePath(fillet_polyline(self.route_waypoints(route), fillet))

    def width_at(self, pts):
        """Road width at each query point = width of the nearest edge."""
        pts = np.atleast_2d(pts)
        best_d = np.full(len(pts), np.inf)
        best_w = np.zeros(len(pts))
        for e in self.edges.values():
            P = e["pts"]
            for a, b in zip(P[:-1], P[1:]):
                ab = b - a
                t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
                proj = a + t[:, None] * ab
                d = np.hypot(*(pts - proj).T)
                m = d < best_d
                best_d[m] = d[m]
                best_w[m] = e["width"]
        return best_w, best_d

    def edge_of_point(self, p):
        best, bk = np.inf, None
        for k, e in self.edges.items():
            P = e["pts"]
            for a, b in zip(P[:-1], P[1:]):
                ab = b - a
                t = np.clip(np.dot(p - a, ab) / max(ab @ ab, 1e-9), 0, 1)
                d = np.linalg.norm(p - (a + t * ab))
                if d < best:
                    best, bk = d, k
        return bk, best


# ---------------------------------------------------------------------------
# Agents and behaviours
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    id: int
    cls: str
    x: float
    y: float
    yaw: float
    v: float
    behavior: object = None
    L: float = 0.0
    W: float = 0.0
    active: bool = True

    def __post_init__(self):
        spec = CLASSES[self.cls]
        self.L = self.L or spec["L"]
        self.W = self.W or spec["W"]

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @property
    def vel(self):
        return self.v * np.array([np.cos(self.yaw), np.sin(self.yaw)])

    def corners(self):
        return rect_corners(self.x, self.y, self.yaw, self.L, self.W)

    def step(self, world, dt):
        if self.behavior is not None and self.active:
            self.behavior.step(self, world, dt)


def _trigger(trig, world, agent):
    """trig = ('time', t) or ('ego_dist', d) or None."""
    if trig is None:
        return True
    kind, val = trig
    if kind == "time":
        return world.t >= val
    if kind == "ego_dist":
        return np.hypot(world.ego.x - agent.x, world.ego.y - agent.y) <= val
    if kind == "ego_x":
        return world.ego.x >= val
    return True


def _objects(agent, world, include_ego=True):
    objs = [(o.x, o.y, o.L, o.W, o.v, o.yaw, o.cls) for o in world.agents
            if o is not agent and o.active]
    if include_ego:
        e = world.ego
        objs.append((e.x, e.y, e.L, e.W, e.v, e.yaw, "ego"))
    return objs


def _rel(agent, objs, path=None, s_ag=None, d_ag=None):
    """Longitudinal/lateral offsets of objects in the agent's frame. If the
    agent follows a path, use path (Frenet) coordinates so curved roads work."""
    if not objs:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    ox = np.array([o[0] for o in objs])
    oy = np.array([o[1] for o in objs])
    oyaw = np.array([o[5] for o in objs])
    ov = np.array([o[4] for o in objs])
    if path is not None:
        so, do = path.to_frenet(ox, oy)
        _, _, th, _ = path.interp(so)
        return so - s_ag, do - d_ag, ov * np.cos(oyaw - th)
    c, s = np.cos(agent.yaw), np.sin(agent.yaw)
    rx, ry = ox - agent.x, oy - agent.y
    return c * rx + s * ry, -s * rx + c * ry, ov * np.cos(oyaw - agent.yaw)


def idm_accel(agent, world, v0, a_max=1.5, b=3.0, T=1.2, s0=2.0, include_ego=True,
              path=None, s_ag=None, d_ag=None):
    """Intelligent-driver-model accel using a corridor ahead of the agent."""
    gap, dv, lead = 1e9, 0.0, None
    objs = _objects(agent, world, include_ego)
    lx, ly, ovl = _rel(agent, objs, path, s_ag, d_ag)
    for k, o in enumerate(objs):
        oL, oW = o[2], o[3]
        g = lx[k] - (agent.L + oL) / 2
        buf = 0.1 if g > 0 else 0.0   # already alongside: only true overlap counts
        if 0 < lx[k] < 40 and abs(ly[k]) < (agent.W + oW) / 2 + buf:
            if g < gap:
                gap, dv = g, agent.v - ovl[k]
                lead = dict(ly=float(ly[k]), v=float(ovl[k]), W=oW, gap=float(g))
    v = agent.v
    sstar = s0 + max(0.0, v * T + v * dv / (2 * np.sqrt(a_max * b)))
    acc = a_max * (1 - (v / max(v0, 0.1)) ** 4) - a_max * (sstar / max(gap, 0.1)) ** 2
    return float(np.clip(acc, -8.0, a_max)), lead


def corridor_clear(agent, world, shift, ahead=30.0, behind=6.0, half=None,
                   path=None, s_ag=None, d_ag=None):
    """Is the lateral band at `shift` (agent frame, +left) free of other objects?"""
    half = half if half is not None else agent.W / 2 + 0.6
    objs = _objects(agent, world, True)
    lx, ly, ovl = _rel(agent, objs, path, s_ag, d_ag)
    for k, o in enumerate(objs):
        rng = ahead + (40.0 if ovl[k] < -0.5 else 0.0)
        if -behind < lx[k] < rng and abs(ly[k] - shift) < half + o[3] / 2:
            return False
    return True


class PathFollow:
    """Follow a world polyline with a lateral offset that wanders (no lanes).

    events: list of dicts {trigger, d_off?, v_des?, duration?, lat_speed?}
    """

    def __init__(self, pts, v_des, d_off=0.0, wander=0.0, s0=0.0, events=None,
                 reverse=False, idm=True, seed=0, loop=False):
        pts = np.asarray(pts, float)
        if reverse:
            pts = pts[::-1]
        self.path = ReferencePath(pts, smooth=False)
        self.v_des = v_des
        self.v_des0 = v_des
        self.d = d_off
        self.d_tgt = d_off
        self.d_base = d_off
        self.lat_speed = 0.6
        self.wander = wander
        self.s = s0
        self.events = [dict(e) for e in (events or [])]
        self.idm = idm
        self.rng = np.random.default_rng(seed)
        self.ou = 0.0
        self.hold_until = None
        self.loop = loop
        self.blocked_t = 0.0
        self.last_lat_yaw = 0.0
        self.bypass_until = None
        self.bypass = 0.0

    def place(self, agent):
        x, y, th = self.path.to_cart(self.s, self.d)
        agent.x, agent.y, agent.yaw = float(x), float(y), float(th)

    def step(self, agent, world, dt):
        for ev in self.events:
            if ev.get("done"):
                continue
            if _trigger(ev.get("trigger"), world, agent):
                ev["done"] = True
                if "d_off" in ev:
                    self.d_base = ev["d_off"]
                    self.lat_speed = ev.get("lat_speed", 1.0)
                if "v_des" in ev:
                    self.v_des = ev["v_des"]
                    if "duration" in ev:
                        self.hold_until = world.t + ev["duration"]
        if self.hold_until is not None and world.t >= self.hold_until:
            self.v_des = self.v_des0
            self.hold_until = None
        # Ornstein-Uhlenbeck lateral wander (non-lane-based motion)
        if self.wander > 0:
            self.ou += -0.5 * self.ou * dt + self.wander * np.sqrt(dt) * self.rng.normal()
            self.ou = float(np.clip(self.ou, -1.5 * self.wander * 2, 1.5 * self.wander * 2))
        if self.idm:
            acc, lead = idm_accel(agent, world, self.v_des, path=self.path, s_ag=self.s, d_ag=self.d)
        else:
            acc, lead = float(np.clip((self.v_des - agent.v) * 1.0, -6, 1.5)), None
        # informal bypass of a stopped obstacle (no lanes, just go around it)
        if lead is not None and abs(lead["v"]) < 0.3 and agent.v < 0.5 and lead["gap"] < 8 \
                and self.v_des > 0.5 and self.bypass_until is None:
            self.blocked_t += dt
        else:
            self.blocked_t = 0.0
        if self.blocked_t > 1.5 and self.bypass_until is None:
            clear_ = lead["W"] / 2 + agent.W / 2 + 0.5
            rel = (lead["ly"] - clear_) if lead["ly"] >= 0 else (lead["ly"] + clear_)
            if corridor_clear(agent, world, rel, path=self.path, s_ag=self.s, d_ag=self.d):
                self.bypass = self.d + rel - self.d_base
                self.bypass_until = world.t + 8.0
        if self.bypass_until is not None:
            if world.t > self.bypass_until:
                self.bypass_until = None
                self.bypass = 0.0
            elif lead is not None and lead["gap"] > 0.8:
                acc = max(acc, 0.6 if agent.v < 1.5 else acc)
        # courtesy: squeeze away from the ego vehicle when passing close by
        courtesy = 0.0
        if agent.cls != "pedestrian":
            e = world.ego
            so, do = self.path.to_frenet(e.x, e.y)
            lx, ly = float(so[0]) - self.s, float(do[0]) - self.d
            need = (agent.W + e.W) / 2 + 0.9
            if -3 < lx < 30 and abs(ly) < need:
                courtesy = -np.sign(ly if abs(ly) > 1e-3 else 1.0) * min(need - abs(ly), 1.0)
        self.courtesy = 0.8 * getattr(self, "courtesy", 0.0) + 0.2 * courtesy
        self.d_tgt = self.d_base + self.ou + self.bypass + self.courtesy
        lat_sp = self.lat_speed if self.bypass_until is None else 0.8
        dd = np.clip(self.d_tgt - self.d, -lat_sp * dt, lat_sp * dt)
        if agent.v < 0.3 and self.bypass_until is None:
            dd = 0.0
        elif agent.v < 1.0:
            dd *= max(agent.v, 0.3)
        # last-resort safety: human drivers stop rather than drive into the ego car
        if agent.v > 0.01 or acc > 0:
            e = world.ego
            er = rect_corners(e.x, e.y, e.yaw, e.L + 0.4, e.W)
            for tau in (0.4, 0.8, 1.4):
                fx, fy, fth = self.path.to_cart(self.s + max(agent.v, 0.5) * tau,
                                                self.d + dd / dt * tau * 0.5)
                fr = rect_corners(float(fx), float(fy), float(fth) + self.last_lat_yaw,
                                  agent.L + 0.2, agent.W + 0.1)
                if rects_overlap(fr, er):
                    acc = -7.0
                    dd = 0.0
                    break
        v_new = max(0.0, agent.v + acc * dt)
        self.s += 0.5 * (agent.v + v_new) * dt
        self.d += dd
        if self.loop and self.s > self.path.length:
            self.s -= self.path.length
        if self.s > self.path.length + 5:
            agent.active = False
        x, y, th = self.path.to_cart(self.s, self.d)
        lim = 0.15 if agent.L < 6 else 0.03
        lat_yaw = float(np.clip(np.arctan2(dd / dt, max(v_new, 2.5)), -lim, lim)) if dt > 0 else 0.0
        agent.x, agent.y = float(x), float(y)
        agent.yaw = float(th + lat_yaw)
        self.last_lat_yaw = lat_yaw
        agent.v = float(v_new)


class Crosser:
    """Walk along waypoints once triggered: pedestrians darting out, cattle.

    pauses: list of (fraction_of_path, duration). turn_back: fraction at
    which the agent reverses direction (hesitating pedestrian / cattle).
    """

    def __init__(self, pts, speed, trigger=None, pauses=None, turn_back=None,
                 seed=0, jitter=0.0):
        self.pts = np.asarray(pts, float)
        seg = np.diff(self.pts, axis=0)
        self.cum = np.concatenate([[0.0], np.cumsum(np.hypot(*seg.T))])
        self.speed = speed
        self.trigger = trigger
        self.started = False
        self.s = 0.0
        self.dir = 1.0
        self.pauses = sorted(pauses or [])
        self.pause_until = None
        self.turn_back = turn_back
        self.turned = False
        self.rng = np.random.default_rng(seed)
        self.jitter = jitter

    def place(self, agent):
        p = self._at(0.0)
        agent.x, agent.y = p
        d = self.pts[1] - self.pts[0]
        agent.yaw = float(np.arctan2(d[1], d[0]))
        agent.v = 0.0

    def _at(self, s):
        return np.array([np.interp(s, self.cum, self.pts[:, 0]),
                         np.interp(s, self.cum, self.pts[:, 1])])

    def step(self, agent, world, dt):
        if not self.started:
            if _trigger(self.trigger, world, agent):
                self.started = True
            else:
                agent.v = 0.0
                return
        if self.pause_until is not None:
            if world.t < self.pause_until:
                agent.v = 0.0
                return
            self.pause_until = None
        frac = self.s / self.cum[-1]
        if self.pauses and frac >= self.pauses[0][0] and self.dir > 0:
            _, dur = self.pauses.pop(0)
            self.pause_until = world.t + dur
            agent.v = 0.0
            return
        if self.turn_back is not None and not self.turned and frac >= self.turn_back:
            self.turned = True
            self.dir = -1.0
            self.pause_until = world.t + 1.0
            agent.v = 0.0
            return
        spd = self.speed * (1 + self.jitter * self.rng.normal())
        # people and animals do not walk into a car that has already stopped
        e = world.ego
        if e.v < 1.0:
            nxt = self._at(np.clip(self.s + self.dir * 0.8, 0, self.cum[-1]))
            er = rect_corners(e.x, e.y, e.yaw, e.L + 0.4, e.W + 0.4)
            ar = rect_corners(nxt[0], nxt[1], agent.yaw, agent.L, agent.W)
            if rects_overlap(er, ar):
                agent.v = 0.0
                return
        self.s = float(np.clip(self.s + self.dir * spd * dt, 0, self.cum[-1]))
        if self.dir < 0 and self.s <= 0:
            # finished retreat: go again after a pause
            self.dir = 1.0
            self.pause_until = world.t + 2.0
        p = self._at(self.s)
        ahead = self._at(np.clip(self.s + self.dir * 0.3, 0, self.cum[-1]))
        d = ahead - p
        if np.hypot(*d) > 1e-6:
            agent.yaw = float(np.arctan2(d[1], d[0]))
        agent.x, agent.y = p
        agent.v = spd
        if self.s >= self.cum[-1] and self.dir > 0:
            agent.v = 0.0
            self.speed = 0.0


class Wander:
    """Random walk inside a disc (market pedestrians, loitering cattle)."""

    def __init__(self, center, radius, speed, seed=0, turn_sigma=1.2):
        self.c = np.asarray(center, float)
        self.r = radius
        self.speed = speed
        self.rng = np.random.default_rng(seed)
        self.ts = turn_sigma
        self.stop_until = 0.0

    def place(self, agent):
        pass

    def step(self, agent, world, dt):
        if world.t < self.stop_until:
            agent.v = 0.0
            return
        if self.rng.random() < 0.01:
            self.stop_until = world.t + self.rng.uniform(1, 4)
        agent.yaw += self.ts * np.sqrt(dt) * self.rng.normal()
        p = agent.pos + self.speed * dt * np.array([np.cos(agent.yaw), np.sin(agent.yaw)])
        # people step around a car that has stopped (they do not walk into it)
        e = world.ego
        if e.v < 1.0:
            er = rect_corners(e.x, e.y, e.yaw, e.L + 0.5, e.W + 0.5)
            ar = rect_corners(p[0], p[1], agent.yaw, agent.L, agent.W)
            if rects_overlap(er, ar):
                away = agent.pos - np.array([e.x, e.y])
                agent.yaw = float(np.arctan2(away[1], away[0]))
                agent.v = 0.0
                return
        if np.linalg.norm(p - self.c) > self.r:
            to_c = self.c - agent.pos
            agent.yaw = float(np.arctan2(to_c[1], to_c[0]) + self.rng.normal() * 0.5)
            p = agent.pos
        agent.x, agent.y = p
        agent.v = self.speed


# ---------------------------------------------------------------------------
# World container
# ---------------------------------------------------------------------------
@dataclass
class World:
    network: RoadNetwork
    agents: list
    ego: object = None
    potholes: list = field(default_factory=list)      # (x, y, r)
    occluders: list = field(default_factory=list)     # Nx2 polygons
    decor: list = field(default_factory=list)         # (polygon, color) drawing only
    t: float = 0.0

    def step(self, dt):
        for a in self.agents:
            a.step(self, dt)
        self.t += dt

    def active_agents(self):
        return [a for a in self.agents if a.active]
