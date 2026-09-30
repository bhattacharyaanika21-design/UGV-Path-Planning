"""Global route planning and blockage-triggered rerouting.

A* over the road graph. When the fused track list shows static obstacles
that leave no passable gap across the road ahead (e.g. a delivery truck
unloading next to a pushcart in a market lane), the affected edge is marked
blocked and a new route is computed from the next junction ahead of the
vehicle. This is the "Navigation Toolbox" layer of the architecture
(plannerAStarGrid / graph search in MATLAB).
"""
from __future__ import annotations

import heapq

import numpy as np

STATIC_CLASSES = {"car", "bus", "truck", "tractor", "auto", "pushcart", "bullockcart",
                  "twowheeler", None}


def astar(net, start, goal):
    h = lambda n: float(np.linalg.norm(net.nodes[n] - net.nodes[goal]))
    openq = [(h(start), 0.0, start, [start])]
    seen = set()
    while openq:
        f, g, n, path = heapq.heappop(openq)
        if n == goal:
            return path
        if n in seen:
            continue
        seen.add(n)
        for m in net.neighbors(n):
            e = net.edge(n, m)
            if e["blocked"] or m in seen:
                continue
            L = float(np.sum(np.hypot(*np.diff(e["pts"], axis=0).T)))
            g2 = g + L / e["speed"]
            heapq.heappush(openq, (g2 + h(m) / 15.0, g2, m, path + [m]))
    return None


class Router:
    def __init__(self, net, route, W_ego=1.8):
        self.net = net
        self.route = list(route)
        self.goal = route[-1]
        self.W = W_ego
        self.events = []
        self.blocked_at = None

    def node_s(self, ref):
        return {n: float(ref.to_frenet(*self.net.nodes[n])[0][0]) for n in self.route}

    def check_blockage(self, ref, s_ego, tracks, width_fn, t):
        """Return the list of blocked cross-sections ahead (or None).
        Uses confirmed tracks that have been stationary for > 2.5 s."""
        cands = []
        for tr in tracks:
            if tr.cls not in STATIC_CLASSES or tr.speed > 0.7 or t - tr.static_since < 2.0:
                continue
            s, d = ref.to_frenet(tr.pos[0], tr.pos[1])
            s, d = float(s[0]), float(d[0])
            if not (4.0 < s - s_ego < 70.0):
                continue
            L, W = tr.dims
            _, _, th, _ = ref.interp(s)
            rel = tr.yaw - th
            wd = abs(L * np.sin(rel)) + abs(W * np.cos(rel))
            ld = abs(L * np.cos(rel)) + abs(W * np.sin(rel))
            cands.append((s, d, wd, ld))
        if not cands:
            return None
        cands.sort()
        blocked = []
        for s_i, _, _, ld_i in cands:
            group = [c for c in cands if abs(c[0] - s_i) < (ld_i + c[3]) / 2 + 2.0]
            lo, hi = width_fn(np.array([s_i]))
            lo, hi = float(lo[0]), float(hi[0])
            ivs = sorted((c[1] - c[2] / 2, c[1] + c[2] / 2) for c in group)
            free, cur = 0.0, lo
            for a, b in ivs:
                free = max(free, a - cur)
                cur = max(cur, b)
            free = max(free, hi - cur)
            if free < self.W + 0.6:
                blocked.append(s_i)
        return blocked or None

    def reroute(self, ref, s_ego, s_block, t):
        ns = self.node_s(ref)
        # edge containing the blockage
        x, y, _ = ref.to_cart(s_block, 0.0)
        ek, _ = self.net.edge_of_point(np.array([float(x), float(y)]))
        if ek is None:
            return None
        # next junction ahead of us but before the blockage
        options = [n for n in self.route if s_ego + 9.0 < ns[n] < s_block]
        if not options:
            # maybe we are exactly on the edge after the last node: none possible
            return None
        self.net.edges[ek]["blocked"] = True
        for n in reversed(options):
            sub = astar(self.net, n, self.goal)
            if sub:
                idx = self.route.index(n)
                new_route = self.route[:idx] + sub
                self.events.append(dict(t=t, blocked_edge=ek, via=n, route=new_route))
                self.route = new_route
                return new_route
        self.net.edges[ek]["blocked"] = False
        return None
