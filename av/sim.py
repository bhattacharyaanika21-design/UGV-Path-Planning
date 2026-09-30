"""Closed-loop simulation: world -> sensors -> fusion -> prediction ->
behaviour FSM / router -> lattice planner -> controller -> vehicle -> world.

Rates: world + control 20 Hz, perception 10 Hz, planning 5 Hz periodic
plus event-triggered replanning at 10 Hz whenever (a) a new object is
confirmed near the path, (b) the current plan is invalidated by fresh
predictions, or (c) the global route changes.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .behavior import BehaviorFSM
from .planner import LatticePlanner
from .prediction import predict_all
from .router import Router
from .sensors import SensorSuite
from .tracker import MultiObjectTracker, Track
from .vehicle import EgoVehicle, TrackingController
from .world import rect_distance, rects_overlap

DT = 0.05


@dataclass
class ScenarioSpec:
    key: str
    title: str
    description: str
    world: object
    route: list
    ego_init: tuple
    zone_speed: object = 10.0            # float or fn(s)
    keep_left: bool = True
    d_pref: object = None                # float or fn(s); default left of centre
    t_max: float = 60.0
    junctions: list = field(default_factory=list)
    junction_cap: float = 5.0
    goal_margin: float = 6.0
    view: float = 38.0


class RouteWidth:
    """Drivable bounds (d_min, d_max) along the current reference.

    Found by casting along the normal of the reference at every sample and
    keeping the contiguous stretch that lies on the route's road surface
    (edge rectangles + junction areas). This handles curves and filleted
    junction turns correctly. A slowly varying road-edge estimation error is
    added because there are no painted edges to detect.
    """

    def __init__(self, net, route, ref, rng, max_extra=0.6):
        segs, widths = [], []
        for u, v in zip(route[:-1], route[1:]):
            e = net.edge(u, v)
            P = e["pts"]
            for a, b in zip(P[:-1], P[1:]):
                segs.append((a, b))
                widths.append(e["width"])
        nodes = []
        for n in route:
            ws = [net.edge(n, m)["width"] for m in net.neighbors(n)]
            nodes.append((net.nodes[n], max(ws) / 2))
        offs = np.arange(-9.0, 9.01, 0.1)
        xr, yr, th, _ = ref.interp(ref.s)
        nx, ny = -np.sin(th), np.cos(th)
        PX = xr[:, None] + offs[None, :] * nx[:, None]
        PY = yr[:, None] + offs[None, :] * ny[:, None]
        inside = np.zeros(PX.shape, bool)
        wmax = np.zeros(len(ref.s))
        for (a, b), w in zip(segs, widths):
            ab = b - a
            t = np.clip(((PX - a[0]) * ab[0] + (PY - a[1]) * ab[1]) / max(ab @ ab, 1e-9), 0, 1)
            dx = PX - (a[0] + t * ab[0])
            dy = PY - (a[1] + t * ab[1])
            inside |= (dx * dx + dy * dy) <= (w / 2) ** 2
            wmax = np.maximum(wmax, w)
        for c, r in nodes:
            inside |= (PX - c[0]) ** 2 + (PY - c[1]) ** 2 <= r ** 2
        zero = len(offs) // 2
        lo = np.zeros(len(ref.s))
        hi = np.zeros(len(ref.s))
        for i in range(len(ref.s)):
            row = inside[i]
            if not row[zero]:
                # reference slightly off the surface: fall back to nominal width
                lo[i], hi[i] = -wmax[i] / 2, wmax[i] / 2
                continue
            j = zero
            while j + 1 < len(offs) and row[j + 1]:
                j += 1
            k = zero
            while k - 1 >= 0 and row[k - 1]:
                k -= 1
            hi[i] = min(offs[j], wmax[i] / 2 + max_extra)
            lo[i] = max(offs[k], -wmax[i] / 2 - max_extra)
        n = len(ref.s)
        noise = np.convolve(rng.normal(0, 0.12, n + 40), np.ones(41) / 41 * 4.0, "valid")[:n]
        self.s = ref.s
        self.lo = lo + np.abs(noise) * 0.5
        self.hi = hi - np.abs(noise[::-1]) * 0.5
        self.w = hi - lo

    def __call__(self, s):
        s = np.asarray(s)
        return np.interp(s, self.s, self.lo), np.interp(s, self.s, self.hi)


def oncoming_sight(ref, s_ego, width, ego, tracks, occluders, max_d=80.0):
    """How far ahead can we see along the oncoming half of the road?

    Rays from the ego to points on the oncoming side (quarter width right of
    centre) are tested against tracked objects (as rectangles) and mapped
    roadside occluders. Returns the first blocked distance."""
    from .sensors import _segments_blocked
    from .world import rect_corners
    edges, owners = [], []
    for tr in tracks:
        L, W = tr.dims
        c = rect_corners(tr.pos[0], tr.pos[1], tr.yaw, L, W)
        for i in range(4):
            edges.append([c[i], c[(i + 1) % 4]])
            owners.append(tr.id)
    for k, poly in enumerate(occluders):
        P = np.asarray(poly)
        for i in range(len(P)):
            edges.append([P[i], P[(i + 1) % len(P)]])
            owners.append(-1000 - k)
    if not edges:
        return max_d
    edges = np.array(edges)
    owners = np.array(owners)
    ds = np.arange(8.0, max_d + 0.1, 4.0)
    lo, _ = width(s_ego + ds)
    x, y, _ = ref.to_cart(s_ego + ds, lo / 2)
    tg = np.column_stack([x, y])
    org = np.repeat(np.array([[ego.x, ego.y]]), len(tg), axis=0)
    blocked = _segments_blocked(org, tg, edges, owners, np.full(len(tg), -1))
    if not blocked.any():
        return max_d
    return float(ds[np.argmax(blocked)])


def _as_fn(v, default):
    if v is None:
        return default
    if callable(v):
        return v
    return lambda s, v=v: v


def run(spec: ScenarioSpec, seed=0, record=True, verbose=False, debug_at=None):
    rng = np.random.default_rng(seed + 1000)
    Track._next = 1
    world = spec.world
    net = world.network
    x0, y0, yaw0, v0 = spec.ego_init
    ego = EgoVehicle(x0, y0, yaw0, v0)
    world.ego = ego
    for a in world.agents:
        if a.behavior is not None and hasattr(a.behavior, "place"):
            a.behavior.place(a)

    router = Router(net, spec.route)
    ref = net.route_path(router.route)
    width = RouteWidth(net, router.route, ref, rng)
    d_pref_fn = _as_fn(spec.d_pref, None)
    if d_pref_fn is None:
        d_pref_fn = lambda s: float(np.clip(np.mean(width(s)[1]) / 2.0, 0.0, 2.2))

    def junction_s(r, route):
        return [float(r.to_frenet(*net.nodes[n])[0][0]) for n in spec.junctions if n in route]

    fsm = BehaviorFSM(_as_fn(spec.zone_speed, None), d_pref_fn,
                      junction_s(ref, router.route), spec.junction_cap)
    sensors = SensorSuite(seed=seed)
    tracker = MultiObjectTracker()
    planner = LatticePlanner(ego.L, ego.W)
    ctrl = TrackingController()

    pothole_map = []
    plan, t_plan = None, 0.0
    hyps = []
    log = dict(t=[], x=[], y=[], yaw=[], v=[], a=[], delta=[], clearance=[], ttc=[],
               state=[], v_des=[])
    plans, reactions, frames, events = [], [], [], []
    track_err = []
    collision = None
    success = False
    potholes_hit = set()
    tried_blocks = []
    creep_until = -1.0
    n_steps = int(spec.t_max / DT)
    ref_version = 0

    for n in range(n_steps):
        t = world.t
        replan_reasons = []
        if n % 2 == 0:
            dets, ph = sensors.sense(world, ego)
            tracker.step(dets, t)
            for (px, py, pr) in ph:
                if all(np.hypot(px - q[0], py - q[1]) > 1.0 for q in pothole_map):
                    pothole_map.append((px, py, pr))
            conf = tracker.confirmed()
            hyps = predict_all(conf, ref)
            s_arr, d_arr = ref.to_frenet(ego.x, ego.y)
            s_ego, d_ego = float(s_arr[0]), float(d_arr[0])

            # ---- global rerouting on blockage -------------------------
            blocks = router.check_blockage(ref, s_ego, conf, width, t) or []
            new = None
            for s_block in blocks:
                if any(abs(s_block - q) < 5 for q in tried_blocks):
                    continue
                tried_blocks.append(s_block)
                new = router.reroute(ref, s_ego, s_block, t)
                if new is None:
                    events.append((t, "blocked_no_alt", f"s={s_block:.0f}"))
                    continue
                if new is not None:
                    ref = net.route_path(new)
                    ref_version += 1
                    width = RouteWidth(net, new, ref, rng)
                    fsm.junction_s = junction_s(ref, new)
                    fsm.mark_reroute(t)
                    hyps = predict_all(conf, ref)
                    s_arr, d_arr = ref.to_frenet(ego.x, ego.y)
                    s_ego, d_ego = float(s_arr[0]), float(d_arr[0])
                    replan_reasons.append("reroute")
                    events.append((t, "reroute", " -> ".join(new)))
                    if verbose:
                        print(f"  t={t:5.1f}s REROUTE via {new}")
                    tried_blocks = []
                    break

            v_des, d_pref = fsm.update(t, ego, s_ego, d_ego, ref, conf, plan)

            new_near = []
            for tr in tracker.new_confirmed:
                if np.hypot(*(tr.pos - ego.pos)) < 70:
                    new_near.append(tr)
            if new_near:
                replan_reasons.append("new_object")
            if plan is not None and not planner.still_valid(plan, hyps, t - t_plan):
                replan_reasons.append("plan_invalid")
            if plan is None or t - t_plan >= 0.2 - 1e-9:
                replan_reasons.append("periodic")
            if replan_reasons:
                ph_fr = []
                for (px, py, pr) in pothole_map:
                    ps, pd = ref.to_frenet(px, py)
                    if -5 < ps[0] - s_ego < 60:
                        ph_fr.append((float(ps[0]), float(pd[0]), pr))
                center_fn = lambda s: 0.0
                sight = oncoming_sight(ref, s_ego, width, ego, conf, world.occluders) \
                    if spec.keep_left else np.inf
                if debug_at is not None and t >= debug_at:
                    return dict(ref=ref, ego=ego, width=width, hyps=hyps, ph=ph_fr, v_des=v_des,
                                d_pref=d_pref, plan=plan, planner=planner, spec=spec,
                                tracks=conf, world=world)
                prev_plan = plan if "reroute" not in replan_reasons else None
                # "honk and creep": after a long wait behind loitering people or
                # animals, inch forward at walking pace with a tighter envelope
                # around them (they step aside for a slowly moving car)
                plan_hyps = hyps
                if fsm.waiting_for(t) > 6.0 or creep_until > t:
                    if fsm.waiting_for(t) > 6.0:
                        creep_until = t + 4.0
                    plan_hyps = []
                    for h in hyps:
                        if h.cls in ("pedestrian", "cattle"):
                            h = copy.copy(h)
                            h.radius = h.base_radius + 0.25 * (h.radius - h.base_radius)
                            h.prob = 0.1 * h.prob
                        plan_hyps.append(h)
                    v_des = min(v_des, 1.2)
                plan = planner.plan(ref, ego, width, plan_hyps, ph_fr, v_des, d_pref, prev_plan,
                                    keep_left=spec.keep_left, center_fn=center_fn, t_now=t,
                                    sight=sight)
                plan.t0 = t
                t_plan = t
                plans.append(dict(t=t, ms=plan.compute_ms, reasons=list(replan_reasons),
                                  feasible=plan.feasible, emergency=plan.emergency,
                                  n_feas=plan.n_feas, n_cand=plan.n_cand, vT=plan.vT, dT=plan.dT))
                for tr in new_near:
                    reactions.append(dict(track=tr.id, cls=tr.cls, t_first=tr.t_first,
                                          t_confirm=tr.t_confirm,
                                          latency=(tr.t_confirm - tr.t_first) + plan.compute_ms / 1000))
            # tracking accuracy vs ground truth
            gt = {a.id: a for a in world.active_agents()}
            for tr in conf:
                a = gt.get(tr.true_id)
                if a is not None:
                    track_err.append(float(np.hypot(*(tr.pos - a.pos))))

        # ---- control + dynamics ------------------------------------------
        a_cmd, delta = ctrl.command(ego, plan, t - t_plan)
        ego.step(a_cmd, delta, DT)
        world.step(DT)

        # ---- ground-truth safety metrics ---------------------------------
        ec = ego.corners()
        clear, ttc = np.inf, np.inf
        for a in world.active_agents():
            cd = np.hypot(a.x - ego.x, a.y - ego.y)
            if cd > 25:
                continue
            ac = a.corners()
            if cd < 8 and rects_overlap(ec, ac):
                clear = 0.0
                if collision is None:
                    collision = dict(t=world.t, agent=a.id, cls=a.cls,
                                     ego_speed=round(float(ego.v), 2))
                continue
            dist = rect_distance(ec, ac) if cd < 12 else cd - (a.L + ego.L) / 2
            clear = min(clear, dist)
            rel = a.pos - ego.pos
            relv = a.vel - np.array([ego.v * np.cos(ego.yaw), ego.v * np.sin(ego.yaw)])
            closing = -np.dot(rel, relv) / max(cd, 1e-6)
            fwd = np.dot(rel, [np.cos(ego.yaw), np.sin(ego.yaw)])
            lat = abs(np.cos(ego.yaw) * rel[1] - np.sin(ego.yaw) * rel[0])  # 2-D cross product
            if closing > 0.3 and fwd > 0 and lat < (a.W + ego.W) / 2 + 0.5:
                ttc = min(ttc, max(dist, 0) / closing)
        # pothole hits (wheel over true pothole)
        cy, sy = np.cos(ego.yaw), np.sin(ego.yaw)
        for pi_, (px, py, pr) in enumerate(world.potholes):
            for side in (-0.8, 0.8):
                for axle in (1.35, -1.35):
                    wx = ego.x + axle * cy - side * sy
                    wy = ego.y + axle * sy + side * cy
                    if np.hypot(wx - px, wy - py) < pr:
                        potholes_hit.add(pi_)

        log["t"].append(world.t); log["x"].append(ego.x); log["y"].append(ego.y)
        log["yaw"].append(ego.yaw); log["v"].append(ego.v); log["a"].append(ego.a)
        log["delta"].append(ego.delta); log["clearance"].append(clear); log["ttc"].append(ttc)
        log["state"].append(fsm.state); log["v_des"].append(v_des)

        if record and n % 2 == 0:
            frames.append(dict(
                t=world.t, ego=(ego.x, ego.y, ego.yaw, ego.v, ego.a, ego.delta),
                agents=[(a.corners(), a.cls) for a in world.active_agents()],
                actors=[(a.id, a.cls, a.x, a.y, a.yaw, a.v, a.L, a.W) for a in world.active_agents()],
                tracks=[(tr.pos.copy(), tr.cls, tr.id, tr.vel.copy()) for tr in conf],
                hard=[h.centers[::2, 0, :].copy() for h in hyps if h.hard and h.cls is not None],
                soft=[h.centers[::2, 0, :].copy() for h in hyps if not h.hard],
                plan=np.column_stack([plan.x, plan.y]), alt=plan.alt_xy[:8],
                emergency=plan.emergency, state=fsm.state, v_des=v_des, caps=dict(fsm.caps),
                potholes=list(pothole_map), ref_version=ref_version,
                ref_xy=ref.xy if (not frames or frames[-1]["ref_version"] != ref_version) else None,
                clearance=clear))
        if collision is not None:
            break
        s_now = float(ref.to_frenet(ego.x, ego.y)[0][0])
        if s_now >= ref.length - spec.goal_margin:
            success = True
            break

    metrics = compute_metrics(spec, log, plans, reactions, track_err, collision, success,
                              potholes_hit, router, fsm, events)
    return dict(metrics=metrics, frames=frames, log=log, plans=plans, events=events,
                spec=spec, ref=ref, route=router.route)


def compute_metrics(spec, log, plans, reactions, track_err, collision, success,
                    potholes_hit, router, fsm, events):
    t = np.array(log["t"])
    v = np.array(log["v"])
    a = np.array(log["a"])
    delta = np.array(log["delta"])
    jerk = np.diff(a) / DT if len(a) > 1 else np.zeros(1)
    kappa = np.tan(delta) / 2.7
    alat = v ** 2 * kappa
    jlat = np.diff(alat) / DT if len(alat) > 1 else np.zeros(1)
    steer_rate = np.diff(delta) / DT if len(delta) > 1 else np.zeros(1)
    ms = np.array([p["ms"] for p in plans]) if plans else np.zeros(1)
    reasons = {}
    for p in plans:
        for r in p["reasons"]:
            reasons[r] = reasons.get(r, 0) + 1
    lat = np.array([r["latency"] for r in reactions]) if reactions else np.array([np.nan])
    clear = np.array(log["clearance"])
    ttc = np.array(log["ttc"])
    dist = float(np.sum(v) * DT)
    return dict(
        scenario=spec.key, title=spec.title,
        completed=bool(success and collision is None),
        collision=collision is not None, collision_info=collision,
        time_s=float(t[-1]) if len(t) else 0.0, distance_m=dist,
        avg_speed_mps=float(dist / max(t[-1], 1e-6)) if len(t) else 0.0,
        min_clearance_m=float(np.min(clear)) if len(clear) else np.inf,
        min_ttc_s=float(np.min(ttc)) if len(ttc) else np.inf,
        replans=len(plans), replan_reasons=reasons,
        replan_latency_ms_mean=float(ms.mean()), replan_latency_ms_p95=float(np.percentile(ms, 95)),
        replan_latency_ms_max=float(ms.max()),
        reaction_latency_s_mean=float(np.nanmean(lat)) if np.isfinite(lat).any() else None,
        reaction_latency_s_max=float(np.nanmax(lat)) if np.isfinite(lat).any() else None,
        rms_long_jerk=float(np.sqrt(np.mean(jerk ** 2))),
        rms_lat_jerk=float(np.sqrt(np.mean(jlat ** 2))),
        max_lat_acc=float(np.max(np.abs(alat))), max_decel=float(-np.min(a)),
        rms_steer_rate_dps=float(np.rad2deg(np.sqrt(np.mean(steer_rate ** 2)))),
        emergency_plans=int(sum(p["emergency"] for p in plans)),
        potholes_total=len(spec.world.potholes), potholes_hit=len(potholes_hit),
        track_rmse_m=float(np.sqrt(np.mean(np.square(track_err)))) if track_err else None,
        reroutes=len(router.events),
        states_visited=sorted(set(log["state"])),
        events=[(round(e[0], 2), e[1], e[2]) for e in events],
    )
