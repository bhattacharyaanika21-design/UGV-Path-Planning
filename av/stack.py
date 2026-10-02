"""AutonomyStack: the full perception -> prediction -> decision -> planning ->
control pipeline as a reusable object with two entry points:

    stack.cycle(t, world)   # 10 Hz: sense, track, predict, route, decide, (re)plan
    stack.control(t)        # 20 Hz: tracking controller -> (a_cmd, steer, v_ref)
It can be driven by the pure-Python simulator, by a
ROS 2 node fed from Gazebo, or by a Simulink bridge.
"""
from __future__ import annotations

import copy

import numpy as np

from .behavior import BehaviorFSM
from .planner import LatticePlanner
from .prediction import predict_all
from .router import Router
from .sensors import SensorSuite
from .sim import RouteWidth, _as_fn, oncoming_sight
from .tracker import MultiObjectTracker, Track
from .vehicle import EgoVehicle, TrackingController


class AutonomyStack:
    def __init__(self, spec, seed=0, ego=None):
        self.spec = spec
        self.rng = np.random.default_rng(seed + 1000)
        Track._next = 1
        net = spec.world.network
        self.net = net
        x0, y0, yaw0, _ = spec.ego_init
        self.ego = ego if ego is not None else EgoVehicle(x0, y0, yaw0, 0.0)
        self.router = Router(net, spec.route)
        self.ref = net.route_path(self.router.route)
        self.width = RouteWidth(net, self.router.route, self.ref, self.rng)
        d_pref_fn = _as_fn(spec.d_pref, None)
        if d_pref_fn is None:
            d_pref_fn = lambda s: float(np.clip(np.mean(self.width(s)[1]) / 2.0, 0.0, 2.2))
        self.fsm = BehaviorFSM(_as_fn(spec.zone_speed, None), d_pref_fn,
                               self._junction_s(), spec.junction_cap)
        self.sensors = SensorSuite(seed=seed)
        self.tracker = MultiObjectTracker()
        self.planner = LatticePlanner(self.ego.L, self.ego.W)
        self.ctrl = TrackingController()
        self.pothole_map = []
        self.plan, self.t_plan = None, 0.0
        self.hyps, self.conf = [], []
        self.tried_blocks = []
        self.creep_until = -1.0
        self.v_des = 0.0
        self.events = []
        self.plans = []
        self.ref_version = 0
        self.goal_reached = False

    # ------------------------------------------------------------------
    def _junction_s(self):
        return [float(self.ref.to_frenet(*self.net.nodes[n])[0][0])
                for n in self.spec.junctions if n in self.router.route]

    def frenet(self):
        s, d = self.ref.to_frenet(self.ego.x, self.ego.y)
        return float(s[0]), float(d[0])

    # ------------------------------------------------------------------
    def cycle(self, t, world):
        """One perception + planning cycle (call at ~10 Hz)."""
        ego = self.ego
        world.ego = ego
        dets, ph = self.sensors.sense(world, ego)
        self.tracker.step(dets, t)
        for (px, py, pr) in ph:
            if all(np.hypot(px - q[0], py - q[1]) > 1.0 for q in self.pothole_map):
                self.pothole_map.append((px, py, pr))
        conf = self.tracker.confirmed()
        self.conf = conf
        self.hyps = predict_all(conf, self.ref)
        s_ego, d_ego = self.frenet()
        reasons = []

        # --- global rerouting on blockage -------------------------------
        for s_block in (self.router.check_blockage(self.ref, s_ego, conf, self.width, t) or []):
            if any(abs(s_block - q) < 5 for q in self.tried_blocks):
                continue
            self.tried_blocks.append(s_block)
            new = self.router.reroute(self.ref, s_ego, s_block, t)
            if new is None:
                self.events.append((t, "blocked_no_alt", f"s={s_block:.0f}"))
                continue
            self.ref = self.net.route_path(new)
            self.ref_version += 1
            self.width = RouteWidth(self.net, new, self.ref, self.rng)
            self.fsm.junction_s = self._junction_s()
            self.fsm.mark_reroute(t)
            self.hyps = predict_all(conf, self.ref)
            s_ego, d_ego = self.frenet()
            reasons.append("reroute")
            self.events.append((t, "reroute", " -> ".join(new)))
            self.tried_blocks = []
            break

        v_des, d_pref = self.fsm.update(t, ego, s_ego, d_ego, self.ref, conf, self.plan)
        new_near = [tr for tr in self.tracker.new_confirmed if np.hypot(*(tr.pos - ego.pos)) < 70]
        if new_near:
            reasons.append("new_object")
        if self.plan is not None and not self.planner.still_valid(self.plan, self.hyps, t - self.t_plan):
            reasons.append("plan_invalid")
        if self.plan is None or t - self.t_plan >= 0.2 - 1e-6:
            reasons.append("periodic")

        if reasons:
            ph_fr = []
            for (px, py, pr) in self.pothole_map:
                ps, pd = self.ref.to_frenet(px, py)
                if -5 < ps[0] - s_ego < 60:
                    ph_fr.append((float(ps[0]), float(pd[0]), pr))
            sight = oncoming_sight(self.ref, s_ego, self.width, ego, conf, world.occluders) \
                if self.spec.keep_left else np.inf
            plan_hyps = self.hyps
            if self.fsm.waiting_for(t) > 6.0 or self.creep_until > t:
                if self.fsm.waiting_for(t) > 6.0:
                    self.creep_until = t + 4.0
                plan_hyps = []
                for h in self.hyps:
                    if h.cls in ("pedestrian", "cattle"):
                        h = copy.copy(h)
                        h.radius = h.base_radius + 0.25 * (h.radius - h.base_radius)
                        h.prob = 0.1 * h.prob
                    plan_hyps.append(h)
                v_des = min(v_des, 1.2)
            prev = self.plan if "reroute" not in reasons else None
            self.plan = self.planner.plan(self.ref, ego, self.width, plan_hyps, ph_fr, v_des, d_pref,
                                          prev, keep_left=self.spec.keep_left,
                                          center_fn=lambda s: 0.0, t_now=t, sight=sight)
            self.plan.t0 = t
            self.t_plan = t
            self.plans.append(dict(t=t, ms=self.plan.compute_ms, reasons=list(reasons),
                                   emergency=self.plan.emergency, n_feas=self.plan.n_feas,
                                   n_cand=self.plan.n_cand, vT=self.plan.vT, dT=self.plan.dT))
        self.v_des = v_des
        if s_ego >= self.ref.length - self.spec.goal_margin:
            self.goal_reached = True
        return reasons

    # ------------------------------------------------------------------
    def control(self, t):
        """Tracking controller (call at ~20 Hz). Returns (a_cmd, steer, v_ref)."""
        if self.plan is None:
            return -2.0, 0.0, 0.0
        if self.goal_reached:
            return -3.0, 0.0, 0.0
        a_cmd, delta = self.ctrl.command(self.ego, self.plan, t - self.t_plan)
        v_ref = float(np.interp(t - self.t_plan + 0.15, self.plan.t, self.plan.v))
        return a_cmd, delta, v_ref
