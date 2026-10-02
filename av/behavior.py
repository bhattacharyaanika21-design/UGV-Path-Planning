"""Decision logic — a Stateflow-style finite-state machine.

States: CRUISE, FOLLOW, NUDGE (lateral offset / overtake), YIELD (VRU or
junction caution), OCCLUSION_CAUTION, WAIT (stopped, blocked), EMERGENCY,
REROUTE. The FSM does not steer the car directly; it shapes the planner's
objective (desired speed, preferred lateral offset) and triggers global
rerouting. Situational speed caps implement defensive Indian-road driving:
  * slow down near unsignalised junctions ("creep and look");
  * slow down when passing a stopped bus/truck that may hide a pedestrian;
  * slow dowm near pedestrians/cattle close to the predicted path.
"""
from __future__ import annotations

import numpy as np

VRU = ("pedestrian", "cattle", "bicycle")
LARGE = ("bus", "truck", "tractor")


class BehaviorFSM:
    def __init__(self, zone_speed_fn, d_pref_fn, junction_s=(), junction_cap=5.0):
        self.state = "CRUISE"
        self.zone_speed_fn = zone_speed_fn
        self.d_pref_fn = d_pref_fn
        self.junction_s = list(junction_s)
        self.junction_cap = junction_cap
        self.wait_since = None
        self.reroute_until = -1.0
        self.history = []
        self.caps = {}

    def update(self, t, ego, s_ego, d_ego, ref, tracks, last_plan):
        v_zone = float(self.zone_speed_fn(s_ego))
        v_des = v_zone
        caps = {}
        # junction creep
        for sj in self.junction_s:
            if -6 < sj - s_ego < 25:
                caps["junction"] = self.junction_cap
        # occlusion & VRU caution
        lead = None
        for tr in tracks:
            s, d = ref.to_frenet(tr.pos[0], tr.pos[1])
            ds, dd = float(s[0]) - s_ego, float(d[0])
            if tr.cls in LARGE and tr.speed < 0.5 and 0 < ds < 30 and abs(dd - d_ego) < 5.0:
                caps["occlusion"] = min(caps.get("occlusion", 99), 4.5)
            if tr.cls in VRU and -3 < ds < 35 and abs(dd - d_ego) < 4.5:
                cap = max(3.0, 2.0 + 0.3 * max(ds, 0))
                caps["vru"] = min(caps.get("vru", 99), cap)
            if 0 < ds < 40 and abs(dd - d_ego) < 1.8 and (lead is None or ds < lead[0]):
                lead = (ds, tr)
        for c in caps.values():
            v_des = min(v_des, c)
        self.caps = caps
        d_pref = float(self.d_pref_fn(s_ego))

        # --- state transitions (for logging / downstream use) -----------
        prev = self.state
        if last_plan is not None and last_plan.emergency:
            st = "EMERGENCY"
        elif t < self.reroute_until:
            st = "REROUTE"
        elif ego.v < 0.3 and (last_plan is not None and last_plan.vT < 0.1):
            st = "WAIT"
        elif last_plan is not None and abs(last_plan.dT - d_pref) > 1.0:
            st = "NUDGE"
        elif "vru" in caps or "junction" in caps:
            st = "YIELD"
        elif "occlusion" in caps:
            st = "OCCLUSION_CAUTION"
        elif lead is not None and lead[1].speed < v_des - 1.0:
            st = "FOLLOW"
        else:
            st = "CRUISE"
        # standstill timer (robust to state flicker while stopped)
        if ego.v < 0.3:
            self.wait_since = t if self.wait_since is None else self.wait_since
        else:
            self.wait_since = None
        self.state = st
        if st != prev:
            self.history.append((t, st))
        return v_des, d_pref

    def waiting_for(self, t):
        return 0.0 if self.wait_since is None else t - self.wait_since

    def mark_reroute(self, t):
        self.reroute_until = t + 2.0
        self.state = "REROUTE"
        self.history.append((t, "REROUTE"))
