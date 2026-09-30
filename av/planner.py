"""Adaptive local planner: sampling-based Frenet lattice with prediction-aware
collision checking.

There are no lanes: lateral targets are sampled across the *whole* drivable
width at 0.5 m resolution, so the car can squeeze past a pushcart, move to
the road edge for an oncoming bus, or swing wide around cattle. Every
candidate is a (lateral path d(s)) x (speed profile s(t)) pair; it is
rejected if it leaves the road, exceeds curvature / lateral-acceleration
limits, or comes closer than a time-growing safety margin to any *hard*
predicted occupancy in the next 3.5 s. Surviving candidates are ranked by
comfort (jerk), progress, clearance to all hypotheses (incl. soft ones such
as "pedestrian may dart"), pothole exposure, road-edge proximity and
consistency with the previous plan. If nothing is feasible the planner
returns a minimum-risk manoeuvre (hardest braking with the least-violating
steer) and flags an emergency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .prediction import TGRID

K = len(TGRID)
DT = TGRID[1] - TGRID[0]
HARD_T = 3.5


def quintic(d0, d1, d2, dT, L):
    """Coefficients of d(x)=sum c_i x^i with end state (dT,0,0) at x=L."""
    L = np.asarray(L, float)
    A0, A1, A2 = d0, d1, d2 / 2.0
    L2, L3, L4, L5 = L ** 2, L ** 3, L ** 4, L ** 5
    b0 = dT - (A0 + A1 * L + A2 * L2)
    b1 = -(A1 + 2 * A2 * L)
    b2 = -(2 * A2)
    A3 = (10 * b0 / L3) - (4 * b1 / L2) + (b2 / (2 * L))
    A4 = (-15 * b0 / L4) + (7 * b1 / L3) - (b2 / L2)
    A5 = (6 * b0 / L5) - (3 * b1 / L4) + (b2 / (2 * L3))
    return np.stack(np.broadcast_arrays(A0, A1, A2, A3, A4, A5), -1)


def eval_quintic(c, x, L):
    """c:(N,6) x:(..., N-broadcast) -> d, d', d'' clamped to end state."""
    xc = np.minimum(x, L)
    c0, c1, c2, c3, c4, c5 = [c[..., i] for i in range(6)]
    d = c0 + c1 * xc + c2 * xc ** 2 + c3 * xc ** 3 + c4 * xc ** 4 + c5 * xc ** 5
    d1 = c1 + 2 * c2 * xc + 3 * c3 * xc ** 2 + 4 * c4 * xc ** 3 + 5 * c5 * xc ** 4
    d2 = 2 * c2 + 6 * c3 * xc + 12 * c4 * xc ** 2 + 20 * c5 * xc ** 3
    past = x >= L
    d1 = np.where(past, 0.0, d1)
    d2 = np.where(past, 0.0, d2)
    return d, d1, d2


@dataclass
class PlanResult:
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    a: np.ndarray
    s: np.ndarray
    d: np.ndarray
    kappa: np.ndarray
    path_xy: np.ndarray
    feasible: bool
    emergency: bool
    dT: float
    vT: float
    n_cand: int
    n_feas: int
    compute_ms: float
    min_gap: float
    cost: float = 0.0
    alt_xy: list = field(default_factory=list)
    t0: float = 0.0
    dp: np.ndarray = None
    dpp: np.ndarray = None


class LatticePlanner:
    def __init__(self, L=4.4, W=1.8, wheelbase=2.7):
        self.L, self.W = L, W
        self.kappa_max = np.tan(np.deg2rad(33)) / wheelbase
        self.alat_max = 3.2
        self.a_max = 2.5
        self.a_min = -7.5
        n = 5
        step = L / n
        self.ego_offs = (np.arange(n) - (n - 1) / 2) * step
        self.ego_r = float(np.hypot(step / 2, W / 2))
        self.w = dict(lat_off=0.6, lat_jerk=40.0, speed=3.0, lon_jerk=0.08, prox=6.0,
                      soft=25.0, late=40.0, pothole=25.0, edge=4.0, consist=0.8,
                      side=0.8, brake=0.5, block=60.0, blind=150.0)

    # ------------------------------------------------------------------
    def _lon_profiles(self, v0, a0, v_des):
        v0 = max(v0, 0.0)
        a0 = float(np.clip(a0, -7.0, 2.5))
        profiles = []
        targets = sorted(set([0.0, 0.25 * v_des, 0.5 * v_des, 0.75 * v_des, v_des,
                              min(v_des, v0), min(v_des, v0 + 1.5)]))
        for vT in targets:
            for T in (2.0, 4.0):
                M = np.array([[3 * T ** 2, 4 * T ** 3], [6 * T, 12 * T ** 2]])
                rhs = np.array([vT - v0 - a0 * T, -a0])
                c3, c4 = np.linalg.solve(M, rhs)
                t = TGRID
                v = np.where(t <= T, v0 + a0 * t + 3 * c3 * t ** 2 + 4 * c4 * t ** 3, vT)
                profiles.append(("keep", vT, T, np.maximum(v, 0.0)))
        for ab in (-2.0, -4.0, -7.0):
            # brake ramp with jerk limit ~ 15 m/s^3 from a0
            a = np.maximum(ab, a0 - 15 * TGRID) if a0 > ab else np.full(K, ab)
            v = np.maximum(v0 + np.concatenate([[0], np.cumsum(0.5 * (a[1:] + a[:-1]) * DT)]), 0)
            profiles.append(("brake", 0.0, -ab, v))
        V = np.array([p[3] for p in profiles])                      # (Nv, K)
        S = np.concatenate([np.zeros((len(V), 1)),
                            np.cumsum(0.5 * (V[:, 1:] + V[:, :-1]) * DT, axis=1)], 1)
        A = np.gradient(V, DT, axis=1)
        J = np.gradient(A, DT, axis=1)
        return profiles, V, S, A, J

    # ------------------------------------------------------------------
    def plan(self, ref, ego, width_fn, hyps, potholes, v_des, d_pref,
             prev=None, keep_left=True, center_fn=None, t_now=None, sight=np.inf):
        tic = time.perf_counter()
        s0, d0 = ref.to_frenet(ego.x, ego.y)
        s0, d0 = float(s0[0]), float(d0[0])
        _, _, th0, k0 = ref.interp(s0)
        dth = np.arctan2(np.sin(ego.yaw - th0), np.cos(ego.yaw - th0))
        d1 = float(np.tan(np.clip(dth, -1.2, 1.2)) * (1 - k0 * d0))
        d2 = 0.0
        a0 = ego.a if ego.v > 0.3 else max(ego.a, 0.0)
        # plan continuity: start from the previous plan's lateral state when the
        # vehicle is tracking it well (avoids re-originating the manoeuvre
        # from the tracking error every cycle)
        if prev is not None and prev.dp is not None and t_now is not None:
            tau = t_now - prev.t0
            if 0 <= tau < TGRID[-1] - 0.5:
                px = np.interp(tau, TGRID, prev.x)
                py = np.interp(tau, TGRID, prev.y)
                if np.hypot(px - ego.x, py - ego.y) < 0.6:
                    d0 = float(np.interp(tau, TGRID, prev.d))
                    d1 = float(np.interp(tau, TGRID, prev.dp))
                    d2 = float(np.interp(tau, TGRID, prev.dpp))

        profiles, V, S, A, Jl = self._lon_profiles(ego.v, a0, v_des)
        Nv = len(profiles)

        # lateral candidates across the full drivable width
        look = s0 + np.linspace(0, max(20.0, ego.v * 4), 12)
        lo, hi = width_fn(look)
        margin = self.W / 2 + 0.25
        dmin, dmax = float(np.max(lo)) + margin, float(np.min(hi)) - margin
        if dmax < dmin:
            mid = 0.5 * (dmin + dmax)
            dmin = dmax = mid
        grid = np.arange(dmin, dmax + 1e-6, 0.5)
        dTs = np.unique(np.round(np.concatenate([grid, [np.clip(d_pref, dmin, dmax)],
                                                 [np.clip(d0, dmin, dmax)]]), 2))
        Ls = np.array([max(8.0, 1.2 * ego.v), max(15.0, 2.2 * ego.v), max(26.0, 3.5 * ego.v)])
        DT_, LL = np.meshgrid(dTs, Ls, indexing="ij")
        DT_, LL = DT_.ravel(), LL.ravel()
        C = quintic(d0, d1, d2, DT_, LL)                              # (Nl, 6)
        Nl = len(DT_)

        # combine lateral x longitudinal -> (Nl, Nv, K)
        X = S[None, :, :]
        Lb = LL[:, None, None]
        Cb = C[:, None, None, :]
        d, dp, dpp = eval_quintic(Cb, X, Lb)
        s = s0 + X + 0 * d
        xr, yr, thr, kr = ref.interp(s)
        over = s - ref.length
        xr = xr + np.where(over > 0, over * np.cos(thr), 0)
        yr = yr + np.where(over > 0, over * np.sin(thr), 0)
        x = xr - d * np.sin(thr)
        y = yr + d * np.cos(thr)
        one_m = np.maximum(1 - kr * d, 0.2)
        yaw = thr + np.arctan2(dp, one_m)
        kappa = (kr + dpp) / one_m
        Vb = np.broadcast_to(V[None], d.shape)

        # ---------------- feasibility: road bounds & dynamics ------------
        lo_s, hi_s = width_fn(s.reshape(-1))
        lo_s = lo_s.reshape(s.shape)
        hi_s = hi_s.reshape(s.shape)
        edge_gap = np.minimum(d - self.W / 2 - lo_s, hi_s - d - self.W / 2)   # (Nl,Nv,K)
        moving = (X > 0.05)
        bad_road = ((edge_gap < 0.05) & moving & (np.arange(K) > 0)).any(-1)
        bad_kappa = ((np.abs(kappa) > self.kappa_max) & moving).any(-1)
        bad_alat = ((Vb ** 2 * np.abs(kappa)) > self.alat_max).any(-1)
        infeasible = bad_road | bad_kappa | bad_alat

        # ---------------- collision checking vs predictions -------------
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        ex = (x[..., None] + self.ego_offs * cos_y[..., None]).astype(np.float32)  # (Nl,Nv,K,5)
        ey = (y[..., None] + self.ego_offs * sin_y[..., None]).astype(np.float32)
        gap_hard = np.full(d.shape, np.inf)
        soft_cost = np.zeros(d.shape[:2])
        prox_cost = np.zeros(d.shape[:2])
        late_cost = np.zeros(d.shape[:2])
        min_gap = np.inf
        reach = 30.0 + ego.v * TGRID[-1]
        # cheap pre-filter: only hypotheses that ever enter the bounding box of
        # all candidate trajectories (inflated) can matter
        pad = self.ego_r + 3.0
        bx0, bx1 = float(x.min()) - pad, float(x.max()) + pad
        by0, by1 = float(y.min()) - pad, float(y.max()) + pad
        rel_h = []
        for h in hyps:
            cx_, cy_ = h.centers[..., 0], h.centers[..., 1]
            if ((cx_ > bx0 - h.radius[:, None]) & (cx_ < bx1 + h.radius[:, None]) &
                    (cy_ > by0 - h.radius[:, None]) & (cy_ < by1 + h.radius[:, None])).any():
                rel_h.append(h)
        # hard-constraint horizon scales with stopping time: at walking pace in
        # a market 2.2 s is enough, at highway speed we look 4.5 s ahead
        v_ref = max(ego.v, 0.5 * v_des)
        self.hard_t = float(np.clip(1.8 + v_ref / 4.0, 2.2, 4.5))
        tmask_hard = TGRID <= self.hard_t
        # margin ramps in over the first 0.6 s: the current state is what it is,
        # so only real contact counts at t=0 (otherwise a car that stopped close
        # to a parked bus could never pull away from it)
        margin_t = 0.05 + 0.25 * np.minimum(TGRID, 0.6) + 0.08 * TGRID
        viol = np.zeros(d.shape)
        ch, sh = np.cos(ego.yaw), np.sin(ego.yaw)
        for h in rel_h:
            rx = h.centers[0, :, 0] - ego.x
            ry = h.centers[0, :, 1] - ego.y
            mv = h.centers[-1, 0] - h.centers[0, 0]
            same_dir = (mv[0] * ch + mv[1] * sh) > 0.5 * np.hypot(*mv) and np.hypot(*mv) > 1.0
            behind = same_dir and np.max(ch * rx + sh * ry) < -self.L / 2 - 0.3
            hard_h = h.hard and not behind
            cx = h.centers[..., 0].astype(np.float32)                # (K, n)
            cy = h.centers[..., 1].astype(np.float32)
            dx = ex[..., :, None] - cx[None, None, :, None, :]       # (Nl,Nv,K,4,n)
            dy = ey[..., :, None] - cy[None, None, :, None, :]
            dist = np.sqrt(dx * dx + dy * dy).min(axis=(-1, -2))     # (Nl,Nv,K)
            gap = dist - self.ego_r - h.radius[None, None, :]
            if hard_h:
                m_h = margin_t + 0.035 * (ego.v + h.speed) * np.minimum(TGRID / 0.6, 1.0)
                # hard horizon grows with closing speed (oncoming bus vs. a
                # pedestrian at walking pace)
                t_h = float(np.clip(1.8 + (v_ref + h.speed) / 4.0, 2.2, 4.8))
                tmask_h = TGRID <= t_h
                g = np.where(tmask_h, gap - m_h, np.inf)
                gap_hard = np.minimum(gap_hard, g)
                viol += np.maximum(-g, 0) * np.exp(-TGRID / 2.0)
                late_cost += self.w["late"] * ((gap < 0) & ~tmask_h).any(-1) * h.prob
                min_gap = min(min_gap, float(gap[..., 0].min()))
            elif h.hard:  # object behind us: it is responsible, keep as soft cost
                soft_cost += 0.3 * self.w["soft"] * ((gap < 0).any(-1))
            else:
                soft_cost += self.w["soft"] * h.prob * (gap < 0.2).any(-1)
            prox_cost += self.w["prox"] * h.prob * (np.exp(-np.maximum(gap, 0) / 0.8)
                                                    * (0.3 + Vb / 8.0)).sum(-1) * DT
        # do not come to rest inside the swept path of an oncoming vehicle
        # (Indian two-lane roads: stopping there creates a head-on standoff)
        block_cost = np.zeros(d.shape[:2])
        fx, fy = ex[..., -1, :], ey[..., -1, :]                       # (Nl,Nv,5)
        for h in rel_h:
            if h.cls in ("pedestrian", "cattle", None) or h.speed < 2.0 or not h.hard:
                continue
            mv = h.centers[-1, 0] - h.centers[0, 0]
            if (mv[0] * ch + mv[1] * sh) > -0.7 * np.hypot(*mv):
                continue  # not oncoming
            pts = h.centers.reshape(-1, 2)
            dd_ = np.sqrt((fx[..., None] - pts[:, 0]) ** 2 + (fy[..., None] - pts[:, 1]) ** 2)
            inside = (dd_.min(axis=(-1, -2)) < self.ego_r + h.base_radius + 0.2)
            block_cost += self.w["block"] * inside
        collide = (gap_hard < 0).any(-1)
        feasible = ~infeasible & ~collide
        moving_prof = np.array([p[1] > 0.05 for p in profiles])
        if ego.v < 0.5 and not feasible[:, moving_prof].any():
            # recovery at standstill: tolerate a small incursion onto the
            # unpaved shoulder / estimated edge rather than freezing forever
            bad_road_r = ((edge_gap < -0.6) & moving & (np.arange(K) > 0)).any(-1)
            feasible = ~(bad_road_r | bad_kappa | bad_alat) & ~collide

        # ---------------- costs -----------------------------------------
        # lateral jerk (in space) from the quintic
        xs = np.linspace(0, 1, 11)[None, :] * LL[:, None]
        c3, c4, c5 = C[:, 3:4], C[:, 4:5], C[:, 5:6]
        d3 = 6 * c3 + 24 * c4 * xs + 60 * c5 * xs ** 2
        lat_jerk = (d3 ** 2).mean(1) * LL                              # (Nl,)
        J = np.zeros((Nl, Nv))
        J += self.w["lat_jerk"] * lat_jerk[:, None]
        J += self.w["lat_off"] * (DT_[:, None] - d_pref) ** 2
        J += self.w["speed"] * ((v_des - V) ** 2).mean(1)[None, :]
        J += self.w["lon_jerk"] * (Jl ** 2).mean(1)[None, :]
        J += self.w["brake"] * (np.minimum(A, 0) ** 2).mean(1)[None, :]
        J += prox_cost + soft_cost + late_cost + block_cost
        J += self.w["edge"] * (np.maximum(0.7 - edge_gap, 0) ** 2).sum(-1) * DT
        if keep_left:
            cl = 0.0 if center_fn is None else center_fn(s)
            J += self.w["side"] * (np.maximum(cl + self.W / 2 - d, 0) ** 2).sum(-1) * DT
            # no blind overtaking: only use the oncoming half if we can see far
            # enough along it (sight distance from occlusion reasoning)
            need = min(78.0, 25.0 + 5.0 * (ego.v + 8.0))  # time in lane x closing speed
            if sight < need:
                over = (cl + self.W / 2 - d) > 0.4
                J += self.w["blind"] * over.any(-1) * (1.0 - sight / need)
        if prev is not None:
            J += self.w["consist"] * (DT_[:, None] - prev.dT) ** 2
        for (ps, pd, pr) in potholes:
            for wheel in (-0.8, 0.8):
                hit = (np.abs(s - ps) < pr + 0.3) & (np.abs(d + wheel - pd) < pr + 0.15)
                J += self.w["pothole"] * (hit * (0.5 + Vb / 5.0) ** 2).sum(-1) * DT

        self.dbg = dict(J=J, feasible=feasible, infeasible=infeasible, collide=collide,
                        bad_road=bad_road, bad_kappa=bad_kappa, bad_alat=bad_alat,
                        prox=prox_cost, soft=soft_cost, late=late_cost, dT=DT_, L=LL,
                        profiles=profiles, gap_hard=gap_hard)
        n_feas = int(feasible.sum())
        emergency = False
        if n_feas > 0:
            Jf = np.where(feasible, J, np.inf)
            i, j = np.unravel_index(np.argmin(Jf), Jf.shape)
        else:
            # minimum-risk manoeuvre: least violation, strongly prefer braking
            emergency = ego.v > 0.5
            risk = viol.sum(-1) + 50.0 * infeasible + 0.01 * J + \
                0.3 * ((DT_ - d_pref) ** 2)[:, None] + 0.01 * block_cost
            brake_idx = [k for k, p in enumerate(profiles) if p[0] == "brake"]
            sub = risk[:, brake_idx]
            i, jj = np.unravel_index(np.argmin(sub), sub.shape)
            j = brake_idx[jj]
        # dense path for the tracking controller
        xs_d = np.arange(0, max(S[j, -1], 1.0) + 25.0, 0.5)
        dd, _, _ = eval_quintic(C[i][None], xs_d[None], LL[i])
        px, py, _ = ref.to_cart(s0 + xs_d, dd[0])
        alt = []
        if n_feas:
            order = np.argsort(np.where(feasible, J, np.inf), axis=None)[:40]
            for o in order[1::3]:
                a_, b_ = np.unravel_index(o, J.shape)
                if np.isfinite(J[a_, b_]):
                    alt.append(np.column_stack([x[a_, b_], y[a_, b_]]))
        ms = (time.perf_counter() - tic) * 1000
        return PlanResult(TGRID.copy(), x[i, j], y[i, j], yaw[i, j], V[j], A[j],
                          s[i, j], d[i, j], kappa[i, j], np.column_stack([px, py]),
                          n_feas > 0, emergency, float(DT_[i]), float(profiles[j][1]),
                          Nl * Nv, n_feas, ms, min_gap, float(J[i, j]), alt,
                          dp=dp[i, j].copy(), dpp=dpp[i, j].copy())

    # ------------------------------------------------------------------
    def still_valid(self, plan, hyps, t_elapsed):
        """Event-trigger check: does the current plan now hit a hard hypothesis?"""
        if plan is None:
            return False
        k0 = int(round(t_elapsed / DT))
        if k0 >= K - 5:
            return False
        idx = np.arange(k0, min(K, k0 + int(getattr(self, "hard_t", HARD_T) / DT)))
        ex = plan.x[idx, None] + self.ego_offs * np.cos(plan.yaw[idx, None])
        ey = plan.y[idx, None] + self.ego_offs * np.sin(plan.yaw[idx, None])
        for h in hyps:
            if not h.hard:
                continue
            ch, sh = np.cos(plan.yaw[k0]), np.sin(plan.yaw[k0])
            rx = h.centers[0, :, 0] - plan.x[k0]
            ry = h.centers[0, :, 1] - plan.y[k0]
            mv = h.centers[-1, 0] - h.centers[0, 0]
            same_dir = (mv[0] * ch + mv[1] * sh) > 0.5 * np.hypot(*mv) and np.hypot(*mv) > 1.0
            if same_dir and np.max(ch * rx + sh * ry) < -self.L / 2 - 0.3:
                continue
            kk = idx - k0
            cx = h.centers[kk][:, None, :, 0]
            cy = h.centers[kk][:, None, :, 1]
            dist = np.sqrt((ex[..., None] - cx) ** 2 + (ey[..., None] - cy) ** 2).min(axis=(1, 2))
            if (dist - self.ego_r - h.radius[kk] < 0.1).any():
                return False
        return True
