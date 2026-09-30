"""Ego vehicle: kinematic bicycle model + trajectory-tracking controller.

Equivalent to a Simulink "Vehicle Body 3DOF" / bicycle model with a pure-
pursuit lateral controller and a feed-forward + PI longitudinal controller.
Actuators have rate limits and a first-order lag.
"""
from __future__ import annotations

import numpy as np


class EgoVehicle:
    def __init__(self, x, y, yaw, v, L=4.4, W=1.8, wheelbase=2.7):
        self.x, self.y, self.yaw, self.v = x, y, yaw, v
        self.a = 0.0
        self.delta = 0.0
        self.L, self.W = L, W
        self.wb = wheelbase
        self.lr = wheelbase / 2
        self.delta_max = np.deg2rad(33)
        self.delta_rate = np.deg2rad(40)
        self.tau_a = 0.15
        self.id = 0
        self.cls = "ego"

    @property
    def pos(self):
        return np.array([self.x, self.y])

    def corners(self):
        from .world import rect_corners
        return rect_corners(self.x, self.y, self.yaw, self.L, self.W)

    def step(self, a_cmd, delta_cmd, dt):
        a_cmd = float(np.clip(a_cmd, -8.0, 2.5))
        self.a += (a_cmd - self.a) * min(1.0, dt / self.tau_a)
        dcmd = float(np.clip(delta_cmd, -self.delta_max, self.delta_max))
        self.delta += float(np.clip(dcmd - self.delta, -self.delta_rate * dt, self.delta_rate * dt))
        beta = np.arctan(self.lr / self.wb * np.tan(self.delta))
        self.x += self.v * np.cos(self.yaw + beta) * dt
        self.y += self.v * np.sin(self.yaw + beta) * dt
        self.yaw += self.v / self.lr * np.sin(beta) * dt
        self.v = max(0.0, self.v + self.a * dt)


class TrackingController:
    def __init__(self, wheelbase=2.7):
        self.wb = wheelbase
        self.v_int = 0.0

    def command(self, ego, plan, t_since_plan):
        # longitudinal: feed-forward plan accel + PI on speed error
        t_look = t_since_plan + 0.15
        v_ref = float(np.interp(t_look, plan.t, plan.v))
        a_ff = float(np.interp(t_look, plan.t, plan.a))
        err = v_ref - ego.v
        self.v_int = float(np.clip(self.v_int + err * 0.05, -2, 2))
        a_cmd = a_ff + 1.2 * err + 0.3 * self.v_int
        v_soon = float(np.interp(t_since_plan + 1.0, plan.t, plan.v))
        if v_soon < 0.05 and ego.v < 0.3:
            a_cmd = min(a_cmd, -1.0)  # hold brake at standstill
        elif ego.v < 0.3 and v_soon > 0.2:
            a_cmd = max(a_cmd, 0.8)   # pull away
        # lateral: pure pursuit on dense planned path from the rear axle
        rear = ego.pos - (self.wb / 2) * np.array([np.cos(ego.yaw), np.sin(ego.yaw)])
        P = plan.path_xy
        Ld = float(np.clip(3.0 + 0.55 * ego.v, 4.0, 18.0))
        dists = np.hypot(P[:, 0] - rear[0], P[:, 1] - rear[1])
        i0 = int(np.argmin(dists))
        ahead = np.where(dists[i0:] >= Ld)[0]
        tgt = P[i0 + ahead[0]] if len(ahead) else P[-1]
        alpha = np.arctan2(tgt[1] - rear[1], tgt[0] - rear[0]) - ego.yaw
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        delta = np.arctan2(2 * self.wb * np.sin(alpha), Ld)
        return a_cmd, float(delta)
