"""ROS-free glue between Gazebo/ROS messages and the av library.

Everything here is plain Python so it can be unit-tested without ROS or
Gazebo (see test_offline.py).
"""
from __future__ import annotations

import json
import math

import numpy as np

WHEELBASE = 2.7
V_MAX = 25.0


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class EgoStateEstimator:
    """Turns odometry (pose + forward speed) into the ego object the stack uses.

    Handles both odometry conventions: world-frame pose, or pose relative to
    the spawn pose (detected on the first message and corrected)."""

    def __init__(self, ego, spawn_xy_yaw):
        self.ego = ego
        self.spawn = spawn_xy_yaw
        self.offset = None
        self.last_t = None
        self.last_v = 0.0

    def update(self, t, x, y, yaw, v):
        if self.offset is None:
            sx, sy, syaw = self.spawn
            relative = math.hypot(x, y) < 0.5 and math.hypot(sx, sy) > 2.0
            self.offset = (sx, sy, syaw) if relative else (0.0, 0.0, 0.0)
        ox, oy, oyaw = self.offset
        if oyaw or ox or oy:
            c, s = math.cos(oyaw), math.sin(oyaw)
            x, y = ox + c * x - s * y, oy + s * x + c * y
            yaw = yaw + oyaw
        e = self.ego
        e.x, e.y, e.yaw, e.v = float(x), float(y), float(yaw), max(0.0, float(v))
        if self.last_t is not None and t > self.last_t:
            a_raw = (e.v - self.last_v) / (t - self.last_t)
            e.a = 0.7 * e.a + 0.3 * float(np.clip(a_raw, -9, 4))
        self.last_t, self.last_v = t, e.v


def twist_from_control(v_now, a_cmd, steer, dt=0.05):
    """Stack output (accel, steering angle) -> Twist for AckermannSteering.

    Speed: AckermannSteering ramps towards the commanded speed at its own
    acceleration limit, so we command exactly one control step ahead
    (v + a*dt); that makes the realised acceleration equal a_cmd instead of
    always hitting the limit.
    Steering: the system computes steer = atan(wheelbase * angular / linear),
    so we invert that exactly."""
    v_cmd = float(np.clip(v_now + a_cmd * dt, 0.0, V_MAX))
    if a_cmd < -0.5 and v_now < 0.3:
        v_cmd = 0.0
    lin = v_cmd
    ang = lin * math.tan(steer) / WHEELBASE if lin > 0.05 else 0.0
    return lin, ang


def agents_to_json(world):
    return json.dumps([dict(id=a.id, cls=a.cls, x=round(a.x, 3), y=round(a.y, 3),
                            yaw=round(a.yaw, 4), v=round(a.v, 3), L=a.L, W=a.W,
                            active=bool(a.active)) for a in world.agents])


def apply_agents_json(world, text):
    by_id = {a.id: a for a in world.agents}
    for d in json.loads(text):
        a = by_id.get(d["id"])
        if a is None:
            continue
        a.x, a.y, a.yaw, a.v, a.active = d["x"], d["y"], d["yaw"], d["v"], d["active"]


PARK_XY = (-500.0, -500.0)   # where inactive agents are parked in Gazebo
