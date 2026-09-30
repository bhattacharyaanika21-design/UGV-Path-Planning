#!/usr/bin/env python3
"""Offline rehearsal of the Gazebo loop — no ROS or Gazebo needed.

It runs the exact same pieces the ROS nodes run (traffic manager world,
EgoStateEstimator, AutonomyStack, twist_from_control) against a stand-in for
Gazebo's AckermannSteering car that only accepts a Twist, with the same rates
as the launch file (agents 20 Hz, perception/planning 10 Hz, control 20 Hz).

    python3 test_offline.py            # all scenarios, seed 0
    python3 test_offline.py S5 S2 --seeds 3
"""
import argparse
import math
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from bharat_sim.av.scenarios import SCENARIOS           # noqa: E402
from bharat_sim.av.stack import AutonomyStack           # noqa: E402
from bharat_sim.av.vehicle import EgoVehicle            # noqa: E402
from bharat_sim.av.world import rects_overlap           # noqa: E402
from bharat_sim.glue import (EgoStateEstimator, agents_to_json,  # noqa: E402
                             apply_agents_json, twist_from_control)


class FakeAckermannCar:
    """Mimics gz AckermannSteering: Twist in, odometry out."""

    def __init__(self, x, y, yaw, wb=2.7):
        self.x, self.y, self.yaw, self.v, self.steer, self.wb = x, y, yaw, 0.0, 0.0, wb
        self.lin, self.ang = 0.0, 0.0

    def step(self, dt):
        acc = np.clip((self.lin - self.v) / dt, -7.5, 2.5)
        self.v = max(0.0, self.v + acc * dt)
        tgt = math.atan(self.wb * self.ang / self.lin) if self.lin > 0.05 else self.steer
        tgt = float(np.clip(tgt, -0.576, 0.576))
        self.steer += float(np.clip(tgt - self.steer, -1.0 * dt, 1.0 * dt))
        # rear-axle kinematic bicycle, state kept at the body centre
        rx = self.x - self.wb / 2 * math.cos(self.yaw)
        ry = self.y - self.wb / 2 * math.sin(self.yaw)
        rx += self.v * math.cos(self.yaw) * dt
        ry += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / self.wb * math.tan(self.steer) * dt
        self.x = rx + self.wb / 2 * math.cos(self.yaw)
        self.y = ry + self.wb / 2 * math.sin(self.yaw)


def run(key, seed, dt=0.05):
    # --- "Gazebo" side: world + traffic manager ---------------------------
    spec_tm = SCENARIOS[key](seed)
    world_tm = spec_tm.world
    x0, y0, yaw0, _ = spec_tm.ego_init
    car = FakeAckermannCar(x0, y0, yaw0)
    proxy = EgoVehicle(x0, y0, yaw0, 0.0)
    world_tm.ego = proxy
    for a in world_tm.agents:
        if a.behavior is not None and hasattr(a.behavior, "place"):
            a.behavior.place(a)
    # --- "autonomy node" side: its own copy of the scenario ---------------
    spec_av = SCENARIOS[key](seed)
    world_av = spec_av.world
    stack = AutonomyStack(spec_av, seed=seed)
    est = EgoStateEstimator(stack.ego, (x0, y0, yaw0))
    t, collision, n = 0.0, None, 0
    t0 = time.time()
    while t < spec_tm.t_max and not stack.goal_reached:
        # odometry -> both nodes (world-frame pose, as OdometryPublisher gives)
        est.update(t, car.x, car.y, car.yaw, car.v)
        proxy.x, proxy.y, proxy.yaw, proxy.v = car.x, car.y, car.yaw, car.v
        # traffic manager tick (20 Hz) and publish agents as JSON
        world_tm.step(dt)
        msg = agents_to_json(world_tm)
        if n % 2 == 0:                                   # autonomy 10 Hz
            apply_agents_json(world_av, msg)
            world_av.t = t
            stack.cycle(t, world_av)
        a_cmd, steer, _ = stack.control(t)               # control 20 Hz
        car.lin, car.ang = twist_from_control(stack.ego.v, a_cmd, steer)
        car.step(dt)
        t += dt
        n += 1
        ec = proxy.corners()
        for a in world_tm.active_agents():
            if math.hypot(a.x - car.x, a.y - car.y) < 8 and rects_overlap(ec, a.corners()):
                collision = (round(t, 2), a.cls, round(car.v, 2))
        if collision:
            break
    ok = stack.goal_reached and collision is None
    ms = np.mean([p["ms"] for p in stack.plans]) if stack.plans else 0
    print(f"{key} seed {seed}: {'PASS' if ok else 'FAIL'}  t={t:5.1f}s  collision={collision}  "
          f"reroutes={sum(e[1]=='reroute' for e in stack.events)}  plan {ms:.0f} ms  "
          f"({time.time()-t0:.0f}s wall)", flush=True)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", default=list(SCENARIOS))
    ap.add_argument("--seeds", type=int, default=1)
    a = ap.parse_args()
    res = [run(k, s) for k in a.keys for s in range(a.seeds)]
    print(f"\n{sum(res)}/{len(res)} passed")
    sys.exit(0 if all(res) else 1)
