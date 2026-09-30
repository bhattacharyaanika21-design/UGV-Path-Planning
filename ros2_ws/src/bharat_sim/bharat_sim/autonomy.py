#!/usr/bin/env python3
"""autonomy: the full Bharat-AV stack driving the Gazebo car.

  subscribes  /ego/odom   (nav_msgs/Odometry, bridged from Gazebo)
              /gt/agents  (std_msgs/String JSON, from traffic_manager)
  publishes   /model/ego_car/cmd_vel  (geometry_msgs/Twist -> AckermannSteering)
              /av/markers (visualization_msgs/MarkerArray, for RViz)
              /av/status  (std_msgs/String JSON: state, speed, target, reasons ...)

Rates: perception + prediction + decision + (re)planning 10 Hz, control 20 Hz.
Every run is logged to ~/bharat_runs/<scenario>_s<seed>_<time>/ (CSV + metrics JSON).

Stage 1 (this file) feeds the camera/LiDAR/radar *models* from av/sensors.py
with Gazebo ground truth (with occlusion). Stage 2 swaps in detections from the
real Gazebo camera and LiDAR topics (see README, step 9).
"""
from __future__ import annotations

import csv
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from bharat_sim.av.scenarios import SCENARIOS
from bharat_sim.av.stack import AutonomyStack
from bharat_sim.av.world import CLASSES, rect_distance, rects_overlap
from bharat_sim.glue import (EgoStateEstimator, apply_agents_json, twist_from_control,
                             yaw_from_quat)

STATE_RGB = {"CRUISE": (0.17, 0.63, 0.17), "FOLLOW": (0.12, 0.47, 0.71), "NUDGE": (0.58, 0.40, 0.74),
             "YIELD": (1.0, 0.5, 0.05), "OCCLUSION_CAUTION": (0.74, 0.74, 0.13), "WAIT": (0.5, 0.5, 0.5),
             "EMERGENCY": (0.84, 0.15, 0.16), "REROUTE": (0.09, 0.75, 0.81)}


def hex_rgba(h, a=1.0):
    h = h.lstrip("#")
    return ColorRGBA(r=int(h[0:2], 16) / 255, g=int(h[2:4], 16) / 255, b=int(h[4:6], 16) / 255, a=a)


class Autonomy(Node):
    def __init__(self):
        super().__init__("autonomy")
        self.declare_parameter("scenario", "S5")
        self.declare_parameter("seed", 0)
        self.declare_parameter("log_dir", os.path.expanduser("~/bharat_runs"))
        self.key = self.get_parameter("scenario").value
        self.seed = int(self.get_parameter("seed").value)
        self.spec = SCENARIOS[self.key](self.seed)
        self.world = self.spec.world
        self.stack = AutonomyStack(self.spec, seed=self.seed)
        x0, y0, yaw0, _ = self.spec.ego_init
        self.est = EgoStateEstimator(self.stack.ego, (x0, y0, yaw0))
        self.world.ego = self.stack.ego

        self.pub_cmd = self.create_publisher(Twist, "/model/ego_car/cmd_vel", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/av/markers", 10)
        self.pub_status = self.create_publisher(String, "/av/status", 10)
        self.create_subscription(Odometry, "/ego/odom", self.on_odom, 20)
        self.create_subscription(String, "/gt/agents", self.on_agents, 10)
        self.create_timer(0.1, self.cycle)
        self.create_timer(0.05, self.control)

        self.have_odom = self.have_agents = False
        self.min_clear_now = math.inf
        self.t_start = None
        self.collision = None
        self.min_clear = math.inf
        self.done = False
        self.ref_drawn = -1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.get_parameter("log_dir").value, f"{self.key}_s{self.seed}_{stamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.csv_f = open(os.path.join(self.run_dir, "ego_log.csv"), "w", newline="")
        self.csv = csv.writer(self.csv_f)
        self.csv.writerow(["t", "x", "y", "yaw", "v", "a", "state", "v_des", "clearance", "reasons",
                           "plan_ms", "n_feas", "emergency"])
        self.get_logger().info(f"autonomy: {self.spec.title} (seed {self.seed}); logging to {self.run_dir}")

    # ------------------------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        p, q, tw = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        self.est.update(self.now(), p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w),
                        math.hypot(tw.x, tw.y))
        self.have_odom = True

    def on_agents(self, msg):
        apply_agents_json(self.world, msg.data)
        self.have_agents = True

    # ------------------------------------------------------------------
    def cycle(self):
        if not (self.have_odom and self.have_agents) or self.done:
            return
        t = self.now()
        if self.t_start is None:
            self.t_start = t
        self.world.t = t
        reasons = self.stack.cycle(t, self.world)
        self.check_safety(t)
        st = self.stack.fsm.state
        ego = self.stack.ego
        plan = self.stack.plan
        self.csv.writerow([f"{t - self.t_start:.2f}", f"{ego.x:.3f}", f"{ego.y:.3f}", f"{ego.yaw:.4f}",
                           f"{ego.v:.3f}", f"{ego.a:.3f}", st, f"{self.stack.v_des:.2f}",
                           f"{self.min_clear_now:.2f}", "|".join(reasons),
                           f"{plan.compute_ms:.1f}" if plan else "", plan.n_feas if plan else "",
                           int(plan.emergency) if plan else ""])
        self.pub_status.publish(String(data=json.dumps(dict(
            t=round(t - self.t_start, 2), state=st, speed_kmh=round(ego.v * 3.6, 1),
            target_kmh=round(self.stack.v_des * 3.6, 1), caps=self.stack.fsm.caps, reasons=reasons,
            safe_paths=f"{plan.n_feas}/{plan.n_cand}" if plan else "", tracks=len(self.stack.conf),
            events=self.stack.events[-3:]), default=str)))
        self.publish_markers()
        if self.stack.goal_reached or self.collision or t - self.t_start > self.spec.t_max:
            self.finish(t)

    def control(self):
        if not self.have_odom or self.stack.plan is None:
            return
        a_cmd, steer, _ = self.stack.control(self.now())
        if self.done:
            a_cmd, steer = -4.0, 0.0
        lin, ang = twist_from_control(self.stack.ego.v, a_cmd, steer)
        msg = Twist()
        msg.linear.x, msg.angular.z = lin, ang
        self.pub_cmd.publish(msg)

    # ------------------------------------------------------------------
    def check_safety(self, t):
        ego = self.stack.ego
        ec = ego.corners()
        clear = math.inf
        for a in self.world.active_agents():
            d = math.hypot(a.x - ego.x, a.y - ego.y)
            if d > 20:
                continue
            ac = a.corners()
            if d < 8 and rects_overlap(ec, ac):
                clear = 0.0
                if self.collision is None:
                    self.collision = dict(t=round(t - self.t_start, 2), agent=a.id, cls=a.cls,
                                          ego_speed=round(ego.v, 2))
                    self.get_logger().error(f"COLLISION with {a.cls} #{a.id}")
            else:
                clear = min(clear, rect_distance(ec, ac) if d < 12 else d - (a.L + ego.L) / 2)
        self.min_clear_now = clear
        self.min_clear = min(self.min_clear, clear)

    def finish(self, t):
        self.done = True
        plans = self.stack.plans
        ms = np.array([p["ms"] for p in plans]) if plans else np.zeros(1)
        m = dict(scenario=self.key, seed=self.seed, title=self.spec.title,
                 completed=bool(self.stack.goal_reached and not self.collision),
                 collision=self.collision, time_s=round(t - self.t_start, 2),
                 min_clearance_m=round(self.min_clear, 3),
                 replans=len(plans), replan_ms_mean=round(float(ms.mean()), 1),
                 replan_ms_p95=round(float(np.percentile(ms, 95)), 1),
                 emergency_plans=int(sum(p["emergency"] for p in plans)),
                 events=self.stack.events)
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=1, default=str)
        self.csv_f.flush()
        self.get_logger().info("RUN FINISHED: " + json.dumps(m, default=str))

    # ------------------------------------------------------------------
    def publish_markers(self):
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        def mk(ns, mid, mtype, scale, color):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = scale
            m.color = color
            m.pose.orientation.w = 1.0
            return m

        clear = Marker()
        clear.header.frame_id = "world"
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        plan = self.stack.plan
        if plan is not None:
            m = mk("plan", 0, Marker.LINE_STRIP, (0.35, 0, 0),
                   ColorRGBA(r=1.0, g=0.15, b=0.15, a=1.0) if plan.emergency else ColorRGBA(r=0.22, g=1.0, b=0.08, a=1.0))
            m.points = [Point(x=float(x), y=float(y), z=0.15) for x, y in zip(plan.x, plan.y)]
            arr.markers.append(m)
            for i, alt in enumerate(plan.alt_xy[:10]):
                m = mk("candidates", i, Marker.LINE_STRIP, (0.06, 0, 0), ColorRGBA(r=0.8, g=0.9, b=1.0, a=0.5))
                m.points = [Point(x=float(x), y=float(y), z=0.1) for x, y in alt]
                arr.markers.append(m)
        # reference line and drivable bounds (redrawn every cycle; cheap)
        ref, W = self.stack.ref, self.stack.width
        ss = np.arange(0, ref.length, 2.0)
        lo, hi = W(ss)
        for nm, dd, col in (("ref", np.zeros_like(ss), ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.6)),
                            ("bound_l", hi, ColorRGBA(r=1.0, g=0.7, b=0.3, a=0.6)),
                            ("bound_r", lo, ColorRGBA(r=1.0, g=0.7, b=0.3, a=0.6))):
            x, y, _ = ref.to_cart(ss, dd)
            m = mk(nm, 0, Marker.LINE_STRIP, (0.08, 0, 0), col)
            m.points = [Point(x=float(a), y=float(b), z=0.05) for a, b in zip(x, y)]
            arr.markers.append(m)
        # fused tracks
        for tr in self.stack.conf:
            L, Wd = tr.dims
            m = mk("tracks", tr.id, Marker.CUBE, (float(L), float(Wd), 0.3),
                   hex_rgba(CLASSES.get(tr.cls, {}).get("color", "#ffffff"), 0.55))
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(tr.pos[0]), float(tr.pos[1]), 2.3
            m.pose.orientation.z, m.pose.orientation.w = math.sin(tr.yaw / 2), math.cos(tr.yaw / 2)
            arr.markers.append(m)
            txt = mk("labels", tr.id, Marker.TEXT_VIEW_FACING, (0, 0, 0.8), ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(tr.pos[0]), float(tr.pos[1]), 3.0
            txt.text = f"{tr.cls or '?'} {tr.speed * 3.6:.0f}km/h"
            arr.markers.append(txt)
        # predictions: hard = red, soft = orange
        hard = mk("pred_hard", 0, Marker.POINTS, (0.25, 0.25, 0), ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.9))
        soft = mk("pred_soft", 0, Marker.POINTS, (0.18, 0.18, 0), ColorRGBA(r=1.0, g=0.75, b=0.3, a=0.7))
        for h in self.stack.hyps:
            tgt = hard if h.hard else soft
            tgt.points += [Point(x=float(p[0]), y=float(p[1]), z=0.2) for p in h.centers[::2, 0, :]]
        arr.markers += [hard, soft]
        # FSM state above the car
        ego = self.stack.ego
        st = self.stack.fsm.state
        r, g, b = STATE_RGB.get(st, (1, 1, 1))
        txt = mk("state", 0, Marker.TEXT_VIEW_FACING, (0, 0, 1.2), ColorRGBA(r=r, g=g, b=b, a=1.0))
        txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = ego.x, ego.y, 3.5
        txt.text = f"{st}  {ego.v * 3.6:.0f}/{self.stack.v_des * 3.6:.0f} km/h"
        arr.markers.append(txt)
        self.pub_mk.publish(arr)


def main():
    rclpy.init()
    node = Autonomy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not node.done and node.t_start is not None:
            node.finish(node.now())
        node.csv_f.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
