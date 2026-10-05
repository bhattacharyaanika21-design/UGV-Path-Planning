#!/usr/bin/env python3
"""traffic_manager: runs the scenario's road users (world.py behaviours:
wander, informal merges, wrong-way riders, darting pedestrians, cattle that
stop or turn back) and teleports the matching Gazebo models every tick.

Publishes /gt/agents (std_msgs/String, JSON) = ground truth for the
autonomy node's sensor models and for the metrics.
"""     
from __future__ import annotations

import shutil
import subprocess

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from bharat_sim.av.scenarios import SCENARIOS
from bharat_sim.av.vehicle import EgoVehicle
from bharat_sim.gen_world import WORLD_NAME
from bharat_sim.glue import PARK_XY, EgoStateEstimator, agents_to_json, quat_from_yaw


class GzPoser:
    """Moves many Gazebo models. Prefers the gz-transport Python bindings
    (one set_pose_vector call per tick); falls back to per-model set_pose,
    then to the `gz service` CLI (slower, so it is called at most at 5 Hz)."""

    def __init__(self, world, logger):
        self.world, self.log = world, logger
        self.mode = None
        self.cli_busy = None
        try:
            from gz.transport13 import Node as GzNode
            from gz.msgs10.boolean_pb2 import Boolean
            from gz.msgs10.pose_pb2 import Pose
            from gz.msgs10.pose_v_pb2 import Pose_V
            self.gz, self.Pose, self.Pose_V, self.Boolean = GzNode(), Pose, Pose_V, Boolean
            self.mode = "bindings"
            logger.info("GzPoser: using gz-transport13 Python bindings")
        except Exception as exc:  # noqa: BLE001
            if shutil.which("gz"):
                self.mode = "cli"
                logger.warn(f"GzPoser: Python bindings unavailable ({exc}); using `gz service` CLI "
                            "(install python3-gz-transport13 for smooth 20 Hz motion)")
            else:
                logger.error("GzPoser: neither gz python bindings nor `gz` CLI found")
        self.vector_ok = True

    def send(self, items):
        """items: list of (name, x, y, yaw)."""
        if self.mode == "bindings":
            if self.vector_ok:
                req = self.Pose_V()
                for name, x, y, yaw in items:
                    p = req.pose.add()
                    self._fill(p, name, x, y, yaw)
                ok, _ = self.gz.request(f"/world/{self.world}/set_pose_vector", req,
                                        self.Pose_V, self.Boolean, 100)
                if ok:
                    return
                self.vector_ok = False
                self.log.warn("set_pose_vector not available; falling back to set_pose per model")
            for name, x, y, yaw in items:
                p = self.Pose()
                self._fill(p, name, x, y, yaw)
                self.gz.request(f"/world/{self.world}/set_pose", p, self.Pose, self.Boolean, 50)
        elif self.mode == "cli":
            if self.cli_busy is not None and self.cli_busy.poll() is None:
                return  # previous call still running: skip this tick
            poses = ", ".join(
                f'{{name: "{n}", position: {{x: {x:.3f}, y: {y:.3f}, z: 0}}, '
                f'orientation: {{x: 0, y: 0, z: {quat_from_yaw(w)[2]:.5f}, w: {quat_from_yaw(w)[3]:.5f}}}}}'
                for n, x, y, w in items)
            self.cli_busy = subprocess.Popen(
                ["gz", "service", "-s", f"/world/{self.world}/set_pose_vector",
                 "--reqtype", "gz.msgs.Pose_V", "--reptype", "gz.msgs.Boolean",
                 "--timeout", "200", "--req", f"pose: [{poses}]"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _fill(p, name, x, y, yaw):
        p.name = name
        p.position.x, p.position.y, p.position.z = x, y, 0.0
        qx, qy, qz, qw = quat_from_yaw(yaw)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw


class TrafficManager(Node):
    def __init__(self):
        super().__init__("traffic_manager")
        self.declare_parameter("scenario", "S5")
        self.declare_parameter("seed", 0)
        self.declare_parameter("rate_hz", 20.0)
        key = self.get_parameter("scenario").value
        seed = int(self.get_parameter("seed").value)
        self.spec = SCENARIOS[key](seed)
        self.world = self.spec.world
        x0, y0, yaw0, _ = self.spec.ego_init
        self.ego = EgoVehicle(x0, y0, yaw0, 0.0)
        self.world.ego = self.ego
        self.est = EgoStateEstimator(self.ego, (x0, y0, yaw0))
        for a in self.world.agents:
            if a.behavior is not None and hasattr(a.behavior, "place"):
                a.behavior.place(a)
        self.poser = GzPoser(WORLD_NAME, self.get_logger())
        self.pub = self.create_publisher(String, "/gt/agents", 10)
        self.create_subscription(Odometry, "/ego/odom", self.on_odom, 10)
        self.dt = 1.0 / float(self.get_parameter("rate_hz").value)
        self.last_t = None
        self.have_odom = False
        self.n = 0
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(f"traffic_manager: {self.spec.title} (seed {seed}), "
                               f"{len(self.world.agents)} road users")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        from bharat_sim.glue import yaw_from_quat
        p, q, tw = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear
        self.est.update(self.now(), p.x, p.y, yaw_from_quat(q.x, q.y, q.z, q.w),
                        (tw.x ** 2 + tw.y ** 2) ** 0.5)
        self.have_odom = True

    def tick(self):
        if not self.have_odom:
            return  # wait for Gazebo to start publishing the ego car
        t = self.now()
        dt = self.dt if self.last_t is None else min(max(t - self.last_t, 0.0), 0.2)
        self.last_t = t
        if dt <= 0:
            return
        self.world.step(dt)
        self.n += 1
        items = [(f"agent_{a.id}", a.x, a.y, a.yaw) if a.active else
                 (f"agent_{a.id}", PARK_XY[0], PARK_XY[1], 0.0) for a in self.world.agents]
        if self.poser.mode == "cli" and self.n % 4:
            pass  # CLI fallback: 5 Hz
        else:
            self.poser.send(items)
        self.pub.publish(String(data=agents_to_json(self.world)))


def main():
    rclpy.init()
    node = TrafficManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
