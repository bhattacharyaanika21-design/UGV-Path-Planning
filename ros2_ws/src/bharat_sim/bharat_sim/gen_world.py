#!/usr/bin/env python3
"""Generate a Gazebo Sim 8 (Harmonic) world for any scenario.

    ros2 run bharat_sim gen_world --scenario S5 --seed 0
    # -> ~/bharat_worlds/S5_s0.sdf    (open with:  gz sim -r ~/bharat_worlds/S5_s0.sdf)

The world contains:
  * a ground plane (the only collision surface the car drives on)
  * unmarked roads, junction areas, dusty shoulders and potholes (visual)
  * buildings / bushes from the scenario occluders (real 3D occlusion for camera + LiDAR)
  * one model per road user, named agent_<id> (moved at runtime by traffic_manager)
  * the ego car (4-wheel Ackermann car with camera, 3D LiDAR, IMU)
"""
from __future__ import annotations

import argparse
import math
import os

from bharat_sim.av.scenarios import SCENARIOS
from bharat_sim.av.world import CLASSES
from bharat_sim.glue import PARK_XY

WORLD_NAME = "indian_roads"


def rgb(hexcol, a=1.0):
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return f"{r:.3f} {g:.3f} {b:.3f} {a}"


def mat(col):
    return (f"<material><ambient>{col}</ambient><diffuse>{col}</diffuse>"
            f"<specular>0.05 0.05 0.05 1</specular></material>")


def box_visual(name, x, y, z, yaw, sx, sy, sz, col):
    return (f'<visual name="{name}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>'
            f"<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry>{mat(col)}</visual>")


def cyl_visual(name, x, y, z, r, h, col, roll=0.0):
    return (f'<visual name="{name}"><pose>{x:.3f} {y:.3f} {z:.3f} {roll:.4f} 0 0</pose>'
            f"<geometry><cylinder><radius>{r:.3f}</radius><length>{h:.3f}</length></cylinder></geometry>"
            f"{mat(col)}</visual>")


def sphere_visual(name, x, y, z, r, col):
    return (f'<visual name="{name}"><pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>'
            f"<geometry><sphere><radius>{r:.3f}</radius></sphere></geometry>{mat(col)}</visual>")


# ---------------------------------------------------------------------------
# Ego car: 4.4 x 1.8 m hatchback, wheelbase 2.7 m, Ackermann steering
# ---------------------------------------------------------------------------
def ego_model_sdf(x=0.0, y=0.0, yaw=0.0, name="ego_car"):
    wb, tw, R, wr_w = 2.7, 1.55, 0.31, 0.22
    body = rgb("#00b3b3")
    tyre = "0.08 0.08 0.08 1"

    def wheel(tag, wx, wy):
        return f"""
    <link name="{tag}_wheel">
      <pose>{wx} {wy} {R} -1.5708 0 0</pose>
      <inertial><mass>15</mass><inertia><ixx>0.4</ixx><iyy>0.4</iyy><izz>0.72</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <collision name="c"><geometry><cylinder><radius>{R}</radius><length>{wr_w}</length></cylinder></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
      <visual name="v"><geometry><cylinder><radius>{R}</radius><length>{wr_w}</length></cylinder></geometry>
        <material><ambient>{tyre}</ambient><diffuse>{tyre}</diffuse></material></visual>
    </link>"""

    def steer_link(tag, wx, wy):
        return f"""
    <link name="{tag}_steering_link">
      <pose>{wx} {wy} {R} 0 0 0</pose>
      <inertial><mass>2</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.01</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
    </link>
    <joint name="{tag}_steering_joint" type="revolute">
      <parent>chassis</parent><child>{tag}_steering_link</child>
      <axis><xyz expressed_in="__model__">0 0 1</xyz><limit><lower>-0.6</lower><upper>0.6</upper><effort>1000</effort></limit></axis>
    </joint>
    <joint name="{tag}_wheel_joint" type="revolute">
      <parent>{tag}_steering_link</parent><child>{tag}_wheel</child>
      <axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
    </joint>"""

    def rear_joint(tag):
        return f"""
    <joint name="{tag}_wheel_joint" type="revolute">
      <parent>chassis</parent><child>{tag}_wheel</child>
      <axis><xyz expressed_in="__model__">0 1 0</xyz><limit><lower>-1e16</lower><upper>1e16</upper></limit></axis>
    </joint>"""

    fx, rx, hy = wb / 2, -wb / 2, tw / 2
    return f"""
  <model name="{name}">
    <pose>{x:.3f} {y:.3f} 0.02 0 0 {yaw:.4f}</pose>
    <link name="chassis">
      <pose>0 0 0.62 0 0 0</pose>
      <inertial><mass>1100</mass><pose>0 0 -0.1 0 0 0</pose>
        <inertia><ixx>420</ixx><iyy>1800</iyy><izz>2000</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <collision name="body"><geometry><box><size>4.4 1.8 0.7</size></box></geometry></collision>
      <visual name="body"><geometry><box><size>4.4 1.8 0.7</size></box></geometry>
        <material><ambient>{body}</ambient><diffuse>{body}</diffuse><specular>0.3 0.3 0.3 1</specular></material></visual>
      <visual name="cabin"><pose>-0.3 0 0.6 0 0 0</pose><geometry><box><size>2.4 1.6 0.55</size></box></geometry>
        <material><ambient>0.1 0.15 0.2 1</ambient><diffuse>0.1 0.15 0.2 1</diffuse></material></visual>

      <sensor name="front_camera" type="camera">
        <pose>1.3 0 0.65 0 0.05 0</pose>
        <always_on>1</always_on><update_rate>15</update_rate><visualize>true</visualize>
        <topic>camera</topic><gz_frame_id>camera_link</gz_frame_id>
        <camera><horizontal_fov>1.92</horizontal_fov>
          <image><width>1280</width><height>720</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>150</far></clip></camera>
      </sensor>
      <sensor name="roof_lidar" type="gpu_lidar">
        <pose>0 0 1.3 0 0 0</pose>
        <always_on>1</always_on><update_rate>10</update_rate><visualize>false</visualize>
        <topic>lidar</topic><gz_frame_id>lidar_link</gz_frame_id>
        <lidar><scan>
          <horizontal><samples>900</samples><resolution>1</resolution><min_angle>-3.14159</min_angle><max_angle>3.14159</max_angle></horizontal>
          <vertical><samples>16</samples><resolution>1</resolution><min_angle>-0.26</min_angle><max_angle>0.26</max_angle></vertical>
        </scan><range><min>0.5</min><max>50</max><resolution>0.02</resolution></range></lidar>
      </sensor>
      <sensor name="imu" type="imu">
        <always_on>1</always_on><update_rate>100</update_rate><topic>imu</topic>
      </sensor>
    </link>
{steer_link("front_left", fx, hy)}{steer_link("front_right", fx, -hy)}
{wheel("front_left", fx, hy)}{wheel("front_right", fx, -hy)}{wheel("rear_left", rx, hy)}{wheel("rear_right", rx, -hy)}
{rear_joint("rear_left")}{rear_joint("rear_right")}

    <plugin filename="gz-sim-ackermann-steering-system" name="gz::sim::systems::AckermannSteering">
      <left_joint>front_left_wheel_joint</left_joint>
      <left_joint>rear_left_wheel_joint</left_joint>
      <right_joint>front_right_wheel_joint</right_joint>
      <right_joint>rear_right_wheel_joint</right_joint>
      <left_steering_joint>front_left_steering_joint</left_steering_joint>
      <right_steering_joint>front_right_steering_joint</right_steering_joint>
      <kingpin_width>{tw}</kingpin_width>
      <steering_limit>0.576</steering_limit>
      <wheel_base>{wb}</wheel_base>
      <wheel_separation>{tw}</wheel_separation>
      <wheel_radius>{R}</wheel_radius>
      <min_velocity>0</min_velocity>
      <max_velocity>25</max_velocity>
      <min_acceleration>-7.5</min_acceleration>
      <max_acceleration>2.5</max_acceleration>
    </plugin>
    <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
      <odom_frame>world</odom_frame>
      <robot_base_frame>{name}</robot_base_frame>
      <odom_topic>/ego/odom_gt</odom_topic>
      <tf_topic>/ego/tf</tf_topic>
      <odom_publish_frequency>50</odom_publish_frequency>
      <dimensions>2</dimensions>
    </plugin>
    <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"/>
  </model>"""


# ---------------------------------------------------------------------------
# Road users (visual only, gravity off: moved kinematically by traffic_manager)
# ---------------------------------------------------------------------------
def agent_visuals(cls, L, W):
    col = rgb(CLASSES[cls]["color"])
    dark = "0.08 0.08 0.08 1"
    v = []
    if cls in ("car", "bus", "truck", "tractor"):
        h = {"car": 1.45, "bus": 3.1, "truck": 3.0, "tractor": 2.4}[cls]
        v.append(box_visual("body", 0, 0, 0.35 + (h - 0.35) / 2, 0, L, W, h - 0.35, col))
        if cls == "truck":
            v.append(box_visual("cab", L / 2 - 1.0, 0, 1.4, 0, 1.8, W, 2.1, rgb("#6d4c86")))
        for i, wx in enumerate((L / 2 - 0.9, -L / 2 + 0.9)):
            for j, wy in enumerate((W / 2 - 0.1, -W / 2 + 0.1)):
                v.append(cyl_visual(f"w{i}{j}", wx, wy, 0.4, 0.4, 0.3, dark, roll=1.5708))
    elif cls == "auto":
        v.append(box_visual("body", 0, 0, 0.75, 0, L, W, 0.9, col))
        v.append(box_visual("canopy", -0.1, 0, 1.55, 0, L - 0.4, W, 0.7, "0.1 0.1 0.1 1"))
        v.append(cyl_visual("wf", L / 2 - 0.35, 0, 0.25, 0.25, 0.15, dark, roll=1.5708))
        v.append(cyl_visual("wr", -L / 2 + 0.4, 0, 0.25, 0.25, W, dark, roll=1.5708))
    elif cls in ("twowheeler", "bicycle"):
        v.append(box_visual("frame", 0, 0, 0.55, 0, L * 0.9, 0.25, 0.4, col))
        v.append(cyl_visual("wf", L / 2 - 0.3, 0, 0.3, 0.3, 0.1, dark, roll=1.5708))
        v.append(cyl_visual("wr", -L / 2 + 0.3, 0, 0.3, 0.3, 0.1, dark, roll=1.5708))
        v.append(cyl_visual("rider", -0.1, 0, 1.15, 0.2, 0.7, rgb("#3c5a8a")))
        v.append(sphere_visual("head", -0.1, 0, 1.62, 0.13, rgb("#8d5524")))
    elif cls == "pedestrian":
        v.append(cyl_visual("legs", 0, 0, 0.45, 0.14, 0.9, rgb("#2f3e5c")))
        v.append(cyl_visual("torso", 0, 0, 1.2, 0.2, 0.6, col))
        v.append(sphere_visual("head", 0, 0, 1.62, 0.12, rgb("#8d5524")))
    elif cls == "cattle":
        v.append(box_visual("body", 0, 0, 1.05, 0, L * 0.7, W, 0.7, rgb("#e8e2d6")))
        v.append(box_visual("head", L / 2 - 0.2, 0, 1.3, 0, 0.45, 0.35, 0.4, rgb("#d9d1c2")))
        v.append(box_visual("hump", 0.25, 0, 1.45, 0, 0.35, 0.4, 0.2, rgb("#cfc6b5")))
        for i, lx in enumerate((0.5, -0.5)):
            for j, ly in enumerate((0.25, -0.25)):
                v.append(cyl_visual(f"leg{i}{j}", lx, ly, 0.35, 0.07, 0.7, rgb("#bdb4a3")))
    else:  # pushcart, bullockcart
        v.append(box_visual("bed", 0, 0, 0.8, 0, L, W, 0.25, col))
        v.append(box_visual("load", 0, 0, 1.05, 0, L * 0.8, W * 0.8, 0.3, rgb("#c49a6c")))
        v.append(cyl_visual("wl", 0, W / 2, 0.4, 0.4, 0.08, rgb("#5a3d1e"), roll=1.5708))
        v.append(cyl_visual("wr", 0, -W / 2, 0.4, 0.4, 0.08, rgb("#5a3d1e"), roll=1.5708))
        if cls == "bullockcart":
            v.append(box_visual("bullock", L / 2 + 0.9, 0, 0.95, 0, 1.6, 0.7, 0.7, rgb("#9a8a78")))
    return "".join(v)


def agent_model(a, active):
    x, y = (a.x, a.y) if active else PARK_XY
    return f"""
  <model name="agent_{a.id}">
    <pose>{x:.3f} {y:.3f} 0 0 0 {a.yaw:.4f}</pose>
    <link name="link">
      <gravity>false</gravity>
      <inertial><mass>100</mass><inertia><ixx>10</ixx><iyy>10</iyy><izz>10</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      {agent_visuals(a.cls, a.L, a.W)}
    </link>
  </model>"""


# ---------------------------------------------------------------------------
def build(key="S5", seed=0, out=None):
    spec = SCENARIOS[key](seed)
    world = spec.world
    net = world.network
    world.ego = type("E", (), dict(x=spec.ego_init[0], y=spec.ego_init[1], yaw=0, v=0, L=4.4, W=1.8))()
    for a in world.agents:
        if a.behavior is not None and hasattr(a.behavior, "place"):
            a.behavior.place(a)

    rural = key in ("S1", "S5")
    ground = rgb("#6f7f3f") if rural else rgb("#9a9486")
    asphalt = rgb("#4a4a46") if key != "S1" else rgb("#5a5048")
    shoulder = rgb("#8a7456")

    # roads: one static model, many visuals (cheap to render)
    rv = []
    for k, e in enumerate(net.edges.values()):
        P = e["pts"]
        for i, (p0, p1) in enumerate(zip(P[:-1], P[1:])):
            L = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
            cx, cy = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            rv.append(box_visual(f"sh{k}_{i}", cx, cy, 0.004, yaw, L + 0.6, e["width"] + 1.2, 0.006, shoulder))
            rv.append(box_visual(f"rd{k}_{i}", cx, cy, 0.008, yaw, L + 0.3, e["width"], 0.006, asphalt))
        for j, p in enumerate(P):
            rv.append(cyl_visual(f"jn{k}_{j}", p[0], p[1], 0.008, e["width"] / 2, 0.006, asphalt))
    for i, (px, py, pr) in enumerate(world.potholes):
        rv.append(cyl_visual(f"pothole{i}", px, py, 0.013, pr, 0.006, "0.12 0.1 0.08 1"))
    roads = f"""
  <model name="roads"><static>true</static><link name="link">{''.join(rv)}</link></model>"""

    # occluders: buildings / bushes (with collision so nothing drives through them)
    occ = []
    for i, poly in enumerate(world.occluders):
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        sx, sy = max(xs) - min(xs), max(ys) - min(ys)
        h, col = (2.4, rgb("#3f6e2f")) if key == "S5" else (7.0, rgb("#c8b89a"))
        occ.append(f"""
  <model name="occluder_{i}"><static>true</static><pose>{cx:.2f} {cy:.2f} {h/2:.2f} 0 0 0</pose>
    <link name="link">
      <collision name="c"><geometry><box><size>{sx:.2f} {sy:.2f} {h:.2f}</size></box></geometry></collision>
      <visual name="v"><geometry><box><size>{sx:.2f} {sy:.2f} {h:.2f}</size></box></geometry>{mat(col)}</visual>
    </link></model>""")

    agents = "".join(agent_model(a, a.active) for a in world.agents)
    x0, y0, yaw0, _ = spec.ego_init
    ego = ego_model_sdf(x0, y0, yaw0)

    sdf = f"""<?xml version="1.0"?>
<!-- {spec.title} | scenario {key}, seed {seed} | generated by bharat_sim/gen_world.py -->
<sdf version="1.9">
<world name="{WORLD_NAME}">
  <physics name="2ms" type="ignored"><max_step_size>0.002</max_step_size><real_time_factor>1.0</real_time_factor></physics>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
  <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
  <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
  <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
  <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>

  <gravity>0 0 -9.81</gravity>
  <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.62 0.75 0.88 1</background><grid>false</grid><shadows>true</shadows></scene>
  <light type="directional" name="sun">
    <cast_shadows>true</cast_shadows><pose>0 0 50 0 0 0</pose>
    <diffuse>0.95 0.92 0.85 1</diffuse><specular>0.2 0.2 0.2 1</specular>
    <direction>-0.4 0.3 -0.85</direction>
  </light>

  <model name="ground"><static>true</static>
    <link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>2000 2000</size></plane></geometry>
        <surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>2000 2000</size></plane></geometry>{mat(ground)}</visual>
    </link>
  </model>
{roads}
{''.join(occ)}
{agents}
{ego}
</world>
</sdf>
"""
    out = out or os.path.expanduser(f"~/bharat_worlds/{key}_s{seed}.sdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(sdf)
    return out


def write_ego_model(path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "model.sdf"), "w") as f:
        f.write('<?xml version="1.0"?>\n<sdf version="1.9">' + ego_model_sdf() + "\n</sdf>\n")
    with open(os.path.join(path, "model.config"), "w") as f:
        f.write("""<?xml version="1.0"?>
<model><name>ego_car</name><version>1.0</version><sdf version="1.9">model.sdf</sdf>
<description>4.4 m Ackermann hatchback with camera, 3D LiDAR and IMU (Bharat-AV)</description></model>
""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="S5", choices=list(SCENARIOS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(build(a.scenario, a.seed, a.out))


if __name__ == "__main__":
    main()
