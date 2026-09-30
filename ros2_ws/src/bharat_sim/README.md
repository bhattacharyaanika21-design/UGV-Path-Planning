# bharat_sim: Bharat-AV in Gazebo Sim 8 (Harmonic)

This package puts the validated Python autonomy stack (fusion, prediction, decision
logic, rerouting and the lane-free lattice planner) into a 3D Gazebo world, and drives a
physically simulated Ackermann car with it. All six scenarios, S1 to S6, are generated
automatically from the same Python scenario definitions. That keeps Gazebo and the
Python Monte-Carlo results directly comparable.

```
 gen_world.py ──► ~/bharat_worlds/S5_s0.sdf ──► gz sim (physics, camera, LiDAR, IMU)
                                                   │  ▲
                        /ego/odom (ros_gz_bridge)  │  │ /model/ego_car/cmd_vel
                                                   ▼  │
 traffic_manager ──set_pose──► road-user models   autonomy (AutonomyStack)
      │  (world.py behaviours)                     ▲       │
      └──────────── /gt/agents (JSON) ─────────────┘       └─► /av/markers, /av/status → RViz
```

---

## Step 0: Check what you have

```bash
gz sim --versions        # should list 8.x  (Gazebo Sim 8 = Harmonic)
lsb_release -a           # 24.04 -> ROS 2 Jazzy (recommended pairing for Harmonic)
                         # 22.04 -> ROS 2 Humble + the "gzharmonic" packages (see note)
```

| Ubuntu | ROS 2 | Install for the bridge |
|---|---|---|
| 24.04 | Jazzy | `sudo apt install ros-jazzy-ros-gz` |
| 22.04 | Humble | `sudo apt install ros-humble-ros-gzharmonic` (not the default `ros-humble-ros-gz`, which is built for Fortress) |

Everything below uses `jazzy`. On Humble, replace `jazzy` with `humble` in the package names.

## Step 1: Install ROS 2 and the pieces we need

Install ROS 2 Jazzy first by following docs.ros.org (Ubuntu deb packages), then:

```bash
sudo apt update
sudo apt install ros-jazzy-desktop ros-jazzy-ros-gz ros-jazzy-rviz2 \
                 python3-colcon-common-extensions python3-numpy python3-scipy
# gz-transport Python bindings, so road users move smoothly at 20 Hz
# (from the Gazebo apt repo you installed Gazebo Sim 8 from):
sudo apt install python3-gz-transport13 python3-gz-msgs10
```

If the Python bindings are not available, the traffic manager falls back to the
`gz service` command-line tool automatically. It still works, but road users only update
about 5 times a second.

## Step 2: Build the workspace

The repository already contains a ROS 2 workspace (`ros2_ws/`):

```bash
cd bharat-av/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
echo "source $(pwd)/install/setup.bash" >> ~/.bashrc   # optional
```

## Step 3: Offline rehearsal (no Gazebo, 1 minute)

This runs the exact node logic against a stand-in for Gazebo's Ackermann car. It proves
your Python environment is fine before any 3D is involved.

```bash
cd bharat-av/ros2_ws/src/bharat_sim
python3 test_offline.py S5 S6        # expect PASS lines
```

## Step 4: Open a scenario in Gazebo on its own

```bash
ros2 run bharat_sim gen_world --scenario S5 --seed 0
gz sim -r ~/bharat_worlds/S5_s0.sdf
```

You should see the rural road, bushes, a herd of cattle, a truck, a following car, and
the cyan ego car. In a **second terminal**, drive the car by hand:

```bash
gz topic -t /model/ego_car/cmd_vel -m gz.msgs.Twist -p "linear: {x: 3.0}, angular: {z: 0.15}"
gz topic -t /model/ego_car/cmd_vel -m gz.msgs.Twist -p "linear: {x: 0.0}"      # stop
gz topic -l | grep -E "camera|lidar|imu|odom"                                  # sensors alive?
gz topic -e -t /ego/odom_gt -n 1                                               # odometry
```

Move a road user by hand. This is exactly what the traffic manager does 20 times a
second:

```bash
gz service -s /world/indian_roads/set_pose --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean \
  --timeout 1000 --req 'name: "agent_1", position: {x: 40, y: 2, z: 0}'
```

If the car drives and the cow jumps, the world is good. Close Gazebo.

## Step 5: Run the full stack

```bash
ros2 launch bharat_sim sim.launch.py scenario:=S5 seed:=0
```

This one command:
1. generates the world;
2. starts Gazebo;
3. starts the bridge;
4. starts `traffic_manager` (road users move);
5. starts `autonomy` (the car drives itself);
6. opens RViz.

The stack starts 4 s after launch.

In RViz (fixed frame `world`):

| Colour | Shows |
|---|---|
| Green line | Chosen path (red during an EMERGENCY) |
| Pale lines | Candidate paths |
| Orange lines | Estimated road edges |
| White line | The route |
| Coloured boxes | Fused tracks, with class and speed labels |
| Red dots | Hard predictions |
| Orange dots | Soft predictions ("might dart", "might swerve") |

The coloured text over the car shows the decision state (CRUISE, YIELD, WAIT, NUDGE…)
and the speed and target speed. The LiDAR point cloud and the front-camera image are
shown too.

Watch the decisions live:

```bash
ros2 topic echo /av/status
```

Other scenarios:

```bash
ros2 launch bharat_sim sim.launch.py scenario:=S1          # village road
ros2 launch bharat_sim sim.launch.py scenario:=S2          # unsignalised junction
ros2 launch bharat_sim sim.launch.py scenario:=S3          # highway merge
ros2 launch bharat_sim sim.launch.py scenario:=S4          # market + reroute
ros2 launch bharat_sim sim.launch.py scenario:=S6          # child dart-out
ros2 launch bharat_sim sim.launch.py scenario:=S2 seed:=3 rviz:=false
```

## Step 6: Results

Every run writes `~/bharat_runs/<scenario>_s<seed>_<time>/` containing:
- `ego_log.csv`: 10 Hz log of position, speed, state, target speed, clearance, replan
  reasons, planner time and safe-path count;
- `metrics.json`: whether it completed, any collision, time, minimum clearance, replan
  latency (mean and p95), emergency count and reroute events.

Collisions are checked every cycle from the Gazebo ground-truth poses, using the same
rectangle test as the Python Monte-Carlo.

To run a batch:

```bash
for s in S1 S2 S3 S4 S5 S6; do for k in 0 1 2; do
  timeout 300 ros2 launch bharat_sim sim.launch.py scenario:=$s seed:=$k rviz:=false
done; done
```

The autonomy node reports `RUN FINISHED`. Press Ctrl-C, or let `timeout` end the run.

For demo footage, use the Gazebo GUI's Video Recorder (top-right plugin menu), or
record the RViz window.

## Step 7 (next stage): real perception from the Gazebo sensors

For now, `autonomy` feeds its camera/LiDAR/radar **models** (from `av/sensors.py`) with
Gazebo ground truth, with occlusion. This matches the validated Python results. Upgrade
one sensor at a time and compare metrics after each:

1. **LiDAR**: a node that subscribes to `/lidar/points`, removes the ground (z < 0.2 m),
   clusters the remaining points (grid or DBSCAN), fits an L-shaped box to each cluster,
   and publishes detections. In `AutonomyStack.cycle`, replace the LiDAR part of
   `sensors.sense()` with those detections.
2. **Camera**: a YOLO model trained on the IDD dataset plus frames captured from your
   own Gazebo worlds, subscribing to `/camera/image`. Associate each box with LiDAR points
   to get its range.
3. **Radar**: keep the model. Gazebo has no automotive radar sensor, and the report
   should say so.

## Troubleshooting

| Symptom | Check / fix |
|---|---|
| Car does not move | `ros2 topic info /model/ego_car/cmd_vel` must show the bridge as a subscriber. Try Step 4's `gz topic` command. |
| Car drifts or wheels spin in place | Look for AckermannSteering errors in the gz console. The joint names in `gen_world.py` must match the plugin block. |
| Road users do not move | Read the `traffic_manager` log. Without the Python bindings it uses the CLI (slower). Test Step 4's `gz service` by hand. |
| RViz shows "Frame [world] does not exist" | The static TF nodes did not start. Check `ros2 run tf2_ros tf2_echo map world`. |
| LiDAR not shown in RViz | Set the display's frame from `ros2 topic echo /lidar/points --field header.frame_id --once`, or add a static TF for that frame name. |
| Low real-time factor | In `gen_world.py`, set `<shadows>false`, lower the camera to 640×360, and drop LiDAR horizontal samples to 450. |
| "unrecognised element gz_frame_id" warning | Harmless. |
| Humble + Fortress instead of Harmonic | Plugin names change: `ignition-gazebo-*-system` with `ignition::gazebo::systems::*`. Harmonic is recommended. |
