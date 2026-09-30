"""Bring up one Bharat-AV scenario in Gazebo Sim 8 (Harmonic).

    ros2 launch bharat_sim sim.launch.py                    # S5 cattle crossing, seed 0
    ros2 launch bharat_sim sim.launch.py scenario:=S2 seed:=3
    ros2 launch bharat_sim sim.launch.py scenario:=S4 rviz:=false

Starts: world generation -> gz sim -> ros_gz_bridge -> static TFs ->
traffic_manager -> autonomy -> (rviz2).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
                            TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def setup(context, *args, **kwargs):
    from bharat_sim.gen_world import build

    scenario = LaunchConfiguration("scenario").perform(context)
    seed = int(LaunchConfiguration("seed").perform(context))
    world_path = build(scenario, seed)                    # writes ~/bharat_worlds/<S>_s<seed>.sdf
    share = get_package_share_directory("bharat_sim")
    gz_share = get_package_share_directory("ros_gz_sim")
    params = [{"use_sim_time": True, "scenario": scenario, "seed": seed}]

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gz_share, "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": f"-r {world_path}", "on_exit_shutdown": "true"}.items())
    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", name="bridge",
                  output="screen",
                  parameters=[{"config_file": os.path.join(share, "config", "bridge.yaml"),
                               "use_sim_time": True}])
    # static frames so RViz can show markers, the LiDAR cloud and the camera
    tf_world = Node(package="tf2_ros", executable="static_transform_publisher", name="tf_world",
                    arguments=["--frame-id", "map", "--child-frame-id", "world"])
    tf_lidar = Node(package="tf2_ros", executable="static_transform_publisher", name="tf_lidar",
                    arguments=["--z", "1.94", "--frame-id", "ego_car", "--child-frame-id", "lidar_link"])
    tf_cam = Node(package="tf2_ros", executable="static_transform_publisher", name="tf_cam",
                  arguments=["--x", "1.3", "--z", "1.29", "--frame-id", "ego_car",
                             "--child-frame-id", "camera_link"])
    tm = Node(package="bharat_sim", executable="traffic_manager", output="screen", parameters=params)
    av = Node(package="bharat_sim", executable="autonomy", output="screen", parameters=params)
    rviz = Node(package="rviz2", executable="rviz2", output="log",
                arguments=["-d", os.path.join(share, "rviz", "bharat.rviz")],
                parameters=[{"use_sim_time": True}],
                condition=IfCondition(LaunchConfiguration("rviz")))
    # give Gazebo a few seconds to load the world before the stack starts
    return [gz, bridge, tf_world, tf_lidar, tf_cam,
            TimerAction(period=4.0, actions=[tm, av]), rviz]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("scenario", default_value="S5",
                              description="S1 village, S2 junction, S3 highway, S4 market, S5 cattle, S6 dart-out"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("rviz", default_value="true"),
        OpaqueFunction(function=setup),
    ])
