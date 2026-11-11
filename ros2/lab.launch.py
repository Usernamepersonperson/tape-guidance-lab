"""Launch the tape_guide_driver + the simulated guide sensor + the follower, closed loop, on one machine.

    ros2 launch ros2/lab.launch.py route:=branch out_dir:=/tmp/agvlab

The simulator sends CAN frames over python-can `udp_multicast` (works in a container, needs no
kernel module); swap `can_interface`/`can_channel` for socketcan/vcan0 on a Linux host.
Ends everything when the simulator has finished.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.abspath(__file__))


def generate_launch_description():
    route = LaunchConfiguration("route")
    out_dir = LaunchConfiguration("out_dir")
    group = LaunchConfiguration("can_channel")
    py = "python3"

    driver = Node(
        package="tape_guide_driver", executable="guide_driver", name="guide_driver", output="screen",
        parameters=[{"can_interface": "udp_multicast", "can_channel": group, "node_id": 21,
                     "frame_id": "guide_link"}])
    sim = ExecuteProcess(
        cmd=[py, os.path.join(HERE, "sim_node.py"), "--route", route, "--channel", group,
             "--out", [out_dir, "/sim.json"]],
        output="screen")
    follower = ExecuteProcess(
        cmd=[py, os.path.join(HERE, "follower_node.py"),
             "--side", PythonExpression(["'right' if '", route, "' == 'branch' else 'left'"]),
             "--out", [out_dir, "/follower.json"]],
        output="screen")
    stop_all = RegisterEventHandler(OnProcessExit(target_action=sim, on_exit=[EmitEvent(event=Shutdown())]))
    return LaunchDescription([
        DeclareLaunchArgument("route", default_value="branch"),
        DeclareLaunchArgument("out_dir", default_value="/tmp/agvlab"),
        DeclareLaunchArgument("can_channel", default_value="239.74.163.77"),
        driver, follower, sim, stop_all,
    ])
