#!/usr/bin/env python3
"""ROS 2 node: tape follower on the tape_guide_driver topics.

Subscribes /guide/track (guide_msgs/Track) and /guide/markers (guide_msgs/Markers),
publishes /cmd_vel (geometry_msgs/Twist). The control law is `agvlab.follower` (no ROS inside),
so the same code runs in the plain-Python lab and here.
"""

import argparse
import json
import statistics
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from guide_msgs.msg import Markers, Track
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from agvlab.follower import Follower, FollowerConfig
from tape_guide_driver import convert


class FollowerNode(Node):
    def __init__(self, args):
        super().__init__("agvlab_follower")
        self.args = args
        self.core = Follower(FollowerConfig(side=args.side))
        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Track, "/guide/track", self._on_track, qos_profile_sensor_data)
        self.create_subscription(Markers, "/guide/markers", self._on_markers, qos_profile_sensor_data)
        self.t0 = time.monotonic()
        self.rx_t = []
        self.create_timer(0.02, self._tick)  # 50 Hz, control rate

    def _now(self) -> float:
        return time.monotonic() - self.t0

    def _on_track(self, msg: Track) -> None:
        self.rx_t.append(self._now())
        self.core.on_track(self._now(), convert.frame_from_track(msg))

    def _on_markers(self, msg: Markers) -> None:
        self.core.on_markers(self._now(), convert.frame_from_markers(msg))

    def _tick(self) -> None:
        v, w = self.core.update(self._now())
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.pub.publish(cmd)

    def write_result(self) -> None:
        iv = [(b - a) * 1000.0 for a, b in zip(self.rx_t, self.rx_t[1:])]
        span = self.rx_t[-1] - self.rx_t[0] if len(self.rx_t) > 1 else 0.0
        res = {
            "track_msgs": len(self.rx_t),
            "track_rate_hz": (len(self.rx_t) - 1) / span if span > 0 else 0.0,
            "track_interval_ms_mean": statistics.fmean(iv) if iv else None,
            "track_interval_ms_sd": statistics.pstdev(iv) if iv else None,
            "follower_stopped": self.core.stopped,
            "markers_seen": self.core.markers_seen,
            "lost_events": self.core.lost_events,
        }
        with open(self.args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        self.get_logger().info(f"follower result: {res}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", help="track to follow after the fork: right = branch")
    ap.add_argument("--out", default="ros_follower_result.json")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = FollowerNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_result()
        node.destroy_node()
        rclpy.try_shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
