#!/usr/bin/env python3
"""ROS 2 node: robot plant + simulated tape-guide sensor on a python-can bus.

Subscribes /cmd_vel (geometry_msgs/Twist), integrates the diff-drive plant in wall-clock time,
and sends the sensor TPDO1/2/3 + heartbeat frames to a python-can bus. The tape_guide_driver
node (started separately) reads that bus and publishes /guide/track etc.; this node
never publishes those topics itself. At the end it writes ground-truth metrics as JSON.
"""

import argparse
import json
import math
import sys
import threading
import time

import can
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from agvlab.lab import RunConfig, World, score
from agvlab.layout import default_layout, nearest_on_polyline
from agvlab.live import to_message


class SimNode(Node):
    def __init__(self, args):
        super().__init__("agvlab_sim")
        self.args = args
        cfg = RunConfig(route=args.route, seed=args.seed)
        self.cfg = cfg
        self.layout = default_layout()
        self.world = World(self.layout, cfg.sensor, cfg.plant, cfg.seed, cfg.start_offset_mm,
                           cfg.start_heading_deg)
        self.bus = can.Bus(interface=args.interface, channel=args.channel)
        self.cmd = (0.0, 0.0)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.dt = 0.005
        self.t0 = time.monotonic()
        self.t_prev = self.t0
        self.t_last_cmd = self.t0
        self.err, self.ts, self.vs = [], [], []
        self.route_path = self.layout.routes[args.route]
        self.stopped_since = None
        self.sent = 0
        self.done = False
        self.n = 0
        self.get_logger().info(f"sim up: route={args.route}, bus={args.interface}/{args.channel}")

    def _on_cmd(self, msg: Twist) -> None:
        self.cmd = (msg.linear.x, msg.angular.z)
        self.t_last_cmd = time.monotonic()

    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(now - self.t_prev, 0.05)
        self.t_prev = now
        v, w = self.cmd
        for fr in self.world.step(v, w, dt):
            self.bus.send(to_message(fr))
            self.sent += 1
        self.n += 1
        t = now - self.t0
        if self.n % 2 == 0:  # ground truth at 100 Hz is plenty and keeps the loop cheap
            _, signed, _ = nearest_on_polyline(self.route_path, self.world.sensor_point())
            self.ts.append(t)
            self.err.append(signed * 1000.0)
            self.vs.append(self.world.plant.v)
        stopped = self.world.plant.v < 1e-3 and t > 5.0 and self.cmd[0] == 0.0
        self.stopped_since = (self.stopped_since or now) if stopped else None
        timeout = t > self.args.max_time
        if (self.stopped_since and now - self.stopped_since > 1.0) or timeout:
            self._finish(timeout)

    def run(self) -> None:
        """Pace physics and CAN on this thread; a 5 ms rclpy timer drifted to ~15 ms under load."""
        # rclpy spins on a helper thread so /cmd_vel callbacks keep flowing.
        threading.Thread(target=rclpy.spin, args=(self,), daemon=True).start()
        next_tick = time.perf_counter()
        while rclpy.ok() and not self.done:
            next_tick += self.dt
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self._tick()

    def _finish(self, timeout: bool) -> None:
        if self.done:
            return
        self.done = True
        w = self.world
        # the follower node's own `stopped` flag is not visible here; a stop is "robot at rest, no command"
        class _F:  # minimal stand-in for score()
            stopped = not timeout
            lost_events = 0
            markers_seen = 2
        m = score(w, _F, self.layout, self.args.route, self.ts, self.err, self.vs, 0, self.ts[-1], 1.0, False)
        m["timeout"] = timeout
        m["frames_sent"] = self.sent
        for k, val in list(m.items()):
            if isinstance(val, float) and math.isnan(val):
                m[k] = None
        with open(self.args.out, "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=1)
        self.get_logger().info(f"sim done: {m}")
        self.bus.shutdown()
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="branch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interface", default="udp_multicast")
    ap.add_argument("--channel", default="239.74.163.77")
    ap.add_argument("--max-time", type=float, default=90.0)
    ap.add_argument("--out", default="ros_sim_result.json")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = SimNode(args)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
