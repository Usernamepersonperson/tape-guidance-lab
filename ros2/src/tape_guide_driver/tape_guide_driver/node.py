"""ROS 2 driver node: tape-guide sensor CAN frames -> /guide/track, /guide/markers, /guide/state.

Parameters: can_interface (python-can interface, default socketcan), can_channel (default can0),
node_id (default 21), frame_id (default guide_link), heartbeat_timeout_s (default 3.0).
Decoding is `agvlab.frames`, the reference implementation of docs/PROTOCOL.md.
"""

import threading
import time

import can
import rclpy
from guide_msgs.msg import Markers, Track
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import UInt8

from agvlab import frames
from tape_guide_driver import convert


class GuideDriver(Node):
    def __init__(self):
        super().__init__("guide_driver")
        self.declare_parameter("can_interface", "socketcan")
        self.declare_parameter("can_channel", "can0")
        self.declare_parameter("node_id", frames.DEFAULT_NODE_ID)
        self.declare_parameter("frame_id", "guide_link")
        self.declare_parameter("heartbeat_timeout_s", 3.0)
        self.node_id = int(self.get_parameter("node_id").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.hb_timeout = float(self.get_parameter("heartbeat_timeout_s").value)
        self.pub_track = self.create_publisher(Track, "/guide/track", qos_profile_sensor_data)
        self.pub_markers = self.create_publisher(Markers, "/guide/markers", qos_profile_sensor_data)
        self.pub_state = self.create_publisher(UInt8, "/guide/state", 10)
        iface = self.get_parameter("can_interface").value
        chan = self.get_parameter("can_channel").value
        self.bus = can.Bus(interface=iface, channel=chan)
        self.last_hb = time.monotonic()
        self.hb_warned = False
        self.decode_errors = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._rx_loop, name="guide-can-rx", daemon=True)
        self._thread.start()
        self.create_timer(1.0, self._watchdog)
        self.get_logger().info(f"guide driver up: node {self.node_id}, {iface}/{chan}")

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            msg = self.bus.recv(timeout=0.05)
            if msg is None:
                continue
            try:
                dec = frames.decode_frame(msg.arbitration_id, bytes(msg.data), self.node_id)
            except frames.DecodeError:
                self.decode_errors += 1
                continue
            if dec is None:
                continue
            stamp = self.get_clock().now().to_msg()
            if isinstance(dec, frames.TrackFrame):
                m = Track()
                m.header.stamp, m.header.frame_id = stamp, self.frame_id
                for k, v in convert.track_fields(dec).items():
                    setattr(m, k, v)
                self.pub_track.publish(m)
            elif isinstance(dec, frames.MarkerFrame):
                m = Markers()
                m.header.stamp, m.header.frame_id = stamp, self.frame_id
                for k, v in convert.markers_fields(dec).items():
                    setattr(m, k, v)
                self.pub_markers.publish(m)
            elif isinstance(dec, frames.HeartbeatFrame):
                self.last_hb = time.monotonic()
                self.hb_warned = False
                self.pub_state.publish(UInt8(data=dec.state))

    def _watchdog(self) -> None:
        if time.monotonic() - self.last_hb > self.hb_timeout and not self.hb_warned:
            self.hb_warned = True
            self.get_logger().warn(f"no heartbeat from node {self.node_id} for {self.hb_timeout:.0f} s")

    def destroy_node(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.bus.shutdown()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = GuideDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
