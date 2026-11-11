# ROS 2 Jazzy layer

Optional. The plain-Python lab (`agvlab/`) does not need any of this.

```
agvlab/sensor.py --CAN frames--> python-can bus (udp_multicast) --> tape_guide_driver (guide_driver node)
      ^                                                                  |  /guide/track
      |  /cmd_vel                                                        v  /guide/markers
 ros2/sim_node.py  <------------------------------------------  ros2/follower_node.py
 (plant + simulated sensor)                                      (agvlab.follower, no ROS inside)
```

| Path | What |
|---|---|
| `src/guide_msgs/` | message package: `Track.msg` (offsets in m, headings in rad, quality 0..1, flags) and `Markers.msg` |
| `src/tape_guide_driver/` | the driver: node `guide_driver` reads the sensor profile (docs/PROTOCOL.md) from a python-can bus, publishes `/guide/track`, `/guide/markers`, `/guide/state` (NMT state from the heartbeat), warns when the heartbeat is missing. Decoding is `agvlab.frames`. `convert.py` holds the ROS-free message conversions that the plain test suite covers |
| `sim_node.py` | diff-drive plant and simulated sensor. Subscribes `/cmd_vel`, sends TPDO1/2/3 and heartbeat to a python-can bus, writes ground-truth metrics JSON. It never publishes `/guide/*` itself |
| `follower_node.py` | subscribes `/guide/track` and `/guide/markers`, publishes `/cmd_vel`. The control law is `agvlab.follower` |
| `lab.launch.py` | starts the driver (node ID 21, `udp_multicast`), the simulator and the follower; ends when the simulator is done |
| `run_in_container.sh` | `colcon build` of both packages, runs both junction choices, then `check_results.py` |
| `check_results.py` | pass/fail thresholds used by CI |

Run it locally with Docker (same script as CI):

```bash
docker run --rm -v "$PWD":/repo -v "$PWD/results/ros":/out -e REPO=/repo -e OUT=/out \
    ros:jazzy-ros-base bash /repo/ros2/run_in_container.sh
```

On a real robot, run the driver with `ros2 run tape_guide_driver guide_driver --ros-args -p can_interface:=socketcan -p can_channel:=can0 -p node_id:=21`
(`agvlab` and python-can must be importable: `pip install .` from the repo root).

`udp_multicast` is used in CI because it works across processes in a plain container with no kernel
module. On a Linux host with `vcan0`, change `can_interface` / `can_channel` in `lab.launch.py` and
`--interface` / `--channel` for `sim_node.py` to `socketcan` / `vcan0`.
