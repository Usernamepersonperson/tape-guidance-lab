#!/usr/bin/env bash
# Build guide_msgs + tape_guide_driver and run the closed loop against the simulated sensor.
# Meant for the official ros:jazzy-ros-base image (CI job "ros2-jazzy"); also works locally:
#   docker run --rm -v "$PWD":/repo -v "$PWD/results/ros":/out -e REPO=/repo -e OUT=/out \
#       ros:jazzy-ros-base bash /repo/ros2/run_in_container.sh
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="${OUT:-/tmp/agvlab}"
WS="${WS:-/tmp/ws}"

apt-get update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip python3-msgpack \
    python3-colcon-common-extensions
python3 -m pip install -q --break-system-packages "python-can>=4.4"

mkdir -p "$WS/src" "$OUT"
cp -r "$REPO/ros2/src/." "$WS/src/"

set +u
source /opt/ros/jazzy/setup.bash
set -u
(cd "$WS" && colcon build --packages-select guide_msgs tape_guide_driver --event-handlers console_direct-)
set +u
source "$WS/install/setup.bash"
set -u
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

status=0
for route in branch straight; do
  mkdir -p "$OUT/$route"
  echo "=== closed loop, route=$route"
  timeout 150 ros2 launch "$REPO/ros2/lab.launch.py" route:="$route" out_dir:="$OUT/$route" \
      can_channel:="239.74.163.$((77 + $(test "$route" = branch && echo 0 || echo 1)))" || true
  python3 "$REPO/ros2/check_results.py" "$OUT/$route" || status=1
done
exit $status
