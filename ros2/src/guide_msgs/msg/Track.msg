# Tape track seen by the guide sensor (SI units, ROS REP-103: lateral offsets left positive,
# headings counter-clockwise positive). On a single tape the left and right values are equal.
std_msgs/Header header
bool detected
bool two_tracks
bool fork
bool merge
bool junction
bool marker_left
bool marker_right
bool fault
float32 offset_left_m
float32 offset_right_m
float32 heading_left_rad
float32 heading_right_rad
float32 quality          # 0.0 .. 1.0
