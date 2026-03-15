"""
ROS 2 Topic Name Registry
==========================
Single source of truth for all topic names and message types used
across scenarios. Import this in every ROS 2 node so that renaming
a topic is always a one-line change here, not a hunt across multiple files.

Usage:
    from common.topics import CUBE_TARGET_POS, DRAKE_HELLO
"""

# ── Cube Commander pipeline ───────────────────────────────────────────────────
# Drake → /cube_target_pos → Isaac Sim
CUBE_TARGET_POS = "/cube_target_pos"          # geometry_msgs/Point

# ── Talker / Listener pipeline ────────────────────────────────────────────────
# Drake → /drake_hello → Listener
DRAKE_HELLO = "/drake_hello"                  # std_msgs/String

# ── Cup Manipulator Tendon pipeline ───────────────────────────────────────────
# Mode 1 (joint_command): Drake publishes joints → Isaac Sim applies → publishes EE
# Mode 2 (ee_command):    Drake publishes EE    → Isaac Sim IK+apply → publishes joints
MANIP_JOINT_COMMAND  = "/manip/joint_command"   # sensor_msgs/JointState  (position only)
MANIP_EE_COMMAND     = "/manip/ee_command"      # geometry_msgs/Point     (target x, y)
MANIP_JOINT_STATE    = "/manip/joint_state"     # sensor_msgs/JointState  (position + velocity)
MANIP_EE_POSITION    = "/manip/ee_position"     # geometry_msgs/Point     (actual x, y, z)
