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
