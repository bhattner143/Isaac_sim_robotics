#!/usr/bin/env python3
"""
ROS 2 Manipulator Publisher — System Python 3.12
=================================================
Reads 'JOINT_CMD:q1,q2' or 'EE_CMD:x,y' lines from stdin
(piped from drake_logic.py) and publishes them to the appropriate
ROS 2 topic:
  JOINT_CMD → /manip/joint_command (sensor_msgs/JointState)
  EE_CMD    → /manip/ee_command    (geometry_msgs/Point)

DO NOT run this script directly. Use run_drake_commander.sh.

Architecture:
    drake_logic.py (conda 3.11)
        └─ stdout pipe ──► ros2_publisher.py (system 3.12)
                                └─ publishes ──► /manip/joint_command OR /manip/ee_command
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState

_CONTROL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class ManipCommanderNode(Node):
    """
    Reads Drake stdout lines and publishes to ROS 2 topics.

    JOINT_CMD:q1,q2  → /manip/joint_command (JointState)
    EE_CMD:x,y       → /manip/ee_command    (Point)
    """

    def __init__(self):
        super().__init__('manip_commander_node')

        self._joint_pub = self.create_publisher(
            JointState, '/manip/joint_command', _CONTROL_QOS,
        )
        self._ee_pub = self.create_publisher(
            Point, '/manip/ee_command', _CONTROL_QOS,
        )
        self.get_logger().info(
            'Manip Commander Node ready — publishing to /manip/joint_command, /manip/ee_command'
        )

        self._stdin_thread = threading.Thread(
            target=self._stdin_loop, daemon=True,
        )
        self._stdin_thread.start()

    def _stdin_loop(self):
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('JOINT_CMD:'):
                self._handle_joint(line)
            elif line.startswith('EE_CMD:'):
                self._handle_ee(line)
            else:
                self.get_logger().warn(f'Unknown: "{line}"')

        self.get_logger().info('stdin EOF — Drake process ended.')
        rclpy.shutdown()

    def _handle_joint(self, line: str):
        try:
            parts = line.replace('JOINT_CMD:', '').split(',')
            q1, q2 = float(parts[0]), float(parts[1])
        except (ValueError, IndexError) as e:
            self.get_logger().error(f'Parse error: "{line}" — {e}')
            return

        msg = JointState()
        msg.name = ['link1_base', 'link2_link1']
        msg.position = [q1, q2]
        self._joint_pub.publish(msg)

    def _handle_ee(self, line: str):
        try:
            parts = line.replace('EE_CMD:', '').split(',')
            x, y = float(parts[0]), float(parts[1])
        except (ValueError, IndexError) as e:
            self.get_logger().error(f'Parse error: "{line}" — {e}')
            return

        msg = Point()
        msg.x = x
        msg.y = y
        msg.z = 0.0
        self._ee_pub.publish(msg)


def main():
    rclpy.init()
    node = ManipCommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down.')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
