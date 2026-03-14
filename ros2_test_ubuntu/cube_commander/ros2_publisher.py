"""
ROS 2 Cube Commander Node — System Python 3.12
===============================================
Reads 'CUBE_POS:x,y,z' lines from stdin (piped from drake_logic.py)
and publishes them as geometry_msgs/Point on /cube_target_pos.

DO NOT run this script directly. Use cube_commander/run_commander.sh.

Architecture:
    drake_logic.py (conda 3.11)
        └─ stdout pipe ──► ros2_publisher.py (system 3.12)
                                └─ publishes ──► /cube_target_pos
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point

# BEST_EFFORT + depth=1: always deliver the latest command, never queue stale ones.
# Critical for impedance control — a delayed force command is worse than no command.
_CONTROL_QOS = QoSProfile(
    reliability = ReliabilityPolicy.BEST_EFFORT,
    history     = HistoryPolicy.KEEP_LAST,
    depth       = 1,
)


class CubeCommanderNode(Node):
    """
    ROS 2 publisher node.

    Reads 'CUBE_POS:x,y,z' from stdin (piped from Drake)
    and publishes geometry_msgs/Point to /cube_target_pos.
    """

    def __init__(self):
        super().__init__('cube_commander_node')

        self.publisher_ = self.create_publisher(Point, '/cube_target_pos', _CONTROL_QOS)
        self.get_logger().info(
            'Cube Commander Node ready — publishing to /cube_target_pos '
            '[BEST_EFFORT, depth=1, 200 Hz]'
        )

        # Read stdin from Drake in background thread
        self._stdin_thread = threading.Thread(
            target=self._stdin_loop, daemon=True
        )
        self._stdin_thread.start()

    def _stdin_loop(self):
        """Read CUBE_POS lines from Drake and publish as geometry_msgs/Point."""
        for line in sys.stdin:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse: CUBE_POS:x,y,z
            if not line.startswith('CUBE_POS:'):
                self.get_logger().warn(f'Unknown message format: "{line}"')
                continue

            try:
                coords = line.replace('CUBE_POS:', '').split(',')
                x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
            except (ValueError, IndexError) as e:
                self.get_logger().error(f'Failed to parse "{line}": {e}')
                continue

            msg = Point()
            msg.x = x
            msg.y = y
            msg.z = z
            self.publisher_.publish(msg)

            self.get_logger().info(
                f'Published cube target → x={x*100:.1f}cm, y={y*100:.1f}cm, z={z*100:.1f}cm'
            )


def main():
    rclpy.init()
    node = CubeCommanderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Cube Commander Node shutting down.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
