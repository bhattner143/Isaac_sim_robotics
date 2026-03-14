"""
ROS 2 Talker Node — System Python 3.12
=======================================
Reads Drake state messages from stdin (piped from script_drake_talker.py)
and publishes them to the /drake_hello ROS 2 topic.

DO NOT run this script directly. Use script_drake_talker.py which
launches this via subprocess pipe.

Architecture:
    script_drake_talker.py (conda Python 3.11, Drake)
        └─ stdout pipe ──► ros2_drake_talker_node.py (system Python 3.12, rclpy)
                                └─ publishes ──► /drake_hello (ROS 2 topic)
"""

import sys
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DrakeTalkerNode(Node):
    """
    ROS 2 publisher node.

    Reads lines from stdin (piped from Drake conda process) and
    publishes each line as a std_msgs/String on /drake_hello.
    """

    def __init__(self):
        super().__init__('drake_talker_node')

        self.publisher_ = self.create_publisher(String, '/drake_hello', 10)
        self.get_logger().info(
            'Drake Talker Node ready — publishing to /drake_hello'
        )

        # Read stdin in a daemon thread so rclpy.spin() runs normally
        self._stdin_thread = threading.Thread(
            target=self._stdin_loop, daemon=True
        )
        self._stdin_thread.start()

    def _stdin_loop(self):
        """Read Drake messages from stdin pipe and publish to ROS 2."""
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith('#'):
                # Skip empty lines and comments (Drake prints # lines to stderr)
                continue

            msg = String()
            msg.data = line
            self.publisher_.publish(msg)
            self.get_logger().info(f'Published → "{line}"')


def main():
    rclpy.init()
    node = DrakeTalkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Drake Talker Node shutting down.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
