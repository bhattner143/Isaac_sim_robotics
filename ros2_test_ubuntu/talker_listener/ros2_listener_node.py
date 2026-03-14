"""
ROS 2 Listener Node — System Python 3.12
=========================================
Subscribes to /drake_hello and prints every message received from Drake.

DO NOT run this script directly. Use script_drake_listener.py which
launches this via subprocess with the correct ROS 2 environment.

Architecture:
    /drake_hello (ROS 2 topic)
        └─ subscribed by ──► ros2_drake_listener_node.py (system Python 3.12)
                                └─ launched by script_drake_listener.py (conda Python 3.11)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DrakeListenerNode(Node):
    """
    ROS 2 subscriber node.

    Listens on /drake_hello topic and prints each Drake
    state message with a received counter.
    """

    def __init__(self):
        super().__init__('drake_listener_node')

        self.subscription_ = self.create_subscription(
            String,
            '/drake_hello',
            self._callback,
            10,
        )
        self._count = 0
        self.get_logger().info(
            'Drake Listener Node ready — listening on /drake_hello'
        )

    def _callback(self, msg: String):
        """Handle incoming message from Drake talker."""
        self._count += 1
        self.get_logger().info(
            f'[msg #{self._count}] Received from Drake: "{msg.data}"'
        )


def main():
    rclpy.init()
    node = DrakeListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Drake Listener shutting down — '
            f'total messages received: {node._count}'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
