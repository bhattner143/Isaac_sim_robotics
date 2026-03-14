"""
ROS 2 Cube Subscriber Node — System Python 3.12
===============================================
Subscribes to /cube_target_pos (geometry_msgs/Point) and forwards
each position as 'CUBE_POS:x,y,z' to stdout to be piped to
the Isaac Sim script.

DO NOT run this script directly. Use cube_commander/run_isaac.sh.

Architecture:
    /cube_target_pos (ROS 2 topic, geometry_msgs/Point)
        └─ subscribed by ──► ros2_subscriber.py (system 3.12)
                                └─ stdout pipe ──► isaac_sim.py (env_isaacsim)
                                                        └─ moves cube in Isaac Sim
"""

import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point

# Must match the publisher QoS exactly — mismatched QoS causes DDS to reject
# the connection silently and no messages are delivered.
_CONTROL_QOS = QoSProfile(
    reliability = ReliabilityPolicy.BEST_EFFORT,
    history     = HistoryPolicy.KEEP_LAST,
    depth       = 1,
)


class CubeListenerNode(Node):
    """
    ROS 2 subscriber node.

    Subscribes to /cube_target_pos and prints 'CUBE_POS:x,y,z'
    to stdout → piped to Isaac Sim test script.
    """

    def __init__(self):
        super().__init__('cube_listener_node')

        self.subscription_ = self.create_subscription(
            Point,
            '/cube_target_pos',
            self._callback,
            _CONTROL_QOS,
        )
        self._count = 0
        self.get_logger().info(
            'Cube Listener Node ready — subscribing to /cube_target_pos '
            '[BEST_EFFORT, depth=1, 200 Hz]'
        )

    def _callback(self, msg: Point):
        """Receive position from Drake, forward to Isaac Sim via stdout."""
        self._count += 1

        self.get_logger().info(
            f'[#{self._count}] Cube target received: '
            f'x={msg.x*100:.1f}cm, y={msg.y*100:.1f}cm, z={msg.z*100:.1f}cm'
        )

        # Forward to Isaac Sim via stdout pipe
        # Format: CUBE_POS:x,y,z
        print(f'CUBE_POS:{msg.x:.4f},{msg.y:.4f},{msg.z:.4f}', flush=True)


def main():
    rclpy.init()
    node = CubeListenerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(
            f'Cube Listener shutting down — '
            f'total positions received: {node._count}'
        )
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
