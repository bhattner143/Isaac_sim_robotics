import rclpy
from std_msgs.msg import String

rclpy.init()
node = rclpy.create_node('test_node')
pub = node.create_publisher(String, 'test_topic', 10)
msg = String()
msg.data = 'Hello ROS 2!'
pub.publish(msg)
node.destroy_node()
rclpy.shutdown()
print('ROS 2 works!')