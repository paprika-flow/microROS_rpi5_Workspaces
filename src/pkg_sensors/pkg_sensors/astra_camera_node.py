# ros2_camera_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')

        self.pub_color = self.create_publisher(Image, '/camera/color/image_raw', 10)

        self.bridge = CvBridge()

        self.cap_color = cv2.VideoCapture(0)
        if not self.cap_color.isOpened():
            self.get_logger().error("Failed to open color camera!")

        # Timer to periodically read and publish images
        self.timer = self.create_timer(0.05, self.publish_images)  # 20 Hz

    def publish_images(self):
        # Publish color image
        ret_color, frame_color = self.cap_color.read()
        if ret_color:
            frame_rgb = cv2.cvtColor(frame_color, cv2.COLOR_BGR2RGB)
            msg_color = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
            self.pub_color.publish(msg_color)


def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.cap_color.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
