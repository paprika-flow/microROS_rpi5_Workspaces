# sidewalk_follower_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import torch

import numpy as np
import cv2

from pkg_FastSCNN.fast_scnn import get_fast_scnn
from pkg_FastSCNN.utils.image_processor import segmentation_inference

from .utils.getEdges import get_edges
from .utils.PID import PID_sidewalk
# from .utils.checking_if_good import good_or_bad


class SidewalkFollower(Node):
    def __init__(self):
        super().__init__('sidewalk_follower')

        # subscribers
        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )

        # publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.bridge = CvBridge()

        # Load FastSCNN model once
        weights_path = '/home/clement_workspace/src/pkg_FastSCNN/weights/fast_scnn_citys.pth'
        self.model = get_fast_scnn('citys', pretrained=False)
        state = torch.load(weights_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.model.eval()

        # PID memory
        self.error_list = [0]

        self.forward_speed = 0.15

        self.get_logger().info("Sidewalk follower ready.")

    def image_callback(self, msg):
        # convert ROS → CV2 image
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except:
            self.get_logger().error("Failed to convert image.")
            return

        # ----- 1. SEGMENTATION -----
        mask = segmentation_inference(frame, self.model, 'cpu')

        # grayscale needed for good_or_bad()
        mask_gray = mask.copy()
        if len(mask_gray.shape) == 3:
            mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)



        if len(mask.shape) == 2:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_bgr = mask

        edges = get_edges(mask_bgr, True)
        slope, intercept = edges[2], edges[1]

        # ----- 3. PID CONTROL -----
        turn, self.error_list = PID_sidewalk(
            -slope, intercept, self.error_list
        )

        # ----- 4. SEND CMD_VEL -----
        twist = Twist()
        twist.linear.x = self.forward_speed
        twist.angular.z = float(turn)
        self.cmd_pub.publish(twist)

        self.get_logger().info(
            f"cmd_vel: linear={twist.linear.x:.2f}, angular={twist.angular.z:.3f}"
        )

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)
        self.get_logger().warn("STOP message sent to /cmd_vel")

def main(args=None):
    rclpy.init(args=args)
    node = SidewalkFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt received! Stopping robot...")
    finally:
        # ALWAYS STOP WHEELS BEFORE SHUTDOWN
        node.stop_robot()

        node.destroy_node()
        rclpy.shutdown()



if __name__ == "__main__":
    main()
