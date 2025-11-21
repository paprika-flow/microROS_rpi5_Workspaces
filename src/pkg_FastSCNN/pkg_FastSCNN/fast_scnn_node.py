import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import os
import time

from .fast_scnn import get_fast_scnn
from .utils.image_processor import segmentation_inference


class FastSCNNNode(Node):
    def __init__(self):
        super().__init__('fast_scnn_node')

        # --------------------------
        # Parameters
        # --------------------------
        self.declare_parameter('weights_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('save_dir', 'segmentation_output')

        self.device = torch.device(
           'cpu'
        )

        self.save_dir = (
            self.get_parameter('save_dir').get_parameter_value().string_value
        )

        os.makedirs(self.save_dir, exist_ok=True)

        weights_path = (
            self.get_parameter('weights_path')
            .get_parameter_value()
            .string_value
        )

        if not os.path.isfile(weights_path):
            self.get_logger().fatal(f"❌ Weights file not found: {weights_path}")
            rclpy.shutdown()
            return

        # --------------------------
        # Load model once
        # --------------------------
        self.get_logger().info("Loading FastSCNN model...")
        self.model = get_fast_scnn('citys', pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info(f"✅ FastSCNN loaded on {self.device}")

        # --------------------------
        # ROS interfaces
        # --------------------------
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_callback, 10
        )

        self.get_logger().info("FastSCNN Node Ready")

    # ============================================================
    # Image Callback — RUNS FAST SCNN & SAVES OUTPUT
    # ============================================================
    def image_callback(self, msg):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed CV bridge conversion: {e}")
            return

        # Run segmentation
        mask = segmentation_inference(cv_img, self.model, self.device)

        # --------------------------
        # Save mask to folder
        # --------------------------
        timestamp = int(time.time() * 1000)
        save_path = os.path.join(self.save_dir, f"mask_{timestamp}.png")

        import cv2
        cv2.imwrite(save_path, mask)
        self.get_logger().info(f"Saved segmentation: {save_path}")


def main(args=None):
    rclpy.init(args=args)
    node = FastSCNNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
