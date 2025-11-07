import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import os
from ament_index_python.packages import get_package_share_directory

# Import your model definition and the processing utility
from .fast_scnn import get_fast_scnn
from .utils.image_processor import segmentation_inference

class FastSCNNNode(Node):
    def __init__(self):
        super().__init__('fast_scnn_node')
        
        # --- Parameters ---
        self.declare_parameter('weights_name', 'fast_scnn_cityscapes.pth')
        self.declare_parameter('device', 'cpu') # Use 'cuda' for GPU
        
        # --- ROS2 Comms ---
        self.subscription = self.create_subscription(Image, 'image_raw', self.image_callback, 10)
        self.publisher = self.create_publisher(Image, 'segmentation_mask', 10)
        self.bridge = CvBridge()
        
        # --- Model Loading ---
        self.device = torch.device(self.get_parameter('device').get_parameter_value().string_value)
        weights_name = self.get_parameter('weights_name').get_parameter_value().string_value
        
        # Find the weights file installed in the package's shared directory
        package_share_dir = get_package_share_directory('pkg_FastSCNN')
        weights_path = os.path.join(package_share_dir, 'weights', weights_name)
        
        if not os.path.exists(weights_path):
            self.get_logger().fatal(f"Weights file not found at: {weights_path}")
            rclpy.shutdown()
            return
            
        self.model = get_fast_scnn('citys', pretrained=False)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.get_logger().info(f'Fast-SCNN model loaded successfully on device: {self.device}')

    def image_callback(self, msg):
        self.get_logger().info('Received image', once=True)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
            
        # Get the segmentation mask from our utility function
        mask = segmentation_inference(cv_image, self.model, self.device)
        
        # Publish the resulting mask as a ROS Image message
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, 'mono8')
            mask_msg.header.stamp = self.get_clock().now().to_msg()
            mask_msg.header.frame_id = msg.header.frame_id
            self.publisher.publish(mask_msg)
        except Exception as e:
            self.get_logger().error(f'Failed to publish mask: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FastSCNNNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()