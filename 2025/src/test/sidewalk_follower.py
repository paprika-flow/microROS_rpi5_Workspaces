#!/usr/bin/env python3
# encoding: utf-8

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import time
import threading

# Import model architecture
from models.fast_scnn import get_fast_scnn

# ==========================================
#               CONFIGURATION
# ==========================================
class Config:
    # Hardware
    CAMERA_INDEX = 0
    FORWARD_SPEED = 0.15
    
    # Logic
    SWITCH_INTERVAL = 45.0  # Seconds between switching sides

    # Model
    WEIGHTS_PATH = "florida_sidewalk.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SIDEWALK_CLASS = 1

    # Edge Detection Constants
    ACCEPTABLE_B = [255]
    ACCEPTABLE_G = [255]
    ACCEPTABLE_R = [255]

# ==========================================
#           YOUR MATH LOGIC
# ==========================================

def get_edges(img, left):
    height, width, channel = img.shape
    stopped = 1
    line_points = []
    previous = np.array([0, 0, 0])
    
    # Determine scan direction based on 'left' flag
    if(left):
        start, end, step = 0, width - 1, 1
    else:
        start, end, step = width - 1, 0, -1

    for i in range(start, end, step):
        if len(line_points) > 150:
            points = np.array(line_points[-20:], dtype=np.float32)
            y = points[:, 0]
            x = np.array([1, 2, 3, 4, 5,6,7,8,9,10, 11, 12, 13, 14, 15,16,17,18,19,20], dtype=np.float32)
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]
            if -0.08 < m < 0.08: break
        
        if len(line_points) > 480: break
        if stopped <= 0: stopped = 1
            
        for j in range(height - stopped, 0, -1):
            pixel = img[j, i]
            b_val, g_val, r_val = pixel
            if(j != height - stopped and (r_val not in Config.ACCEPTABLE_R or g_val not in Config.ACCEPTABLE_G or b_val not in Config.ACCEPTABLE_B)): 
                if not np.array_equal(previous, pixel):
                    line_points.append([j, i])
                else:
                    line_points = []
                break
            previous = pixel
            
    N = len(line_points)
    if N > 1:
        points = np.array(line_points)
        y = points[:,0]
        x = points[:,1]
        A = np.vstack([x, np.ones(len(line_points))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return [m, b, line_points[N-1][0]]
    
    return [0, 0, 0]

# --- PID HELPERS ---

def function_modifier(function, limit):
    function = function * 0.4
    if function > limit: return limit
    elif function < - limit: return -limit
    else: return function

def I_sum_list(lst): return sum(lst)

def D_sum_list(lst, t):
    if t > 0: return lst[t] - lst[t-1]
    return 0

def PID_sidewalk(slope, intercept, error_list, left):
    Kp = 0.5
    Ki = 0.01
    Kd = 0.8
    
    error_intercept = 0
    error_slope = 0
    factor = 100

    if intercept > 320:
        error_intercept = (intercept - 310) / factor
    elif intercept < 290:
        error_intercept = (intercept - 310) / factor
    else:
        # print(f"Intercept * slope = {intercept*slope: .3f}")
        intercept_slope = intercept*slope
        if intercept_slope < 60:
            error_slope =   80 - intercept_slope
        elif slope > 80:
            error_slope = 60 - intercept_slope
           
    error = error_intercept + error_slope / 200
    
    if len(error_list) > 100: error_list.pop(0) 
    if error_list[-1] != 0 or error != 0: error_list.append(error)
        
    t = len(error_list) - 1 
    function = -1 * (Kp * error_list[t] + Ki * I_sum_list(error_list) + Kd * D_sum_list(error_list, t))
    angularz = function_modifier(function, 0.15)
    
    # Invert logic for right side following
    if not left:
        angularz *= -1
    return angularz, error_list

# ==========================================
#               ROS NODE
# ==========================================

class SidewalkHybridController(Node):
    def __init__(self):
        super().__init__('sidewalk_hybrid_controller')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info("Initializing Controller...")

        # 1. Load Model
        self.model = get_fast_scnn('citys', pretrained=False, root='./weights', map_cpu=True)
        self._perform_surgery()
        self._load_weights()
        self.model.to(Config.DEVICE)
        self.model.eval()
        
        # 2. Setup Switching State
        self.left = False # Start following Right side
        self.last_switch_time = time.time()
        
        # Warmup
        if Config.DEVICE == 'cuda':
            dummy = torch.randn(1, 3, 256, 512).to(Config.DEVICE)
            self.model(dummy)

        # 3. Setup Camera
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.latest_frame = None
        self._reader_running = True
        self._reader_thread = threading.Thread(target=self._capture_reader, daemon=True)
        self._reader_thread.start()

        # 4. PID State
        self.error_list = [0]
        
        # Start Loop
        self.create_timer(0.5, self.control_loop)
        
    def _capture_reader(self):
        while self._reader_running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
            else:
                time.sleep(0.01)
    
    def _perform_surgery(self):
        classifier_block = self.model.classifier.conv
        if isinstance(classifier_block, nn.Sequential):
            for i, layer in enumerate(classifier_block):
                if isinstance(layer, nn.Conv2d):
                    classifier_block[i] = nn.Conv2d(layer.in_channels, 2, kernel_size=1)
                    break
        else:
            self.model.classifier.conv = nn.Conv2d(classifier_block.in_channels, 2, kernel_size=1)

    def _load_weights(self):
        if not os.path.exists(Config.WEIGHTS_PATH):
            self.get_logger().error(f"Weights file not found: {Config.WEIGHTS_PATH}")
            sys.exit(1)
        state = torch.load(Config.WEIGHTS_PATH, map_location=Config.DEVICE)
        self.model.load_state_dict(state)

    def preprocess(self, img_bgr):
        img_resized = cv2.resize(img_bgr, (512, 256))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std
        img_chw = img_norm.transpose((2, 0, 1))
        return torch.from_numpy(img_chw).unsqueeze(0).float()

    def control_loop(self):
        frame = self.latest_frame
        if frame is None:
            return  # no frame yet
            
        # --- SWITCHING LOGIC ---
        current_time = time.time()
        if current_time - self.last_switch_time > Config.SWITCH_INTERVAL:
            self.left = not self.left # Toggle direction
            self.last_switch_time = current_time
            
            # CRITICAL: Reset PID error memory. 
            # If we don't do this, the PID will try to correct the left wall 
            # using the error from the right wall, causing a massive jerk.
            self.error_list = [0] 
            
            msg = f"SWITCHED DIRECTION! Now following: {'LEFT' if self.left else 'RIGHT'}"
            self.get_logger().info(msg)
            print(f"\n{msg}\n")
        # -----------------------

        height, width, channel = frame.shape

        # 1. INFERENCE
        input_tensor = self.preprocess(frame).to(Config.DEVICE)
        with torch.no_grad():
            output = self.model(input_tensor)[0]
            pred = torch.argmax(output, 1).squeeze(0).cpu().numpy()
            
        if Config.DEVICE == 'cuda': torch.cuda.synchronize()

        # 2. MASK GENERATION
        mask_small = (pred == Config.SIDEWALK_CLASS).astype(np.uint8) * 255
        h, w = frame.shape[:2]
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 3. GET EDGES (Your Logic)
        # We pass self.left here so the scanner knows which side to scan from
        edges = get_edges(mask_bgr, left=self.left)
        slope = edges[0]
        intercept = edges[1]
        
        # Normalize Coordinate System for PID
        # If following right, we adjust intercept to be relative to the edge, and invert slope
        if not self.left:
            intercept += slope * width
            slope = -slope

        # 4. PID CONTROL
        turn_z, self.error_list = PID_sidewalk(-slope, intercept, self.error_list, self.left)

        # 5. MOVE
        twist = Twist()
        twist.linear.x = Config.FORWARD_SPEED
        twist.angular.z = float(turn_z)
        self.pub.publish(twist)

        # 6. DEBUG DISPLAY
        if abs(slope) > 0.01:
            try:
                y1, y2 = h, int(h/2)
                # Reverse the intercept/slope math for visualization if needed
                viz_intercept = edges[1]
                viz_slope = edges[0]
                
                x1 = int((y1 - viz_intercept) / viz_slope)
                x2 = int((y2 - viz_intercept) / viz_slope)
                
                # Draw Red line for Right, Green line for Left
                color = (0, 255, 0) if self.left else (0, 0, 255)
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)
            except:
                pass

        # Add text to debug indicating mode
        mode_text = f"Mode: {'LEFT' if self.left else 'RIGHT'} | Time to switch: {int(Config.SWITCH_INTERVAL - (current_time - self.last_switch_time))}s"
        cv2.putText(debug_frame := np.hstack((frame, mask_bgr)), mode_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        print(f"[INFO] Mode:{'L' if self.left else 'R'} | Slope:{slope:.3f} | Turn:{turn_z:.3f}")

        cv2.imshow("Sidewalk Follower", debug_frame)
        
        if cv2.waitKey(1) == ord('q'):
            self.stop()
            sys.exit(0)

    def stop(self):
        self._reader_running = False
        if hasattr(self, "_reader_thread"):
            self._reader_thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()
        self.pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = SidewalkHybridController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()