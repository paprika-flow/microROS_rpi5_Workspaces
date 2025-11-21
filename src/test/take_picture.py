#!/usr/bin/env python3
# encoding: utf-8

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2 as cv
import time
import os
import sys
import select
import termios
import tty

msg = """
---------------------------
  CRUISE CONTROL MODE
---------------------------
1. Press 'w' to start moving forward.
2. Press 'a' or 'd' to curve while moving.
3. Press 'q'/'z' to change max speed.
4. SPACE to Stop.

Controls:
   w      : Set Forward Speed (Cruise)
   s      : Set Backward Speed (Cruise)
 a   d    : Add Turning Adjustment (Steer)
 SPACE    : STOP
 
 q / z    : Increase / Decrease Speed
 h        : Take Picture
 CTRL-C   : Quit
---------------------------
"""

class Yahboom_Controller(Node):
    def __init__(self):
        super().__init__('yahboom_controller')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # Settings
        self.cruise_speed = 0.0      # Current forward/backward momentum
        self.set_speed = 0.1         # The target speed setting
        self.turn_speed = 0.0        # Current turning momentum
        self.turn_intensity = 1.0    # How sharp the turn is
        
        self.get_logger().info('Node Started. Ready.')

    def update_movement(self):
        twist = Twist()
        twist.linear.x = float(self.cruise_speed)
        twist.angular.z = float(self.turn_speed)
        self.pub.publish(twist)

    def adjust_max_speed(self, increase=True):
        if increase:
            self.set_speed += 0.05
        else:
            self.set_speed -= 0.05
        
        # Limits
        self.set_speed = max(0.05, min(self.set_speed, 1.0))
        print(f"⚙️  Target Speed Set to: {self.set_speed:.2f} m/s\r")

def getKey(settings):
    """Captures a single keypress from terminal directly."""
    tty.setraw(sys.stdin.fileno())
    # Wait 0.1s for input, else return None
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main(args=None):
    print("Initializing ROS...")
    rclpy.init(args=args)
    node = Yahboom_Controller()
    
    # Save terminal settings so we can restore them later
    settings = termios.tcgetattr(sys.stdin)

    os.makedirs("photos", exist_ok=True)

    # Open Camera
    # Try index 0, if that fails, try index 1 (common on Jetsons)
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        camera = cv.VideoCapture(1)
        if not camera.isOpened():
            print("❌ Error: Could not open camera (tried index 0 and 1).")
            return

    print(msg)

    try:
        while rclpy.ok():
            # 1. Read Camera
            ret, frame = camera.read()

            # 2. Read Key (Non-blocking)
            key = getKey(settings)

            # --- LOGIC IMPLEMENTATION ---
            
            # If no key is pressed, 'turn_speed' should reset to 0 
            # (so it stops turning when you release A/D), 
            # BUT 'cruise_speed' stays (so it keeps driving).
            node.turn_speed = 0.0

            if key == 'w':
                # Set Cruise Control Forward
                node.cruise_speed = node.set_speed
                print(f"Moving Forward ({node.cruise_speed:.2f})\r", end="")
                
            elif key == 's':
                # Set Cruise Control Backward
                node.cruise_speed = -node.set_speed
                print(f"Moving Backward ({node.cruise_speed:.2f})\r", end="")

            elif key == 'a':
                # Left Adjustment (Add positive rotation)
                # We use 'set_speed' for turning so it scales with your speed setting
                node.turn_speed = node.set_speed * node.turn_intensity
                print(f"Steering Left\r", end="")

            elif key == 'd':
                # Right Adjustment (Add negative rotation)
                node.turn_speed = -node.set_speed * node.turn_intensity
                print(f"Steering Right\r", end="")

            elif key == ' ':
                # Emergency Stop
                node.cruise_speed = 0.0
                node.turn_speed = 0.0
                print("🛑 STOPPED\r", end="")

            elif key == 'q':
                node.adjust_max_speed(increase=True)
            
            elif key == 'z':
                node.adjust_max_speed(increase=False)

            elif key == 'h':
                if ret:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    fn = f"photos/photo_{timestamp}.jpg"
                    cv.imwrite(fn, frame)
                    print(f"📸 Saved: {fn}      \r", end="")
                else:
                    print("⚠️  Camera frame empty.\r", end="")

            elif key == '\x03': # CTRL-C
                break

            # 3. Publish the calculated command
            node.update_movement()
            
            # 4. Loop rate sleep
            time.sleep(0.05)

    except Exception as e:
        print(f"\n❌ Error: {e}")

    finally:
        # Restore Terminal (CRITICAL)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        
        # Stop Robot
        stop_twist = Twist()
        node.pub.publish(stop_twist)
        
        camera.release()
        node.destroy_node()
        rclpy.shutdown()
        print("\nExited Cleanly.")

if __name__ == '__main__':
    main()