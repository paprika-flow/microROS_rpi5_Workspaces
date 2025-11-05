#!/usr/bin/env python3
# encoding: utf-8

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2 as cv
import time
import math
import os
from std_msgs.msg import Int32

class ServoControl(Node):
    def __init__(self):
        super().__init__('servo_control')
        self.s1_pub = self.create_publisher(Int32, '/servo_s1', 10)
        self.s2_pub = self.create_publisher(Int32, '/servo_s2', 10)

    def set_servo(self, s1_angle=None, s2_angle=None):
        if s1_angle is not None:
            msg = Int32()
            msg.data = s1_angle
            self.s1_pub.publish(msg)
            self.get_logger().info(f"Sent /servo_s1 = {s1_angle}")

        if s2_angle is not None:
            msg = Int32()
            msg.data = s2_angle
            self.s2_pub.publish(msg)
            self.get_logger().info(f"Sent /servo_s2 = {s2_angle}")


class Yahboom_Forward(Node):
    """A ROS 2 Node to control Yahboom robot and handle camera capture."""

    def __init__(self):
        super().__init__('yahboom_forward_ctrl')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # speed
        self.forward_speed = 0.1  # m/s
        timer_period = 0.1

        self.get_logger().info(f'Node started. Moving forward at {self.forward_speed} m/s.')

    def publish_forward_command(self):
        twist = Twist()
        twist.linear.x = self.forward_speed
        self.pub.publish(twist)
        self.get_logger().info('Moving forward')

    def stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        self.pub.publish(twist)
        self.get_logger().info('Stopped.')

    def backwards(self):
        twist = Twist()
        twist.linear.x = -self.forward_speed
        self.pub.publish(twist)
        self.get_logger().info('Moving backward.')

    def turn(self, degrees):
        twist = Twist()
        radians = math.radians(degrees)
        twist.angular.z = self.forward_speed if radians > 0 else -self.forward_speed
        duration = abs(radians) / abs(self.forward_speed)

        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < duration:
            self.pub.publish(twist)
            time.sleep(0.1)

        twist.angular.z = 0.0
        self.pub.publish(twist)
        self.get_logger().info(f'Turned {degrees} degrees.')

    def turn_right(self):
        twist = Twist()
        twist.angular.z = self.forward_speed * -10.0
        twist.linear.x = 0.0
        self.pub.publish(twist)
    def turn_left(self):
        twist = Twist()
        twist.angular.z = self.forward_speed * 10
        twist.linear.x = 0.0
        self.pub.publish(twist)
    def set_angle(self, radians, speed):
        twist = Twist()
        twist.angular.z = radians
        twist.linear.x = speed
        self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    yahboom_servo = ServoControl()
    yahboom_node = Yahboom_Forward()
    action = 0
    turn_x = 10
    turn_y = -5

    # Create directory for pictures
    os.makedirs("photos", exist_ok=True)

    # Open camera
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        yahboom_node.get_logger().error("Could not open video stream.")
        return

    yahboom_node.get_logger().info("Camera opened. Press 'p' to take a picture.")

    try:
        while rclpy.ok():
            # Capture a frame
            ret, frame = camera.read()
            if not ret:
                yahboom_node.get_logger().error("Failed to capture frame.")
                break

            # Display the frame
            cv.imshow('Robot Camera Feed', frame)

            # Wait for key input
            key = cv.waitKey(10) & 0xFF

            if key == ord('w'):
                yahboom_node.publish_forward_command()
                action = 1
            elif key == ord('s'):
                yahboom_node.backwards()
                action = -1
            elif key == ord('a'):
                yahboom_node.turn_left()
                action = 0
            elif key == ord('d'):
                yahboom_node.turn_right()
                action = 0
            elif key == ord('m'):
                yahboom_node.stop()
                action = 0
            elif key == ord('p'):
                # Take picture
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"photos/photo_{timestamp}.jpg"
                cv.imwrite(filename, frame)
                yahboom_node.get_logger().info(f"📸 Picture saved as {filename}")
            elif key == ord('q'):
                yahboom_node.stop()
                yahboom_node.get_logger().info("Exiting program.")
                break

            elif key == 83:
                turn_x += 3
                yahboom_servo.set_servo(turn_x, None)
            elif key == 81:
                turn_x -= 3
                yahboom_servo.set_servo(turn_x, None)
            elif key == 82:
                turn_y += 3
                yahboom_servo.set_servo(None, turn_y)
            elif key == 84:
                turn_y -= 3
                yahboom_servo.set_servo(None, turn_y)



            # Keep ROS node alive
            rclpy.spin_once(yahboom_node, timeout_sec=0.01)

    except KeyboardInterrupt:
        yahboom_node.stop()
        yahboom_node.get_logger().info("Keyboard interrupt received.")

    finally:
        camera.release()
        cv.destroyAllWindows()
        yahboom_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
