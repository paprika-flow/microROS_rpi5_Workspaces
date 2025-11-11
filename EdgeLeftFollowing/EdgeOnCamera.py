from getEdges import get_edges, show_edges
from TakePicture import Yahboom_Forward, ServoControl
from Segmentation import load_model
from demo2 import demo_folder
from PID import PID_sidewalk
from checking_if_good import good_or_bad
from Node_feature_extraction import extract_features_from_frames

import pickle
import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2 as cv
import numpy as np
import time


def main(args=None):
    os.makedirs("combined", exist_ok=True)
    rclpy.init(args=args)
    yahboom_servo = ServoControl()
    yahboom_node = Yahboom_Forward()
    model = load_model()
    # Load trained model
    with open('model_node.pkl', 'rb') as f:
        model_node = pickle.load(f)

    # Initialize servo
    turn_x, turn_y = 15, -70
    yahboom_servo.set_servo(turn_x, turn_y)

    # Open camera
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        yahboom_node.get_logger().error("Could not open video stream.")
        return

    # Start moving forward continuously
    forward_twist = Twist()
    forward_twist.linear.x = yahboom_node.forward_speed
    yahboom_node.pub.publish(forward_twist)
    last_run = time.time()
    interval = 0.2
    error = [0]
    turn = 0.0
    grays = []
    try:
        while rclpy.ok():
            ret, frame = camera.read()
            if not ret:
                yahboom_node.get_logger().error("Failed to capture frame.")
                break
            current_time = time.time()
            if current_time - last_run >= interval:

                # Segment the frame
                mask = demo_folder(frame, model)
                mask_np = np.array(mask)
                mask_bgr = cv.cvtColor(mask_np, cv.COLOR_RGB2BGR)
                mask_gray = mask_np.copy()
                if len(mask_gray.shape) == 3:
                    mask_gray = cv.cvtColor(mask_gray, cv.COLOR_BGR2GRAY)
                if(good_or_bad(mask_gray) == 1):
                    last_run = time.time()
                    yahboom_node.set_angle(float(turn), yahboom_node.forward_speed)
                    print("bad image")    
                    continue
                # Get edges and compute PID turn
                edges = get_edges(mask_bgr, True)
                turn, error = PID_sidewalk(-edges[2], edges[1], error)
                yahboom_node.set_angle(float(turn), yahboom_node.forward_speed)

                # Display edges and original frame side-by-side
                mask_edged = show_edges(mask_bgr, edges[0], edges[1], edges[2])
                combined = np.hstack((frame, mask_bgr))
                # ---- Checking If it's a Node -----
                
                grays.append(mask_gray)
                if len(grays) > 5:
                    grays.pop(0)

                if len(grays) == 5:
                    features = extract_features_from_frames(grays)
                    X_new = np.array(features).reshape(1, -1)
                    y_pred = model_node.predict(X_new)[0]
                    prob = model_node.predict_proba(X_new)[0]

                    if y_pred == 1:
                        print(f"?? Split detected! (confidence: {prob[1]:.2f})")
                    else:
                        print(f"? No split detected. (confidence: {prob[0]:.2f})")
                
                # Showing and saving the mask
                cv.imshow('Robot Camera Feed', mask_gray)
                timestamp = time.strftime("%Y%m%d_%H%M%S")  # e.g., 20251027_140512
                filename = f"combined/combined_{timestamp}.jpg"
                cv.imwrite(f"cmp/combined_{timestamp}.jpg", mask_gray)
                cv.imwrite(f"cmp_photos/{timestamp}.jpg", frame)
                last_run = time.time()
            # Keyboard overrides
            key = cv.waitKey(1) & 0xFF
            if key == ord('m'):  # stop
                yahboom_node.stop()
            elif key == ord('q'):
                yahboom_node.stop()
                yahboom_node.get_logger().info("Exiting program.")
                break

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
