
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

def edges(camera, model, left):
    ret, frame = camera.read()
    if not ret:
        yahboom_node.get_logger().error("Failed to capture frame.")
        return 0.0
    #Segment the frame
    mask = demo_folder(frame, model)
    mask_np = np.array(mask)
    mask_bgr = cv.cvtColor(mask_np, cv.COLOR_RGB2BGR)
    mask_gray = cv.cvtColor(mask_bgr, cv.COLOR_BGR2GRAY)
    return get_edges(mask_bgr, True)

def drive_until(ratio, greater,camera, model,error, turn, left, yahboom_node):
    while True:
        ret, frame = camera.read()
        if not ret:
            yahboom_node.get_logger().error("Failed to capture frame.")
            break
        #Segment the frame
        mask = demo_folder(frame, model)
        mask_np = np.array(mask)
        mask_bgr = cv.cvtColor(mask_np, cv.COLOR_RGB2BGR)
        mask_gray = cv.cvtColor(mask_bgr, cv.COLOR_BGR2GRAY)
        edges = get_edges(mask_bgr, True)
        print(edges[0])
        if not turn:
            turn, error = PID_sidewalk(-edges[2], edges[1], error, left)
            yahboom_node.set_angle(float(turn), yahboom_node.forward_speed)
        # Stop when fewer than 90% white
        if edges[0] > ratio and greater:
            break
        elif edges[0] < ratio and not greater:
            break
        rclpy.spin_once(yahboom_node, timeout_sec=0.01)


def main(args=None):
    os.makedirs("combined", exist_ok=True)
    rclpy.init(args=args)
    yahboom_servo = ServoControl()
    yahboom_node = Yahboom_Forward()
    model = load_model()
    # Load trained model
    
    # === LOAD MODEL & SCALER ===
    bundle = pickle.load(open("model_and_scaler.pkl", "rb"))
    model_node = bundle["model"]
    scaler = bundle["scaler"]


    # Initialize servo
    turn_x, turn_y = 15, -70

    # Open camera
    camera = cv.VideoCapture(0)
    if not camera.isOpened():
        yahboom_node.get_logger().error("Could not open video stream.")
        return

    
    left = True
    interval = 0.4
    turn = 0.0
    yahboom_node.stop()
    bad_image = False
    try:
        while True:
            yahboom_servo.set_servo(turn_x,turn_y)
            speed_factor = 0.75
            error = [0]
            turn_list = []
            grays = []
            last_run = time.time()
            begi = time.time()
            start = time.time()
            while rclpy.ok():
                ret, frame = camera.read()
                if not ret:
                    yahboom_node.get_logger().error("Failed to capture frame.")
                    break
                current_time = time.time()
                
                if current_time - start >= interval:

                    # Segment the frame
                    mask = demo_folder(frame, model)
                    mask_np = np.array(mask)
                    mask_bgr = cv.cvtColor(mask_np, cv.COLOR_RGB2BGR)
         
                    mask_gray = cv.cvtColor(mask_bgr, cv.COLOR_BGR2GRAY)
                    if(good_or_bad(mask_gray) == 1):
                        yahboom_node.set_angle(float(turn)/2, yahboom_node.forward_speed*speed_factor)
                        print("bad image", len(grays))    
                        bad_image = True
                    else:
                        bad_image = False
                    if not bad_image:
                        # Get edges and compute PID turn
                        edges = get_edges(mask_bgr, left)
                        turn, error = PID_sidewalk(-edges[2], edges[1], error, left)
                        yahboom_node.set_angle(float(turn), yahboom_node.forward_speed*speed_factor)
                        #print("turn:",turn)
                        # Display edges and original frame side-by-side
                        mask_edged = show_edges(mask_bgr, edges[0], edges[1], edges[2])
                        timestamp = time.strftime("%Y%m%d_%H%M%S")  # e.g., 20251027_140512
                        filename = f"combined/combined_{timestamp}.jpg"
                        cv.imwrite(f"cmp/combined_{timestamp}.jpg", mask_edged)

                    #combined = np.hstack((frame, mask_bgr))
                    # ---- Checking If it's a Node -----
                        
                        if current_time - begi > 10:
                            speed_factor = 1
                            grays.append(mask_gray)
                            #print(len(grays), current_time - last_run)
                        if len(grays) > 5:
                            grays.pop(0)
                    
                    '''if len(grays) >=5 and current_time - last_run > 2:
                        yahboom_node.set_angle(0.0, 0.15)
                        features = extract_features_from_frames(grays, left)        
                        #y_pred = model_node.predict(X_new)[0]
                            
                        x_scaled = scaler.transform(features)
                        pred = model_node.predict(x_scaled)[0]
                        print(turn_list)
                        sum_turn = np.sum(turn_list[:-2])
                        print(sum_turn) 
                        if pred == 1:

                            yahboom_node.stop()
                            print(f"Split detected!")
                            if edges[0] < -0.15:
                                grays = []
                                continue

                            if np.sum(sum_turn) < -0.5:
                                yahboom_node.set_angle(-0.4, -yahboom_node.forward_speed*2)
                                time.sleep(0.6)
                                yahboom_node.set_angle(0.4, -yahboom_node.forward_speed*1.5)
                                time.sleep(0.8)
                                grays = []
                            else:
                                break
                        else:
                            print(f"No split detected!")
                        last_run = time.time()'''
                            
                    # Showing and saving the mask
                    timestamp = time.strftime("%Y%m%d_%H%M%S")  # e.g., 20251027_140512
                    turn_list.append(turn)
                        
                    if len(turn_list) > 10:
                        turn_list.pop(0)
                    cv.imwrite(f"cmp/{timestamp}.jpg", mask_bgr)
                    cv.imwrite(f"cmp_photos/{timestamp}.jpg", frame)
                    start = time.time()
                        
                rclpy.spin_once(yahboom_node, timeout_sec=0.01)
            
            #yahboom_node.set_angle(0.0, yahboom_node.forward_speed)
            yahboom_node.stop()
            
            x = input("Forward(f), Right(r), Left(l) or stop(s): ")
            if x == "s":
                yahboom_node.stop()
                break
            elif x == "f":
                sum_last_3t = np.sum(turn_list[-4:-1]) 
                if sum_last_3t > 0.0:
                    yahboom_node.set_angle(-0.2,-yahboom_node.forward_speed)
                    time.sleep(sum_last_3t*4)
                #yahboom_node.set_angle(-sum_last_3t/15, yahboom_node.forward_speed)
                if edges(camera, model, left)[0] < -0.1:
                    drive_until(-0.1, True,camera,model, error, False, left, yahboom_node)
                
                yahboom_node.set_angle(0.0, yahboom_node.forward_speed)
                time.sleep(1)
                drive_until(-0.1, False,camera, model, error, True, left,yahboom_node)
                time.sleep(2)
                yahboom_node.stop()
            elif x == "l":
                yahboom_node.set_angle(0.0, yahboom_node.forward_speed)
                if edges(camera, model,left)[0] < -0.1:
                    yahboom_node.set_angle(0.0, yahboom_node.forward_speed)
                    drive_until(-0.1, True, camera, model, error, False, left, yahboom_node)
                yahboom_node.set_angle(0.0, yahboom_node.forward_speed)
                time.sleep(0.5)
                yahboom_node.set_angle(2.0,-yahboom_node.forward_speed*3.0)
                time.sleep(1)
                yahboom_node.set_angle(1.5,yahboom_node.forward_speed*2.0)
                time.sleep(1)
                drive_until(-0.15, False,camera, model, error, True, left, yahboom_node)
                yahboom_node.stop()
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
