import cv2
from Skeleton_lines import process_image_and_compute_skeleton
from PID import PID_sidewalk
import time

if __name__ == "__main__":
  start_time = time.time()

  img = cv2.imread("photos\\img_20260213_041057_sidewalk_mask.png", cv2.IMREAD_GRAYSCALE)
  img2 = cv2.imread("photos\\img_20260213_041600_sidewalk_mask.png", cv2.IMREAD_GRAYSCALE)
  img3 = cv2.imread("photos\\image222.png", cv2.IMREAD_GRAYSCALE)
  # Resize all images to (480, 640)
  target_size = (640, 480)  # OpenCV uses (width, height)
  error_list = [0]

  img = cv2.resize(img, target_size)
  img2 = cv2.resize(img2, target_size)
  img3 = cv2.resize(img3, target_size)

  straight_path, best_angle, area_percentage_difference = process_image_and_compute_skeleton(img)
  angular_z, error_list = PID_sidewalk(area_percentage_difference, best_angle, error_list)
  print(f"how much to turn: {angular_z:.2f}, of angle: {best_angle:.2f} and area difference: {area_percentage_difference:.2f}%")
  straight_path, best_angle, area_percentage_difference = process_image_and_compute_skeleton(img2, straight_path)
  angular_z, error_list = PID_sidewalk(area_percentage_difference, best_angle, error_list)
  print(f"how much to turn: {angular_z:.2f}, of angle: {best_angle:.2f} and area difference: {area_percentage_difference:.2f}%")
  straight_path, best_angle, area_percentage_difference = process_image_and_compute_skeleton(img3, straight_path)
  angular_z, error_list = PID_sidewalk(area_percentage_difference, best_angle, error_list)
  print(f"how much to turn: {angular_z:.2f} of angle: {best_angle:.2f} and area difference: {area_percentage_difference:.2f}%")
  print("Execution time: %.2f seconds" % (time.time() - start_time))