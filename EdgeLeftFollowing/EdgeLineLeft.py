import cv2
import numpy as np

acceptable_b = [128, 232]
acceptable_g = [64, 35]
acceptable_r= [128, 244]
K = np.array([
    [524.29384, 0, 543.91968],
    [0, 557.31848, 273.30891],
    [0, 0, 1]
])
dist = np.array([-0.406832, 0.080272, -0.002452, -0.062491, 0.0])



'''def get_edges(img, width_minor, width_greater):
  height, width, channel = img.shape

  stopped = 1
  line_points = []
  previous = np.array([0, 0, 0])

  for i in range(width_greater -1, width_minor, -1):
    if stopped <= 0:
      stopped = 1
    for j in range(height - stopped, 0, -1):
      pixel = img[j, i]       # pixel is [B, G, R]
      b, g, r = pixel
      if( r not in acceptable_r or g not in acceptable_g or b not in acceptable_b): 
        if not np.array_equal(previous, pixel):
          line_points.append([j, i ])
          print([j, i])
        stopped = j -20 
        break
      previous = pixel
  
  pts = np.array(line_points, dtype=np.float32).reshape(-1, 1, 2)
  undistorted = cv2.undistortPoints(pts, K, dist)
  points = undistorted.reshape(-1, 2)

  x = points[:, 0]  # horizontal
  y = points[:, 1]  # vertical
 


  # Fit line y = m*x + b
  A = np.vstack([x, np.ones_like(x)]).T
  m, b = np.linalg.lstsq(A, y, rcond=None)[0]


  print(f"Slope (normalized): {m:.6f}, Intercept: {b:.6f}")

  N = len(line_points)
  if N > 1:
    points = np.array(line_points)

    y = points[:,0]
    x = points[:,1]
    # Fit line: y = m*x + b
    A = np.vstack([x, np.ones(len(line_points))]).T
    ml, bl = np.linalg.lstsq(A, y, rcond=None)[0]

    print(f"Slope: {ml}, Intercept: {bl}")

  return ml, bl'''

def get_edges(img, left):
  height, width, channel = img.shape

  stopped = 1
  line_points = []
  previous = np.array([0, 0, 0])
  if(left):
    start = 0
    end = width - 1
  else:
    start = width - 1
    end = 0


  for i in range(start, end, 1):
    if len(line_points) > 150:
      # Take the last 5 points
      points = np.array(line_points[-20:], dtype=np.float32)

      y = points[:, 0]
      x = np.array([1, 2, 3, 4, 5,6,7,8,9,10, 11, 12, 13, 14, 15,16,17,18,19,20], dtype=np.float32)

      # Fit line y = m*x + b
      A = np.vstack([x, np.ones(len(x))]).T
      m, b = np.linalg.lstsq(A, y, rcond=None)[0]
      print(m)

      # Check if slope is nearly flat
      if -0.08 < m < 0.08:
        print(f"at ({i}, {j})")
        break
    if len(line_points) > 480:
      break
    if stopped <= 0:
      stopped = 1
    for j in range(height - stopped, 0, -1):
      pixel = img[j, i]       # pixel is [B, G, R]
      b, g, r = pixel
      if(j != height - stopped and (r not in acceptable_r or g not in acceptable_g or b not in acceptable_b)): 
        if not np.array_equal(previous, pixel):
          line_points.append([j, i ])
        else:
          line_points = []
        break
      previous = pixel
  N = len(line_points)
  if N > 1:
    points = np.array(line_points)

    y = points[:,0]
    x = points[:,1]
    # Fit line: y = m*x + b
    A = np.vstack([x, np.ones(len(line_points))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    print(f"Slope: {m}, Intercept: {b}")
  pts = np.array(line_points, dtype=np.float32).reshape(-1, 1, 2)
  undistorted = cv2.undistortPoints(pts, K, dist)
  points = undistorted.reshape(-1, 2)
  if N > 1:

    y = points[:,0]
    x = points[:,1]
    # Fit line: y = m*x + b
    A = np.vstack([x, np.ones(len(line_points))]).T
    mn, bn = np.linalg.lstsq(A, y, rcond=None)[0]

    print(f"Slope(normalized): {mn}, Intercept: {bn}")
    
  
  return [m, b, mn, bn, line_points[N-1][0]]


# Load image
img = cv2.imread("/home/pi/microROS_rpi5_Workspaces/test_result/photo_20251007_030253.png")
height, width, channel = img.shape
# Define the cropping region
# Syntax: img[y1:y2, x1:x2]
left_cropped = img[240:480, 000:240]
right_cropped = img[240:480, 400:640]
width_right_cropped = 320
left_slope, left_intercept, left_slope_n, left_intercept_n, last  = get_edges(img, True)
#right_slope, right_intercept, right_slope_n, right_intercept_n, last  = get_edges(img, False)
# left_slope, left_intercept = get_edges(left_cropped)
# Show or save cropped image

width = 640
height_cropped = 240
for x in range(width-1, 0, -1):
    if x < 640:
      y_left = int(left_slope*(x) + left_intercept)
      yn_left = int(left_slope_n*x + ( (left_slope*(width_right_cropped) + left_intercept)- left_slope_n*(width_right_cropped)))
      #y_right = int(right_slope*(x) + right_intercept)
      #yn_right = int(right_slope_n*x + ( (right_slope*(width_right_cropped) + right_intercept)- right_slope_n*(width_right_cropped)))
    else: 
      continue
    y_img_left = min(max(y_left, 0), height - 1)
    img[y_img_left, x] = [0, 0, 0]

    y_img_n_left = min(max(yn_left, 0), height - 1)
    img[y_img_n_left, x] = [0, 0, 0]

    '''y_img_right = min(max(y_right, 0), height - 1)
    img[y_img_right, x] = [0, 0, 0]

    y_img_n_right = min(max(yn_right, 0), height - 1)
    img[y_img_n_right, x] = [0, 0, 0]'''


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
