import cv2
import numpy as np

acceptable_b = [128, 232,60]
acceptable_g = [64, 35,20]
acceptable_r= [128, 244,220]
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
    width = int(width/2)


  for i in range(width -1, width - 320, -1):
    if stopped <= 0:
      stopped = 1
    for j in range(height - stopped, 0, -1):
      pixel = img[j, i]       # pixel is [B, G, R]
      b, g, r = pixel
      if(j != height - stopped and (r not in acceptable_r or g not in acceptable_g or b not in acceptable_b)): 
        if not np.array_equal(previous, pixel):
          line_points.append([j, i ])
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
img = cv2.imread("C:\\Users\\User\\Downloads\\Fast-SCNN\\Fast-SCNN-pytorch\\test_result\\photo_20251006_215623.png")
height, width, channel = img.shape
# Define the cropping region
# Syntax: img[y1:y2, x1:x2]
left_cropped = img[240:480, 000:240]
right_cropped = img[240:480, 400:640]
width_right_cropped = 320
left_slope, left_intercept, left_slope_n, left_intercept_n, last  = get_edges(img, True)
right_slope, right_intercept, right_slope_n, right_intercept_n, last  = get_edges(img, False)
# left_slope, left_intercept = get_edges(left_cropped)
# Show or save cropped image

width = 640
height_cropped = 240
for x in range(width-1, 0, -1):
    if x < 640:
      y_left = int(left_slope*(x) + left_intercept)
      yn_left = int(left_slope_n*x + ( (left_slope*(width_right_cropped) + left_intercept)- left_slope_n*(width_right_cropped)))
      y_right = int(right_slope*(x) + right_intercept)
      yn_right = int(right_slope_n*x + ( (right_slope*(width_right_cropped) + right_intercept)- right_slope_n*(width_right_cropped)))
    else: 
      continue
    y_img_left = min(max(y_left, 0), height - 1)
    img[y_img_left, x] = [0, 0, 0]

    y_img_n_left = min(max(yn_left, 0), height - 1)
    img[y_img_n_left, x] = [0, 0, 0]

    y_img_right = min(max(y_right, 0), height - 1)
    img[y_img_right, x] = [0, 0, 0]

    y_img_n_right = min(max(yn_right, 0), height - 1)
    img[y_img_n_right, x] = [0, 0, 0]


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
