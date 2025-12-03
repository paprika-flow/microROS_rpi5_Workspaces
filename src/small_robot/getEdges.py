import cv2
import numpy as np

acceptable_b = [255]
acceptable_g = [255]
acceptable_r= [255]
K = np.array([
    [524.29384, 0, 543.91968],
    [0, 557.31848, 273.30891],
    [0, 0, 1]
])
dist = np.array([-0.406832, 0.080272, -0.002452, -0.062491, 0.0])


def get_edges(img, left):
  height, width, channel = img.shape

  stopped = 1
  line_points = []
  previous = np.array([0, 0, 0])
  if(left):
    start = 0
    end = width - 1
    step = 1
  else:
    start = width - 1
    end = 0
    step = -1

  for i in range(start, end, step):
    if len(line_points) > 150:
      # Take the last 5 points
      points = np.array(line_points[-20:], dtype=np.float32)

      y = points[:, 0]
      x = np.array([1, 2, 3, 4, 5,6,7,8,9,10, 11, 12, 13, 14, 15,16,17,18,19,20], dtype=np.float32)

      # Fit line y = m*x + b
      A = np.vstack([x, np.ones(len(x))]).T
      m, b = np.linalg.lstsq(A, y, rcond=None)[0]
      
      # Check if slope is nearly flat
      if -0.08 < m < 0.08:
        #  print(f"at ({i}, {j})")
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
        #else:
          #line_points = []
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

    
  pts = np.array(line_points, dtype=np.float32).reshape(-1, 1, 2)
  if pts.shape[0] == 0:
      print("No Line points found")
      return[0,0,0,0,0]
  undistorted = cv2.undistortPoints(pts, K, dist)
  points = undistorted.reshape(-1, 2)
  if N > 1:

    y = points[:,0]
    x = points[:,1]
    # Fit line: y = m*x + b
    A = np.vstack([x, np.ones(len(line_points))]).T
    mn, bn = np.linalg.lstsq(A, y, rcond=None)[0]
  if not left:
    b += m * 640
    
    
  
  return [m * step, b, mn * step, bn, line_points[N-1][0]]

def show_edges(img, left_slope, left_intercept, left_slope_n, width = 640, height = 480):
    height_cropped = int(height/2)
    width_cropped = int(width/2)
    for x in range(width-1, 0, -1):
        if x < 640:
            y_left = int(left_slope*(x) + left_intercept)
            yn_left = int(left_slope_n*x + ( (left_slope*(width_cropped) + left_intercept)- left_slope_n*(width_cropped)))
        else:
            continue
        y_img_left = min(max(y_left, 0), height - 1)
        img[y_img_left, x] = [0, 0, 0]

        y_img_n_left = min(max(yn_left, 0), height - 1)
        img[y_img_n_left, x] = [0, 0, 0]
    return img
