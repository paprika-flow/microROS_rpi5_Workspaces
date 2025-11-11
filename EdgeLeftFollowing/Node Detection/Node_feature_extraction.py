import cv2 as cv      # For image processing
import numpy as np    # For numerical operations

def get_edges_top(img, left):
  height, width, channel = img.shape
  m = 0
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
      # Take the last 20 points
      points = np.array(line_points[-20:], dtype=np.float32)

      y = points[:, 0]
      x = np.array([1, 2, 3, 4, 5,6,7,8,9,10, 11, 12, 13, 14, 15,16,17,18,19,20], dtype=np.float32)

      # Fit line y = m*x + b
      A = np.vstack([x, np.ones(len(x))]).T
      m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    if len(line_points) > 480:
      break
    if stopped <= 0:
      stopped = 1
    for j in range(0 + stopped , height):
      pixel = img[j, i]       # pixel is [B, G, R]
      b, g, r = pixel
      if(j != height - stopped and (r != 0 or g != 0 or b != 0)): 
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
    # Fit line: y = m*x + b
    A = np.vstack([x, np.ones(len(line_points))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]  
    
  return [m, b]



def get_distance_mask_points(mask):
    mask = (mask > 0.5).astype(np.uint8)
    black_mask = 1 - mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_area = max([cv.contourArea(c) for c in contours], default=0)

    # === NEW FEATURE: mean position of white and black pixels ===
    white_coords = np.column_stack(np.where(mask > 0))
    black_coords = np.column_stack(np.where(black_mask > 0))

    # Mean positions (y, x)
    white_mean = np.mean(white_coords, axis=0) if len(white_coords) > 0 else np.array([0, 0])
    black_mean = np.mean(black_coords, axis=0) if len(black_coords) > 0 else np.array([0, 0])

    # Compute distance and ratio
    distance = np.linalg.norm(white_mean - black_mean)  # Euclidean distance

    return distance, largest_area




def extract_features_from_frames(grays):
    distance_list = [[], []]
    largest_area_list = [[], []]
    edges_slope_list = []

    for gray in grays:
        height, width = gray.shape
        for i in range(1, 3):
            left_strip = int(0.15 * i * width)
            distance, largest_area = get_distance_mask_points(gray[:, :left_strip])
            distance_list[i - 1].append(distance)
            largest_area_list[i - 1].append(largest_area)

        edges_slope_list.append(get_edges_top(cv.cvtColor(gray, cv.COLOR_GRAY2BGR), True)[0])

    section1_distance = np.mean(distance_list[0])
    section2_distance = np.mean(distance_list[1])
    section1_la = np.mean(largest_area_list[0])
    section2_la = np.mean(largest_area_list[1])
    slope_average = np.mean(edges_slope_list)

    return np.array([
        section1_distance, np.std(distance_list[0]),
        section2_distance, np.std(distance_list[1]),
        section1_la, np.std(largest_area_list[0]),
        section2_la, np.std(largest_area_list[1]),
        slope_average, np.std(edges_slope_list)
    ])
