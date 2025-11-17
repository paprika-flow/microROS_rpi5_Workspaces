import cv2 as cv      # For image processing
import numpy as np    # For numerical operations

def get_edges_top(gray, left):
    height, width = gray.shape
    m = 0.0
    b = 0.0
    stopped = 1
    line_points = []
    previous = 0

    # scanning direction
    if left:
        start = 0
        end = width
        step = 1
    else:
        start = width - 1
        end = -1
        step = -1

    for i in range(start, end, step):

        # fit local slope when enough points
        if len(line_points) > 150:
            y = np.array([p[0] for p in line_points[-20:]], dtype=np.float32)
            x = np.arange(1, len(y) + 1, dtype=np.float32)
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        if len(line_points) > 480:
            break

        if stopped <= 0:
            stopped = 1

        for j in range(stopped, height):
            pixel = gray[j, i]  # single channel
            # non-black pixel
            if pixel != 0:
                if pixel != previous:
                    line_points.append([j, i])
                break
            previous = pixel

    N = len(line_points)
    if N > 1:
        pts = np.array(line_points)
        y = pts[:, 0]
        x = pts[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return [m, b]







def get_distance_mask_points(mask):
    """Input: single-channel mask (0/255 or 0/1). Returns: dy, dx, largest_area.
       dy = vertical abs difference (rows), dx = horizontal abs difference (cols).
    """
    # ensure binary single-channel
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    else:
        # normalize 255->1
        mask = (mask > 0).astype(np.uint8)

    black_mask = 1 - mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_area = max([cv.contourArea(c) for c in contours], default=0)

    white_coords = np.column_stack(np.where(mask > 0))
    black_coords = np.column_stack(np.where(black_mask > 0))

    white_mean = np.mean(white_coords, axis=0) if len(white_coords) > 0 else np.array([0.0, 0.0])
    black_mean = np.mean(black_coords, axis=0) if len(black_coords) > 0 else np.array([0.0, 0.0])

    dy = abs(white_mean[0] - black_mean[0])  # rows
    dx = abs(white_mean[1] - black_mean[1])  # cols

    return dy, dx, largest_area





def extract_features_from_frames(grays):
    num_sections = 2

  
    distance_y_list = [[] for _ in range(num_sections)]
    distance_x_list = [[] for _ in range(num_sections)]
    largest_area_list = [[] for _ in range(num_sections)]
    edges_slope_list = []
  
    for gray in grays:
        height, width = gray.shape
        for i in range(1, 3):
            left_strip = int(0.15 * i * width)
            strip_mask = gray[:, :left_strip]
            dy, dx, largest_area = get_distance_mask_points(strip_mask)

            distance_y_list[i-1].append(dy)
            distance_x_list[i-1].append(dx)
            largest_area_list[i-1].append(largest_area)


        edges_slope_list.append(get_edges_top(cv.cvtColor(gray, cv.COLOR_GRAY2BGR), True)[0])

  
    features = np.array([[
        np.mean(distance_y_list[0]),
        np.mean(distance_y_list[1]),
        np.mean(largest_area_list[0]),
        np.mean(largest_area_list[1]),
        np.mean(edges_slope_list),

        np.std(distance_x_list[0]),
        np.std(distance_x_list[1]),
        np.std(distance_y_list[0]),
        np.std(distance_y_list[1]),
        np.std(largest_area_list[0]),
        np.std(largest_area_list[1]),
        np.std(edges_slope_list)
    ]])

    return features

