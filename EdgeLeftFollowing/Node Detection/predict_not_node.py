import cv2 as cv
import os
import numpy as np
import pickle

# === LOAD MODEL & SCALER ===
bundle = pickle.load(open("model_and_scaler.pkl", "rb"))
model = bundle["model"]
scaler = bundle["scaler"]

# === SAME FUNCTIONS AS TRAINING ===
def get_edges(img, left):
    height, width, channel = img.shape
    m = 0
    b = 0
    stopped = 1
    line_points = []
    previous = np.array([0, 0, 0])

    if left:
        start = 0
        end = width - 1
        step = 1
    else:
        start = width - 1
        end = 0
        step = -1

    for i in range(start, end, step):
        if len(line_points) > 150:
            points = np.array(line_points[-20:], dtype=np.float32)
            y = points[:, 0]
            x = np.arange(1, len(y) + 1, dtype=np.float32)
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        if len(line_points) > 480:
            break

        if stopped <= 0:
            stopped = 1

        for j in range(0 + stopped, height):
            pixel = img[j, i]
            b_, g_, r_ = pixel
            if (j != height - stopped) and (r_ != 0 or g_ != 0 or b_ != 0):
                if not np.array_equal(previous, pixel):
                    line_points.append([j, i])
                else:
                    line_points = []
                break
            previous = pixel

    N = len(line_points)
    if N > 1:
        points = np.array(line_points)
        y = points[:, 0]
        x = points[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return [m, b]

def get_distance_mask_points(mask):
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8)
    else:
        mask = (mask > 0).astype(np.uint8)

    black_mask = 1 - mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_area = max([cv.contourArea(c) for c in contours], default=0)

    white_coords = np.column_stack(np.where(mask > 0))
    black_coords = np.column_stack(np.where(black_mask > 0))

    white_mean = np.mean(white_coords, axis=0) if len(white_coords) > 0 else np.array([0.0, 0.0])
    black_mean = np.mean(black_coords, axis=0) if len(black_coords) > 0 else np.array([0.0, 0.0])

    dy = abs(white_mean[0] - black_mean[0])
    dx = abs(white_mean[1] - black_mean[1])

    return dy, dx, largest_area


# === FOLDER TO SCAN ===
folder ="NotNode"

# === SLIDING WINDOW STATE ===
window_size = 5
num_sections = 2

distance_y_list = [[] for _ in range(num_sections)]
distance_x_list = [[] for _ in range(num_sections)]
largest_area_list = [[] for _ in range(num_sections)]
edges_slope_list = []

previous_filename = "000000.jpg"

print("\n=== Predictions on NotNode Folder ===")

for filename in sorted(os.listdir(folder)):
    path = os.path.join(folder, filename)
    if not os.path.isfile(path):
        continue

    try:
        cur_id = int(filename[-10:-4])
        prev_id = int(previous_filename[-10:-4])
    except:
        previous_filename = filename
        continue

    # Jump reset
    if cur_id - prev_id > 7:
        distance_y_list = [[] for _ in range(num_sections)]
        distance_x_list = [[] for _ in range(num_sections)]
        largest_area_list = [[] for _ in range(num_sections)]
        edges_slope_list = []

    previous_filename = filename

    img = cv.imread(path)
    if img is None:
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    height, width, ch = img.shape

    # === Extract section features ===
    for i in range(1, 3):
        left_strip = int(0.15 * i * width)
        strip_mask = gray[:, :left_strip]
        dy, dx, area = get_distance_mask_points(strip_mask)
        distance_y_list[i-1].append(dy)
        distance_x_list[i-1].append(dx)
        largest_area_list[i-1].append(area)

    # Edges slope
    edges_slope_list.append(get_edges(img, True)[0])

    # Need full sliding window to predict
    if len(distance_x_list[0]) < window_size:
        continue

    # === Compute same 12 features as training ===
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

    # Scale + predict
    x_scaled = scaler.transform(features)
    pred = model.predict(x_scaled)[0]

    # Only print if model says it's a NODE
    if pred == 1:
        print(f"⚠️  Model flagged as NODE → {filename}")
        print(f"{filename}")
    # Slide window
    distance_y_list[0].pop(0)
    distance_y_list[1].pop(0)
    distance_x_list[0].pop(0)
    distance_x_list[1].pop(0)
    largest_area_list[0].pop(0)
    largest_area_list[1].pop(0)
    if len(edges_slope_list) > window_size:
        edges_slope_list.pop(0)
