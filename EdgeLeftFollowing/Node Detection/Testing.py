import cv2 as cv
import os
import time
import numpy as np
import matplotlib.pyplot as plt

folder_path_1 = "Node"
folder_path_2 = "NotNode"

def safe_save_and_show(x, y_std, y_mean, title, filename):
    plt.figure(figsize=(12, 4))
    plt.scatter(range(1, x + 1), y_std, color='blue', label='std')
    plt.scatter(range(1, x + 1), y_mean, color='red', label='mean')
    plt.title(title)
    plt.xlabel("Window index")
    plt.ylabel(title)
    plt.grid(False)
    plt.xticks(np.arange(0, x + 1, 1))
    plt.xlim(0, x + 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


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
            pixel = img[j, i]       # pixel is [B, G, R]
            b_, g_, r_ = pixel
            # check for non-black pixel (assuming black background)
            if (j != height - stopped) and (r_ != 0 or g_ != 0 or b_ != 0):
                if not np.array_equal(previous, pixel):
                    line_points.append([j, i])
                
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

from scipy.stats import ttest_ind

def calculating_parameters_dataset(folder_path):
    # ---- Parameters ----
    num_sections = 2
    window_size = 5
    frame_count = 0

    # ---- Containers ----
    distance_y_list = [[] for _ in range(num_sections)]
    distance_x_list = [[] for _ in range(num_sections)]
    largest_area_list = [[] for _ in range(num_sections)]
    edges_slope_list = []

    std_section1_la = []
    mean_section1_la = []
    std_section2_la = []
    mean_section2_la = []

    std_section1_x = []
    mean_section1_x = []
    std_section2_x = []
    mean_section2_x = []

    std_section1_y = []
    mean_section1_y = []
    std_section2_y = []
    mean_section2_y = []

    std_slope = []
    mean_slope = []

    previous_filename = "000000.jpg"

    # ---- Main loop ----
    for filename in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        # file name parsing
        try:
            cur_id = int(filename[-10:-4])
            prev_id = int(previous_filename[-10:-4])
        except:
            previous_filename = filename
            continue

        # reset sliding window on sudden jumps
        if cur_id - prev_id > 7:
            distance_y_list = [[] for _ in range(num_sections)]
            distance_x_list = [[] for _ in range(num_sections)]
            largest_area_list = [[] for _ in range(num_sections)]
            edges_slope_list = []
            frame_count = 0

        previous_filename = filename

        # load image
        img = cv.imread(file_path)
        if img is None:
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        height, width, channel = img.shape

        # === Extract for sections ===
        for i in range(1, 3):
            left_strip = int(0.15 * i * width)
            strip_mask = gray[:, :left_strip]
            dy, dx, largest_area = get_distance_mask_points(strip_mask)

            distance_y_list[i-1].append(dy)
            distance_x_list[i-1].append(dx)
            largest_area_list[i-1].append(largest_area)

        # === Edge slope ===
        edges_slope_list.append(get_edges(img, True)[0])

        frame_count += 1

        # === Sliding window statistics ===
        if len(distance_x_list[0]) >= window_size and len(distance_x_list[1]) >= window_size:
            section1_distance_x = np.mean(distance_x_list[0])
            section2_distance_x = np.mean(distance_x_list[1])
            section1_distance_y = np.mean(distance_y_list[0])
            section2_distance_y = np.mean(distance_y_list[1])
            section1_la = np.mean(largest_area_list[0])
            section2_la = np.mean(largest_area_list[1])
            slope_average = np.mean(edges_slope_list)

            # record values
            std_section1_x.append(np.std(distance_x_list[0]))
            mean_section1_x.append(section1_distance_x)
            std_section2_x.append(np.std(distance_x_list[1]))
            mean_section2_x.append(section2_distance_x)

            std_section1_y.append(np.std(distance_y_list[0]))
            mean_section1_y.append(section1_distance_y)
            std_section2_y.append(np.std(distance_y_list[1]))
            mean_section2_y.append(section2_distance_y)

            std_section1_la.append(np.std(largest_area_list[0]))
            mean_section1_la.append(section1_la)
            std_section2_la.append(np.std(largest_area_list[1]))
            mean_section2_la.append(section2_la)

            std_slope.append(np.std(edges_slope_list))
            mean_slope.append(slope_average)

            # slide window
            distance_x_list[0].pop(0)
            distance_x_list[1].pop(0)
            distance_y_list[0].pop(0)
            distance_y_list[1].pop(0)
            largest_area_list[0].pop(0)
            largest_area_list[1].pop(0)
            if len(edges_slope_list) > window_size:
                edges_slope_list.pop(0)

    # return ALL extracted lists
    return {
        "mean_section1_y": mean_section1_y,
        "mean_section2_y": mean_section2_y,
        "mean_section1_la": mean_section1_la,
        "mean_section2_la": mean_section2_la,
        "mean_slope": mean_slope,

        "std_section1_x": std_section1_x,
        "std_section2_x": std_section2_x,
        "std_section1_y": std_section1_y,
        "std_section2_y": std_section2_y,
        "std_section1_la": std_section1_la,
        "std_section2_la": std_section2_la,
        "std_slope": std_slope
    }
node_data = calculating_parameters_dataset("Node")
notnode_data = calculating_parameters_dataset("NotNode")

stats_results = {}

for key in node_data:
    arr1 = np.array(node_data[key])
    arr2 = np.array(notnode_data[key])
    size1 = len(arr1)  # number of sliding windows computed
    size2 = len(arr2)
    print(size1)
    print(size2)
    if key[:3] != "std":
        safe_save_and_show(size1, node_data["std"+key[4:]], node_data[key], "Section 1 - Largest Area",
                    f"C:\\Users\\User\\Downloads\\Node\\{key}_node.png")
        safe_save_and_show(size2, notnode_data["std"+key[4:]], notnode_data[key], "Section 1 - Largest Area",
                    f"C:\\Users\\User\\Downloads\\Node\\{key}_not_node.png")


    # Welch’s t-test (unequal variance)
    t, p = ttest_ind(arr1, arr2, equal_var=False, nan_policy="omit")

    stats_results[key] = {"t_value": float(t), "p_value": float(p)}

for key, result in stats_results.items():
    print(f"{key}:  t = {result['t_value']:.4f},  p = {result['p_value']:.6f}")



