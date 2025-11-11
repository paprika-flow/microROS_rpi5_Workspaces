import cv2 as cv
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# === EDGE DETECTION FUNCTION ===
def get_edges(img, left):
    height, width, channel = img.shape
    m = 0
    stopped = 1
    line_points = []
    previous = np.array([0, 0, 0])
    start, end, step = (0, width - 1, 1) if left else (width - 1, 0, -1)

    for i in range(start, end, step):
        if len(line_points) > 150:
            # Fit line using last 20 points
            points = np.array(line_points[-20:], dtype=np.float32)
            y = points[:, 0]
            x = np.arange(1, 21, dtype=np.float32)
            A = np.vstack([x, np.ones(len(x))]).T
            m, b = np.linalg.lstsq(A, y, rcond=None)[0]

        if len(line_points) > 480:
            break
        if stopped <= 0:
            stopped = 1
        for j in range(0 + stopped, height):
            b, g, r = img[j, i]
            if j != height - stopped and (r != 0 or g != 0 or b != 0):
                if not np.array_equal(previous, img[j, i]):
                    line_points.append([j, i])
                else:
                    line_points = []
                break
            previous = img[j, i]
    if len(line_points) > 1:
        points = np.array(line_points)
        y = points[:, 0]
        x = points[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return [m, b]


# === DISTANCE CALCULATION FUNCTION ===
def get_distance_mask_points(mask):
    mask = (mask > 0.5).astype(np.uint8)
    black_mask = 1 - mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_area = max([cv.contourArea(c) for c in contours], default=0)

    white_coords = np.column_stack(np.where(mask > 0))
    black_coords = np.column_stack(np.where(black_mask > 0))

    white_mean = np.mean(white_coords, axis=0) if len(white_coords) > 0 else np.array([0, 0])
    black_mean = np.mean(black_coords, axis=0) if len(black_coords) > 0 else np.array([0, 0])

    distance = np.linalg.norm(white_mean - black_mean)
    return distance, largest_area


# === MAIN TRAINING PIPELINE ===
size = 3
distance_list = []
largest_area_list = []
edges_slope_list = []
j = 0
previous_filename = "000000.jpg"

input_dir = r"C:\Users\User\Downloads\Node"
categories = ['NotNode', 'Node']

data = []
labels = []

for category_idx, category in enumerate(categories):
    category_path = os.path.join(input_dir, category)
    for filename in sorted(os.listdir(category_path)):
        file_path = os.path.join(category_path, filename)
        if not os.path.isfile(file_path):
            continue

        # Reset when frame jump is large
        if int(filename[-10:-4]) - int(previous_filename[-10:-4]) > 7:
            edges_slope_list = []
            distance_list = [[] for _ in range(size)]
            largest_area_list = [[] for _ in range(size)]
            j = 0

        previous_filename = filename
        img = cv.imread(file_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        height, width, _ = img.shape

        # Extract two strips (left and near-left)
        for i in range(1, 3):
            left_strip = int(0.15 * i * width)
            distance, largest_area = get_distance_mask_points(gray[:, :left_strip])
            distance_list[i - 1].append(distance)
            largest_area_list[i - 1].append(largest_area)

        edges_slope_list.append(get_edges(img, True)[0])

        # Once we have 5 frames, compute features
        if j == 5:
            section1_distance = np.mean(distance_list[0])
            section2_distance = np.mean(distance_list[1])
            section1_la = np.mean(largest_area_list[0])
            section2_la = np.mean(largest_area_list[1])
            slope_average = np.mean(edges_slope_list)

            features = [
                section1_distance, np.std(distance_list[0]),
                section2_distance, np.std(distance_list[1]),
                section1_la, np.std(largest_area_list[0]),
                section2_la, np.std(largest_area_list[1]),
                slope_average, np.std(edges_slope_list)
            ]

            data.append(features)
            labels.append(category_idx)

            # Slide window forward
            distance_list[0].pop(0)
            distance_list[1].pop(0)
        else:
            j += 1

data = np.array(data)
labels = np.array(labels)
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)


# === TRAIN / TEST SPLIT ===
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# === MODEL TRAINING ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# === EVALUATION ===
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {score * 100:.2f}%")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# === SAVE MODEL ===
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl ✅")
