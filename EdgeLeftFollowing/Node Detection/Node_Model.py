import cv2 as cv
import os
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from extracting_features import get_edges_top, get_distance_mask_points


# === MAIN TRAINING PIPELINE ===
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
# mean_section1_x = []
std_section2_x = []
# mean_section2_x = []

std_section1_y = []
mean_section1_y = []
std_section2_y = []
mean_section2_y = []

std_slope = []
mean_slope = []

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
        edges_slope_list.append(get_edges_top(img, True)[0])

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
            # mean_section1_x.append(section1_distance_x)
            std_section2_x.append(np.std(distance_x_list[1]))
            # mean_section2_x.append(section2_distance_x)

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
            labels.append(category_idx)


data = np.column_stack([
    mean_section1_y,
    mean_section2_y,
    mean_section1_la,
    mean_section2_la,
    mean_slope,
    std_section1_x,
    std_section2_x,
    std_section1_y,
    std_section2_y,
    std_section1_la,
    std_section2_la,
    std_slope
])

labels = np.array(labels) # 0 = NotNode, 1 = Node
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)


# === TRAIN / TEST SPLIT ===
'''x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)'''
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)
# === Standarization

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 

# === MODEL TRAINING ===
'''model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)'''




from sklearn.model_selection import GridSearchCV
'''When you later load the model to predict new data, you must standardize it with the same scaler you saved.'''
params = {
    'C': [0.5, 1, 2, 5],
    'gamma': ['scale', 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(class_weight='balanced'), params, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)

print("Best params:", grid.best_params_)
model = grid.best_estimator_


# === EVALUATION ===
y_pred = model.predict(x_test)
score = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {score * 100:.2f}%")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# === SAVE MODEL ===
with open("model_and_scaler.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)


print("\nModel saved as model.pkl ✅")


