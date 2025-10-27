import os
import pickle
import cv2  # type: ignore
from skimage.io import imread # type: ignore
from sklearn.svm import LinearSVC  # k-NN classifier from scikit-learn
from skimage.transform import resize # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC # type: ignore
from sklearn.metrics import accuracy_score # type: ignore
import numpy as np # type: ignore

def mask_features(mask):
    
    mask = (mask > 0.5).astype(np.uint8)
    black_mask = 1 - mask
    num_labels, _ = cv2.connectedComponents(mask)
    coveage = np.mean(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = max([cv2.contourArea(c) for c in contours], default=0)
    solidity = 0
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_area = cv2.contourArea(hull)
        solidity = largest_area / hull_area if hull_area > 0 else 0
    # === NEW FEATURE: mean position of white and black pixels ===
    white_coords = np.column_stack(np.where(mask > 0))
    black_coords = np.column_stack(np.where(black_mask > 0))

    # Mean positions (y, x)
    white_mean = np.mean(white_coords, axis=0) if len(white_coords) > 0 else np.array([0, 0])
    black_mean = np.mean(black_coords, axis=0) if len(black_coords) > 0 else np.array([0, 0])

    # Compute distance and ratio
    distance = np.linalg.norm(white_mean - black_mean)  # Euclidean distance
    pos_ratio = (white_mean[0] + white_mean[1]) / (black_mean[0] + black_mean[1] + 1e-6)  # avoid division by zero

    return np.array([pos_ratio, distance, largest_area, solidity])
print("oi")

input_dir = r'C:\\Users\\User\\Downloads\\model_sidewalk\\microROS_rpi5_Workspaces\\separated_classes_output'
categories = ['good', 'bad']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(mask_features(img))
        labels.append(category_idx)


data = np.array(data)
labels = np.array(labels)

#train / test set

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)



pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(kernel='rbf'))
])

parameters = {
    'svc__C': [1, 10, 100, 1000],
    'svc__gamma': [0.01, 0.001, 0.0001]
}



grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)


# test performance

best_estimator = grid_search.best_estimator_

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score*100)))
print(classification_report(y_test, y_prediction))
print(confusion_matrix(y_test, y_prediction))





'''
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()'''

pickle.dump(best_estimator, open('./model.p', 'wb'))