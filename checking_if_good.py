import os
import pickle
import cv2
import numpy as np
from skimage.io import imread
from skimage.transform import resize

# === Load the trained model ===
model = pickle.load(open('./model.p', 'rb'))
print("✅ Model loaded successfully.")

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


test_dir = r'C:\\Users\\User\\Downloads\\model_sidewalk\\microROS_rpi5_Workspaces\\separated_classes_output\\test_images'

for file in os.listdir(test_dir):
    if not file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue

    img_path = os.path.join(test_dir, file)
    img = imread(img_path)
    img = resize(img, (15, 15))  # same size as training

    features = mask_features(img).reshape(1, -1)
    prediction = model.predict(features)[0]

    label = "GOOD" if prediction == 0 else "BAD"

    print(f"{file} → Predicted: {label}")


cv2.destroyAllWindows()