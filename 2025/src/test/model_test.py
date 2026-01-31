#!/usr/bin/env python3
# encoding: utf-8

import cv2
import numpy as np
import torch
import torch.nn as nn
import sys
import os
import time
import argparse

# Import model architecture
try:
    from models.fast_scnn import get_fast_scnn
except ImportError:
    print("Error: Could not import 'models.fast_scnn'.") 
    print("Make sure you run this script from the folder containing the 'models' directory.")
    sys.exit(1)

# ==========================================
#               CONFIGURATION
# ==========================================
class Config:
    DEFAULT_WEIGHTS = "florida_sidewalk.pth"
    CAMERA_INDEX = 0
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SIDEWALK_CLASS = 1

# ==========================================
#           HELPER FUNCTIONS
# ==========================================

def perform_surgery(model):
    classifier_block = model.classifier.conv
    if isinstance(classifier_block, nn.Sequential):
        for i, layer in enumerate(classifier_block):
            if isinstance(layer, nn.Conv2d):
                classifier_block[i] = nn.Conv2d(layer.in_channels, 2, kernel_size=1)
                break
    else:
        model.classifier.conv = nn.Conv2d(classifier_block.in_channels, 2, kernel_size=1)
    return model

def load_model(weights_path, device):
    print(f"[INFO] Loading model structure...")
    model = get_fast_scnn('citys', pretrained=False, root='./weights', map_cpu=True)
    model = perform_surgery(model)
    
    print(f"[INFO] Loading weights from: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file not found at {weights_path}")
        sys.exit(1)
        
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def preprocess_image(img_bgr):
    img_resized = cv2.resize(img_bgr, (512, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_float - mean) / std
    img_chw = img_norm.transpose((2, 0, 1))
    return torch.from_numpy(img_chw).unsqueeze(0).float()

# ==========================================
#                 MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Test FastSCNN Model with Camera Capture.")
    parser.add_argument("-w", "--weights", type=str, default=Config.DEFAULT_WEIGHTS, help="Path to .pth weights file")
    args = parser.parse_args()

    # 1. Load Model First (So we don't keep camera waiting)
    model = load_model(args.weights, Config.DEVICE)
    print(f"[INFO] Model loaded on {Config.DEVICE}")

    # 2. Open Camera
    print(f"[INFO] Opening Camera {Config.CAMERA_INDEX}...")
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    # 3. Warm Up (Wait 1 Second)
    print("[INFO] Warming up camera for 1 second...")
    time.sleep(1.0)

    # 4. Capture One Frame
    print("[INFO] Capturing frame...")
    ret, frame = cap.read()
    cap.release() # Release immediately

    if not ret:
        print("[ERROR] Failed to capture frame.")
        sys.exit(1)
        
    print(f"[INFO] Frame captured. Shape: {frame.shape}")

    # 5. Preprocess
    input_tensor = preprocess_image(frame).to(Config.DEVICE)

    # 6. Inference
    print("[INFO] Running inference...")
    t0 = time.time()
    with torch.no_grad():
        output = model(input_tensor)[0]
        pred = torch.argmax(output, 1).squeeze(0).cpu().numpy()
    print(f"[INFO] Inference took: {time.time() - t0:.3f}s")

    # 7. Mask Generation
    print("[INFO] Generating visualization...")
    mask_small = (pred == Config.SIDEWALK_CLASS).astype(np.uint8) * 255
    h, w = frame.shape[:2]
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 8. Visualization (Green Overlay)
    overlay = frame.copy()
    overlay[mask == 255] = [0, 255, 0] 
    blended = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Stack: Original | Mask | Overlay
    debug_view = np.hstack((frame, mask_bgr, blended))
    
    # Resize if too big for screen
    display_h, display_w = debug_view.shape[:2]
    if display_w > 1800:
        scale = 1800 / display_w
        debug_view = cv2.resize(debug_view, None, fx=scale, fy=scale)

    print("[INFO] Displaying result. Press any key to exit.")
    cv2.imshow("Camera Test: Original | Mask | Overlay", debug_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()