import os
import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from pkg_FastSCNN.fast_scnn import get_fast_scnn

# ====== CONFIG ======
IMAGE_PATH = "/home/clement_workspace/src/pkg_FastSCNN/test/photo_20251006_214629.jpg"
WEIGHTS_DIR = "/home/clement_workspace/src/pkg_FastSCNN/weights"
DATASET = "citys"   # FastSCNN was trained on Cityscapes
# ====================

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load FastSCNN with your pretrained checkpoint
    model = get_fast_scnn(
        DATASET,
        pretrained=True,
        root=WEIGHTS_DIR,
        map_cpu=True     # set True if weights were trained on GPU but loading on CPU
    ).to(device)

    model.eval()
    print("[INFO] FastSCNN model loaded.")
    return model, device


def preprocess(image_path, device):
    img = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0).to(device)


def run_inference(model, device, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)[0]     # output is (B, C, H, W)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return pred


def visualize_mask(image_path, pred_mask):
    # Map road (0) + sidewalk (1) to white
    ROAD = 0
    SIDEWALK = 1

    binary = np.isin(pred_mask, [ROAD, SIDEWALK]).astype(np.uint8) * 255

    # Save result next to original
    out_path = image_path.replace(".jpg", "_mask.png")
    cv2.imwrite(out_path, binary)
    print(f"[INFO] Saved segmentation mask:\n{out_path}")


def main():
    model, device = load_model()
    image_tensor = preprocess(IMAGE_PATH, device)
    pred_mask = run_inference(model, device, image_tensor)
    visualize_mask(IMAGE_PATH, pred_mask)


if __name__ == "__main__":
    main()
