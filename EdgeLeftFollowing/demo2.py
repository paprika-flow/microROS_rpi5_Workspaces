import os
import cv2
import argparse
import torch
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Class names for Cityscapes / Citys
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

parser = argparse.ArgumentParser(description='Binary sidewalk+road segmentation on folder of images')
parser.add_argument('--dataset', type=str, default='citys', help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights', help='Directory for saving checkpoint models')
parser.add_argument('--input-folder', default='./photos/photo_20251006_215401.jpg', type=str, help='path to folder with input pictures')
parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)
args = parser.parse_args()

ROAD_CLASS = 0
SIDEWALK_CLASS = 1

def process_image(img, model, device):
    # Transform input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    # Handle both ndarray and file input
    if isinstance(img, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        image = Image.open(img).convert('RGB')

    image = transform(image).unsqueeze(0).to(device)

    # Run model
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().numpy()

    # Binary mask for road + sidewalk
    binary_mask = np.isin(pred, [ROAD_CLASS, SIDEWALK_CLASS]).astype(np.uint8) * 255
    return binary_mask


def demo_folder(img, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model once
    #model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=True).to(device)
    print('? Model loaded!')

    # Process all images in folder
    return process_image(img, model, device)
