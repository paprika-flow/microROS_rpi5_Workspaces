import os
import argparse
import torch
import numpy as np
import cv2

from demo2 import process_image 
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image
from utils.visualize import get_color_pallete

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='citys',
                    help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='/home/pi/Downloads/photos/photos/photo_20251007_030253.jpg',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)

args = parser.parse_args()

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_fast_scnn(args.dataset, pretrained=True,
                          root=args.weights_folder, map_cpu=True).to(device)
    print('Finished loading model!')
    model.eval()
    return model

def demo(img, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

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

    

    with torch.no_grad():
        outputs = model(image)

    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset).convert('RGB')  # ? Convert to RGB here

    return mask
