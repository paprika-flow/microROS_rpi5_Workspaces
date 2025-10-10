import os
import argparse
import torch
from torchvision import transforms
from fast_scnn import get_fast_scnn
from PIL import Image
import numpy as np

# Class names for Cityscapes / Citys
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

parser = argparse.ArgumentParser(description='Binary sidewalk+road segmentation on folder of images')
parser.add_argument('--dataset', type=str, default='citys', help='dataset name (default: citys)')
parser.add_argument('--weights-folder', default='./weights', help='Directory for saving checkpoint models')
parser.add_argument('--input-folder', default='./photos', type=str, help='path to folder with input pictures')
parser.add_argument('--outdir', default='./separated_classes_output', help='path to save the result')
parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)
args = parser.parse_args()

ROAD_CLASS = 0
SIDEWALK_CLASS = 1

def process_image(image_path, model, device, outdir):
    # Create dedicated folder for this image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    image_outdir = os.path.join(outdir, base_name)
    os.makedirs(image_outdir, exist_ok=True)

    # Transform input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    orig = Image.open(image_path).convert('RGB')
    image = transform(orig).unsqueeze(0).to(device)

    # Run model
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs[0], dim=1)[0]  # [num_classes, H, W]
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().numpy()

    # -----------------------------
    # Create binary mask (road+sidewalk)
    # -----------------------------
    binary_mask = np.isin(pred, [ROAD_CLASS, SIDEWALK_CLASS]).astype(np.uint8) * 255
    mask_image = Image.fromarray(binary_mask)
    mask_path = os.path.join(image_outdir, f"{base_name}_road_sidewalk_bw.png")
    mask_image.save(mask_path)

    # -----------------------------
    # Overlay side by side with original
    # -----------------------------
    overlay = Image.blend(orig, mask_image.convert('RGB'), alpha=0.7)
    w, h = orig.size
    combined = Image.new('RGB', (w * 2, h))
    combined.paste(orig, (0, 0))
    combined.paste(overlay, (w, 0))
    combined_path = os.path.join(image_outdir, f"{base_name}_overlay.png")
    combined.save(combined_path)

    # -----------------------------
    # Save per-class probability maps
    # -----------------------------
    prob_dir = os.path.join(image_outdir, "probs")
    os.makedirs(prob_dir, exist_ok=True)
    for idx, cls_name in enumerate(CLASS_NAMES):
        prob_map = (probs[idx].cpu().numpy() * 255).astype(np.uint8)
        prob_path = os.path.join(prob_dir, f"{cls_name}_prob.png")
        Image.fromarray(prob_map).save(prob_path)

    print(f"✅ Processed: {image_path} -> {image_outdir}")


def demo_folder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model once
    model = get_fast_scnn(args.dataset, pretrained=True, root=args.weights_folder, map_cpu=args.cpu).to(device)
    print('✅ Model loaded!')

    # Process all images in folder
    image_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img_file in sorted(image_files):
        img_path = os.path.join(args.input_folder, img_file)
        process_image(img_path, model, device, args.outdir)

    print("✅ Finished processing all images.")


if __name__ == '__main__':
    demo_folder()
