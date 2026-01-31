import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Define the classes your model was trained on
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic_light", "traffic_sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# Define the specific classes you want to include in the binary mask
ROAD_CLASS = 0
SIDEWALK_CLASS = 1

def segmentation_inference(cv_image, model, device):
    """
    Takes an OpenCV image, runs it through the Fast-SCNN model,
    and returns a binary segmentation mask.
    """
    # 1. Convert OpenCV BGR image to PIL RGB image
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    # 2. Define the same transformations as used during training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 3. Run inference with the model
    with torch.no_grad():
        outputs = model(input_tensor)
        # Get the class prediction for each pixel
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().numpy()

    # 4. Create a binary mask where pixels are 255 if they belong
    #    to the road or sidewalk class, and 0 otherwise.
    binary_mask = np.isin(pred, [ROAD_CLASS, SIDEWALK_CLASS]).astype(np.uint8) * 255
    
    return binary_mask