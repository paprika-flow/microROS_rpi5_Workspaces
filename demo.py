import os
import time
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from visualize import get_color_pallete
from fast_scnn import get_fast_scnn

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    # Load Fast-SCNN once at startup
    model = get_fast_scnn('citys', pretrained=True, root='./weights').to(device)
    model.eval()
    print("✅ Model loaded!")

    # Preprocessing transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    os.makedirs("output", exist_ok=True)
    frame_count = 0

    print("🎥 Starting live segmentation... Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Failed to read frame")
                continue

            # Convert BGR (OpenCV) → RGB (PIL)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Preprocess and run model
            image_tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
            pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()

            # Create overlay
            mask = get_color_pallete(pred, 'citys').convert('RGB')
            overlay = Image.blend(img_pil, mask, alpha=0.7)
            overlay_np = np.array(overlay)

            # Combine and display
            combined = np.hstack((img_rgb, overlay_np))
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.imshow('Fast-SCNN Live Segmentation', combined_bgr)

            # Save every 50 frames (optional)
            frame_count += 1
            if frame_count % 50 == 0:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"output/frame_{timestamp}.jpg", combined_bgr)
                print(f"💾 Saved frame_{timestamp}.jpg")

            # Quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("🛑 Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Stream ended.")


if __name__ == "__main__":
    main()
