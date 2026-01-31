import cv2

def main():
    cap = cv2.VideoCapture(0)   # try 0, 1, 2 depending on your camera

    if not cap.isOpened():
        print("❌ Could not open camera.")
        return

    # --- NEW CODE ADDED HERE ---
    WINDOW_NAME = "Camera Feed"
    WIDTH = 640
    HEIGHT = 480
    
    # 1. Create a named window and set the flag to allow resizing
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL) 
    
    # 2. Set the desired size
    cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)
    # --- END NEW CODE ---

    print(f"📷 Camera opened. Window size set to {WIDTH}x{HEIGHT}. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Use the same name defined above
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()