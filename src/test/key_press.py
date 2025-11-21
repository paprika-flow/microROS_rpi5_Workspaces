import cv2

cv2.namedWindow("Key Tester")
img = 255 * cv2.ones((200, 300, 3), dtype=cv2.uint8) # Blank image
cv2.imshow("Key Tester", img)

print("Press arrow keys and observe the output. Press ESC to exit.")

while True:
    key = cv2.waitKeyEx(0) # Wait indefinitely for a key press
    if key == 27: # ESC key
        break
    if key != -1: # Only print if a key was actually pressed
        print(f"Key code: {key}")

cv2.destroyAllWindows()