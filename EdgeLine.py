import cv2
import numpy as np

acceptable_b = [128, 232]
acceptable_g = [64, 35]
acceptable_r= [128, 244]

def get_edges(img):
  height, width, channel = img.shape

  stopped = 1
  line_points = []
  previous = np.array([0, 0, 0])

  for i in range(width -1, 0, -1):
    if stopped <= 0:
      stopped = 1
    for j in range(height - stopped, 0, -1):
      pixel = img[j, i]       # pixel is [B, G, R]
      b, g, r = pixel
      if(j != height - stopped and r not in acceptable_r or g not in acceptable_g or b not in acceptable_b): 
        if not np.array_equal(previous, pixel):
          line_points.append([j, i ])
        stopped = j -20 
        break
      previous = pixel
  N = len(line_points)
  if N > 1:
    points = np.array(line_points)

    y = points[:,0]
    x = points[:,1]
    # Fit line: y = m*x + b
    A = np.vstack([x, np.ones(len(line_points))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    print(f"Slope: {m}, Intercept: {b}")
    
  
  return [m, b]



# Load image
img = cv2.imread("C:\\Users\\User\\Downloads\\Fast-SCNN\\Fast-SCNN-pytorch\\test_result\\photo_20251006_215623.png")

# Define the cropping region
# Syntax: img[y1:y2, x1:x2]
left_cropped = img[240:480, 000:240]
right_cropped = img[240:480, 400:640]
right_slope, right_intercept = get_edges(right_cropped)
left_slope, left_intercept = get_edges(left_cropped)
# Show or save cropped image
print(right_slope, right_intercept)
print(left_slope, left_intercept)

width = 640
height_cropped = 240
for x in range(width-1, 0, -1):
    if x < 240:
      y = int(left_slope*x + left_intercept)
    elif x >= 400:
      y = int(right_slope*(x -400) + right_intercept)
    else: 
      continue
    img[y + height_cropped if y< 240 else 239,x] = [0,0,0]

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
