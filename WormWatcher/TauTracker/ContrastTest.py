import cv2
import numpy as np

# Load image from PNG file
img_path = r'C:\Dropbox\Tau Scientific Instruments\Jobs\2024-05 Erik JHU BIPs\Sample images for sharing\Contrast adjustment Nov 2024\Plate_1_TAU (zoom).png'
img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)

# Check if image was successfully loaded
if img is None:
    print("Error: Could not read image.")
    exit()

# If the image is colored, convert it to grayscale
if len(img.shape) == 3:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    img_gray = img

# Compute lower and upper limits (2nd and 98th percentiles)
lower_limit = np.percentile(img_gray, 48)
upper_limit = np.percentile(img_gray, 95)

# Avoid division by zero in case of a flat image
if upper_limit == lower_limit:
    upper_limit = img_gray.max()
    lower_limit = img_gray.min()

# Adjust contrast by normalizing to uint8 with adjusted limits
img_clipped = np.clip(img_gray, lower_limit, upper_limit)
img_normalized = ((img_clipped - lower_limit) / (upper_limit - lower_limit) * 255).astype(np.uint8)

# Make the image smaller
img_normalized_small = cv2.resize(img_normalized, (0, 0), fx=0.5, fy=0.5)

# Save the image to the same folder and name with _normalized appended
# unpack format string e.g. .bmp or .png
fmt = img_path.split('.')[-1]
cv2.imwrite(img_path.replace('.' + fmt, '_normalized.' + fmt), img_normalized)

# Display the image
cv2.imshow('Adjusted Image', img_normalized_small)
cv2.waitKey(0)
cv2.destroyAllWindows()
