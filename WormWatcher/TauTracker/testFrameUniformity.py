# testFrameUniformity
#
# test script to test whether any frames in a series have acute changes or defects, like black pixels
# from a failed frame grab etc. 
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy

# Find files
pdata = r""
fnames = os.listdir(pdata)
fnames = [f for f in fnames if "png" in f]

# Save suspect frames
diff_thresh = 40
pout = os.path.join(pdata,"diff frames")
if not os.path.exists(pout):
    os.makedirs(pout)

# Look for changed pixels
diff_vals = []
for i in range(1,len(fnames)):

    print("loading frame %d of %d"  % (i, len(fnames)-1))

    # Set the reference image, either loading it or using last iteration's test image
    if i == 1:
        img0 = cv2.imread(os.path.join(pdata,fnames[i-1]),cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        img0 = copy.deepcopy(img1)
    
    # Set the test image
    img1 = cv2.imread(os.path.join(pdata, fnames[i]),cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Get diff image
    diff_img = np.abs(img1-img0)

    # Store the maximum pixel difference between these images.
    diff_vals.append(np.max(diff_img))

    # Save the difference image if greater than thresh. Normalize it 2 1
    if np.max(diff_img) > diff_thresh:
        diff_img /= np.max(diff_img)
        diff_img = (diff_img*255).astype(np.uint8)
        cv2.imwrite(os.path.join(pout,fnames[i]),diff_img)
        
# Save csv
fout = "max pixel diffs.csv"
arr = np.array(diff_vals)
np.savetxt(os.path.join(pdata,fout),arr,delimiter=",")

# Plot results
plt.figure(1)
plt.plot(arr)
plt.show()

# Print top differences for easy lookup
arr_sorted = np.sort(arr)

for i in range(-10,-1):
    print("Frame %d difference %0.2f" % (i,arr_sorted[i]))


