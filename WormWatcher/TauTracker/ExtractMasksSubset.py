# Extract a subset of masks, corresponding to images in folder pimg, from a  folder of masks pmasks_in
# And put the matching masks in pmasks_out

# -----------------------------  Parameters - input and output folders  ------------------------------- #

pimg = r"H:\Tau worm mask RCNN\Model 3a - worms aggregation rev 1\Data for refinement\DatasetImages"
pmasks_in = r"H:\Tau worm mask RCNN\Model 3 - worms aggregation\Data for training\DatasetMasks"
pmasks_out = r"H:\Tau worm mask RCNN\Model 3a - worms aggregation rev 1\Data for refinement\DatasetMasks"


# -----------------------------  Find matching masks  ------------------------------- #

import os, shutil, cv2
import matplotlib.pyplot as plt
import numpy as np

def displayMask(fmask):
    # When finished show the scaled version of the last masks
    img = cv2.imread(os.path.join(pmasks_out, fmask), cv2.IMREAD_GRAYSCALE)

    # Get min non-zero value and max value
    min_val = img[img > 0].min()
    max_val = img.max()

    # Normalize the image excluding zeros
    mask = np.where(img > 0, 255, 0).astype(np.uint8)
    norm_img = cv2.normalize(img.astype(np.float32), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U, mask=mask)

    # Assign original zeros back to normalized image
    norm_img[img == 0] = 0


    plt.imshow(norm_img)
    plt.show()


fimg_list = os.listdir(pimg)
for i,fimg in enumerate(fimg_list):

    # Find mask file
    found_mask = False
    for fmask in os.listdir(pmasks_in):
        if fimg in fmask.replace("_mask",""):
            found_mask = True
            break
    if not found_mask: raise Exception(f"Mask not found for image {fimg}")

    # Copy it to output location
    shutil.copy(os.path.join(pmasks_in, fmask),os.path.join(pmasks_out, fmask))

    # Print update
    print(f"found and copied mask for image {i+1} of {len(fimg_list)}")
    displayMask(os.path.join(pmasks_out,fmask))

