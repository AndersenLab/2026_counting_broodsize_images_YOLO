# RCNN_json2mask()
#
# Converts data labeled in myvision.ai, output in VGG JSON format, to masks compatible
# with Mask_RCNN_Train. It automatically formats into DatasetImages and DatasetMasks
# subfolders, so there's no need to run RCNN_FormatLabeledDataset.py.
#
# You will still need to augment.
#
# Pdata must contain:
#   (1) The image files, with names matching the names in the JSON file
#   (2) One Json file
#
# Note that only images with any worms labeled are given masks



#####    PARAMETERS   #####

pdata_list = [
    #r"H:\Tau worm mask RCNN\Model 3 - worms aggregation\Data for labeling\well00",
    #r"H:\Tau worm mask RCNN\Model 3 - worms aggregation\Data for labeling\well03",
    #r"H:\Tau worm mask RCNN\Model 3 - worms aggregation\Data for labeling\well04",
    #r"H:\Tau worm mask RCNN\Model 3 - worms aggregation\Data for labeling\well05",
    #r"I:\Tau worm mask RCNN\Model 4a - worms single plate DF rev 1 small imgs\Data for labeling"
    #r"I:\Tau worm mask RCNN\Model 4b - with new data Oct 2025\Data for labeling"
    r"I:\Tau worm mask RCNN\Model 4d - Validation data only\Data for labeling"
]

pout = r"I:\Tau worm mask RCNN\Model 4d - Validation data only\Original"

#####   LIBRARIES   #####

import json
import os
import cv2
import numpy as np

#####   Find and load JSON dict   #####
for pdata in pdata_list:

    if not "Data for labeling" in pdata: raise Exception("WARNING: This data should be in a Data for labeling folder...")

    # Load VGG JSON info
    fjsons = [ fn for fn in os.listdir(pdata) if ".json" in fn]
    if len(fjsons) != 1: raise Exception("Too many or too few JSONS in folder")
    fjson = fjsons[0]

    with open(os.path.join(pdata,fjson)) as jfile:
        d = json.load(jfile)


    # Create output paths if they don't exist yet
    if not os.path.exists(os.path.join(pout,"DatasetImages")): 
        os.makedirs(os.path.join(pout,"DatasetImages"))

    if not os.path.exists(os.path.join(pout,"DatasetMasks")):
        os.makedirs(os.path.join(pout,"DatasetMasks"))

    # Generate masks for each 
    N_completed = 0
    for n,key in enumerate(d):

        # Check whether this key has any regions
        if len(d[key]["regions"]) == 0:
            continue

        # Load the corresponding image
        fimg = os.path.join(pdata,d[key]["filename"])
        img = cv2.imread(fimg,cv2.IMREAD_COLOR) # Training uses color images

        # Initialize corresponding mask as all zeros (should also be color)
        mask = np.zeros(img.shape)

        # Draw all worm outlines in the red channel of the mask
        startnum = 100
        for i,region in enumerate(d[key]["regions"]):

            # COnvert XY to NP array
            x = np.array(region["shape_attributes"]["all_points_x"])
            y = np.array(region["shape_attributes"]["all_points_y"])
            contour = np.stack((x,y),axis=1).reshape((-1,1,2)).astype(np.int32)

            # Draw contour in red channel as filled
            idx = startnum+i
            mask = cv2.drawContours(mask,[contour],0,(0,0,idx),-1)




        # Write image and corresponding mask to output in formatted subfolders
        fout_img = os.path.join(pout,"DatasetImages",d[key]["filename"])
        fout_mask= os.path.join(pout,"DatasetMasks",
                                    d[key]["filename"].replace(".png","_mask.png"))
        cv2.imwrite(fout_img,img)
        cv2.imwrite(fout_mask,mask)
        N_completed += 1
        if N_completed%10 == 0:
            print("Completed %d images..." % N_completed)


    print("Formatted & Sorted %d labeled images" % N_completed)

