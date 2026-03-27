import cv2
import numpy as np
from verifyDirlist import *
import csv


path_dataset = r'C:\Users\pdb69\Desktop\WormCamp Cluster Assays\Testing orig format'

image_paths, img_names = verifyDirlist(path_dataset, False, ".png")
bbox_paths,bbox_names = verifyDirlist(path_dataset, False, ".csv")

skipped_empty = 0
skipped_csv = 0

for idx, image_path in enumerate(image_paths):
    print(f"Masking image {idx + 1} of {len(image_paths)}.")
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    bbox_file = image_path.replace('.png', '.csv')
    if bbox_file not in bbox_paths:
        print(f"    No corresponding bounding box file for image: {image_path}")
        skipped_csv += 1
        continue

    #print(f'image shape: {image.shape}')
    #print(f'image type: {image.dtype}')
    #final_mask = np.zeros(image.shape, dtype = image.dtype)
    final_mask_blur = np.zeros(image.shape, dtype = image.dtype)

    bboxes = []
    with open(bbox_file, 'r') as bboxes_file:
        bboxreader = csv.reader(bboxes_file)
        fields = next(bboxreader)
        for row in bboxreader:
            bboxes.append(row)

    if len(bboxes) == 0:
        print(f"    No bounding boxes drawn for image: {image_path}")
        skipped_empty += 1
        continue

    current_worm_pix_val = 100
    for bbox in bboxes:
        x = int(bbox[1])
        y = int(bbox[2])
        w = int(bbox[3])
        h = int(bbox[4])

        img_roi = image[y:y+h, x:x+w] # extract the worm roi based on its bounding box
        #cv2.imshow("roi", img_roi)
        #cv2.waitKey(0)

        blur = cv2.GaussianBlur(img_roi,(3,3),0) # slightly blur the roi image to reduce noise. This will helo Otsu produce a better mask
        #cv2.imshow("blur", blur)
        #cv2.waitKey(0)

        #otsu_val, otsu_img =  cv2.threshold(img_roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Create a mask by thresholding the image with Otsu
        otsu_val_b, otsu_img_b =  cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Create a mask by thresholding the image with Otsu
        #print(f'otsu img: \n{otsu_img}')
        #print(f"otsu shape: {otsu_img.shape}")
        #cv2.imshow("thresh", otsu_img)
        #cv2.waitKey(0)
        
        for row in range(otsu_img_b.shape[0]):
            for col in range(otsu_img_b.shape[1]):
                #if otsu_img[row][col] != 0:
                #    final_mask[row + y][col + x] = current_worm_pix_val
                if otsu_img_b[row][col] != 0:
                    final_mask_blur[row + y][col + x] = current_worm_pix_val

        current_worm_pix_val += 1


        #cv2.imshow("image", image)
        #cv2.imshow("final_mask", final_mask)
        #cv2.waitKey(0)

    #cv2.imshow("image", image)
    #cv2.imshow("final_mask", final_mask)
    #cv2.imshow("final_mask_blur", final_mask_blur)
    #cv2.waitKey(0)

    # Convert the final mask into an RGB image (which is necessary for Mask RCNN training)
    blank_channel = np.zeros(image.shape, dtype = image.dtype)


    mask_bgr = np.dstack((blank_channel,blank_channel, final_mask_blur)) # We want the masks on the red channel, which is the 3rd channel in Open CV (they use BGR format)

    mask_path = image_path.replace('.png', '_mask.png')
    cv2.imwrite(mask_path, mask_bgr)
   
print("Done Otsu masking the images.")
print(f"{len(image_paths)} images were found. {len(bbox_paths)} csv files were found.")
print(f"{len(image_paths) - skipped_csv - skipped_empty} of {len(image_paths)} images were masked.\n\t{skipped_empty} images were skipped because the images were empty and had no worms. (The mask RCNN network requires every input image to have a corresponding mask.)\n\t{skipped_csv} images were skipped because the image had no corresponding csv file.")


##### Debugging bad bounding box.
#p = r'C:\Users\pdb69\Desktop\WormCamp Cluster Assays\TrainingDataset\DatasetMasks\C4_3C_5_2019-06-03 (21-05-35)_ROI=22 (1)_mask.png'
#image = cv2.imread(p,cv2.IMREAD_COLOR)
#cv2.imshow("Image", image)
##cv2.waitKey(0)
#bad_mask = (image==143).astype(np.uint8)
#bad_mask *= 200

##print(f'unique ids: {np.unique(image)}')
#cv2.imshow("Mask", bad_mask)
#cv2.waitKey(0)

#pos = np.where(bad_mask)
#print(pos)
#xmin = np.min(pos[1])
#xmax = np.max(pos[1])
#ymin = np.min(pos[0])
#ymax = np.max(pos[0])
#print(xmin, xmax, ymin, ymax)

