"""
    RCNN_WormCamp_Segment - adapted from RCNN_Server

    Non-server-based (e.g. simple) segmentation of an image using an RCNN model. 
    Does not interface with C++, just returns data back in python variables

"""

# Imports
import time
import socket
from threading import Thread
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import numpy as np
import numpy.matlib
import os
from RCNN_Helper import loadRCNN


import matplotlib.pyplot as plt

# TODO: REMOVE THIS FUNCTION ITS FOR DEBUGGING ONLY. Show a matplotlib image OR list of images for debugging (up to 4)
def showDebugImage(images:list):

    # Force list
    if type(images) != list:  images = [images]

    # Show debug with as many subplots as needed up to 4
    plt.figure(1)

    for i,img in enumerate(images):

        if len(images) > 1: plt.subplot(2,2,i+1)
        plt.imshow(img)
        plt.colorbar

    plt.show()



print_elapsed_time = True
start = time.time()
def printElapsedTime(start,label,print_elapsed_time):
    end = time.time()
    print("-- ", label, " took ", round(end-start,3), "seconds")
    return end


# Images must be a list of images (even if only one)
#
# There seems to be a performance improvement of supplying a list instead of doing one-by-one using
# CUDA mode. Not much difference using CPU mode.
#
# TODO: Post-processing has not been updated to list-format an only operates on first image

def RCNN_Segment(model, images, params = None, use_cuda = False, obj_conf_thresh = 0.7, pix_conf_thresh = 0.8, 
                    ol_thresh = 0.75, clump_thresh = 0.05, min_obj_area = 10, show_timing = True):

    # If a parameters dict was supplied
    if params is not None:
        use_cuda = params["use cuda"]
        obj_conf_thresh = params["object conf thresh"]
        pix_conf_thresh = params["pix conf thresh"]
        ol_thresh = params["overlap thresh"]
        ol_erode = params["overlap erode size"]
        touch_dilate = params["touch dilate size"]
        clump_thresh = params["clump thresh"]
        min_obj_area = params["min object area"]
        show_timing = params["show timing"]
        



    # If a single image was supplied, convert to list
    if type(images) == np.ndarray: images = [images]

    # Convert all images to tensor format
    image_tensor = []
    for i in range(len(images)):
        pil2tensor = torchvision.transforms.ToTensor()
        image_tensor.append(pil2tensor(images[i]))
        if use_cuda:
            image_tensor[i] = image_tensor[i].cuda()

    ## PERFORM WORM IDENTIFICATION ON THE LIST OF IMAGES
    # Perform inference
    start = time.time()
    model.eval() # For inference
    with torch.no_grad():
        predictions_worms = model(image_tensor)           # Returns prediction

    if show_timing: start = printElapsedTime(start,"Worm Inference" ,print_elapsed_time)    


    # POST PROCESS AND DISAMBIGUATE ALL OF THE IMAGES

    # Extract masks from predictions_worms
    predictions_worms = predictions_worms[0] # Always 1 element if 1 element list of images is supplied
    masks_worms:np.ndarray = predictions_worms['masks'].permute([2,3,1,0]).cpu().numpy()
    conf_worms = predictions_worms['masks'].cpu().numpy() # TODO: Could use it instead of method below for conf of whole worm
    N_masks_worms = masks_worms.shape[3]

    # Eliminate tiny masks and masks with low overall confidence or area
    eliminated_idx = []
    compared_idx = [] # store pairs as string, e.g. 1 and 2 is "12" to mark compared pairs and avoid redoing
    for n in range(N_masks_worms):
        this_mask = np.mean(masks_worms[:,:,:,n],axis=2)>pix_conf_thresh
        #this_mask = masks_worms[:,:,0,n]>obj_conf_thresh
        if np.sum(this_mask) < min_obj_area:
            eliminated_idx.append(n)
            masks_worms[:,:,:,n] = np.zeros(masks_worms[:,:,:,n].shape,np.uint8)

    # Erode masks to help separate blobs that truly overlap (same worm) from blobs from two separate worms 
    # just touching
    #
    # An alternative to erode is just to increase the confidence threshold for pixels, but this has additional
    # consquences for completely overlooking lower confidence objects. 
    #print("DEBUG! ERODE")
    masks_worms_eroded = masks_worms.copy()
    if ol_erode is not None and ol_erode > 1:
        for n in range(N_masks_worms):
            masks_worms_eroded[:,:,0,n] = cv2.erode(masks_worms_eroded[:,:,0,n],(ol_erode,ol_erode))

    # Dilate masks to improve detection of touching worms (dilation starts from the original mask,
    # not the eroded one. )
    if touch_dilate is not None and touch_dilate > 1:
        for n in range(N_masks_worms):
            masks_worms[:,:,0,n] = cv2.dilate(masks_worms[:,:,0,n],(touch_dilate,touch_dilate))

    if show_timing: start = printElapsedTime(start,"Eliminate low conf objects" ,print_elapsed_time) 
    # Consolidate masks that are actually for the same worm
    touch_count = np.zeros((N_masks_worms,))

    for n in range(N_masks_worms):
        if n in eliminated_idx: continue
        this_mask = np.mean(masks_worms[:,:,:,n],axis=2)>pix_conf_thresh
        this_mask_eroded = np.mean(masks_worms_eroded[:,:,:,n],axis=2)>pix_conf_thresh
        A1 = np.sum(this_mask)

        is_duplicate = False # We will detemine if this mask is substantially the same as another
        is_touching = False # Or whether it's two separate worms touching

        # Compare it with all other masks
        for m in range(N_masks_worms):
            
            # Skip self comparisons
            if n==m: continue

            # Skip masks already eliminated
            if m in eliminated_idx: continue

            # Skip comparisons already done
            if str(n)+"-"+str(m) in compared_idx: continue

            # Check how much the smaller mask overlaps with the larger mask.  Check separately using the eroded and original images

            def calcOverlap(mask_arr, this_mask_use):
                test_mask = mask_arr[:,:,0,m]>pix_conf_thresh
                A2 = np.sum(test_mask)
                A_min = float(min((A1,A2)))
                ol_mask = this_mask_use & test_mask
                A_ol = float(np.sum(ol_mask))
                return A_ol, A_min, test_mask
            
            A_ol_orig, A_min_orig, test_mask_orig = calcOverlap(masks_worms, this_mask)
            A_ol_erode,A_min_erode, test_mask_erode = calcOverlap(masks_worms_eroded, this_mask_eroded)

            # Check if overlap occurs (multiple masks that are actually the same worm) using ERODED masks 
            if A_ol_erode/A_min_erode > ol_thresh:
                new_mask = this_mask_eroded | test_mask_erode
                new_mask = np.stack((new_mask,)*3,axis=-1).shape
                masks_worms_eroded[:,:,:,n] # Replace nth mask
                masks_worms_eroded[:,:,:,m] = np.zeros(masks_worms_eroded[:,:,:,n].shape,np.uint8)
                                # Set mth mask to empty
                eliminated_idx.append(m) # Mark mth mask as eliminated
                touch_count[m] = -1 # Negative touch count indicates duplicate worm

            # If overlaps by 10% or more, using the ORIGINAL masks, count them as clumping
            elif A_ol_orig/A_min_orig > clump_thresh:
                touch_count[n]+=1 # Increment the touching count for the n'th worm if the m'th is touching AND vice versa
                touch_count[m]+=1
            
            # Mark pairs that have already been compared in both orders
            compared_idx.append(str(n)+"-"+str(m))
            compared_idx.append(str(m)+"-"+str(n))
    
    if show_timing: start = printElapsedTime(start,"Object consolidation" ,print_elapsed_time)   
    

    # Apply eliminations 
    to_delete = touch_count < 0
    masks_worms = masks_worms[:,:,:,~to_delete]
    touch_count = touch_count[~to_delete]

    # Generate an overall mask (mostly useful for visualization)
    masks_sum = np.zeros(images[0].shape,np.uint8)
    boxes_sum = np.zeros(images[0].shape,np.uint8)
    temp_masks = np.any(masks_worms>pix_conf_thresh,axis=3).astype(np.uint8)*255
    temp_boxes = np.any(masks_worms,axis=3).astype(np.uint8)*255
    masks_sum[:,:,2] = temp_masks[:,:,0] # red channel BGR
    masks_sum[:,:,0] = temp_boxes[:,:,0] # put boxes in blue channel BGR

    alpha=0.7
    weighted_img_worms = cv2.addWeighted(images[0],alpha,masks_sum,1-alpha,None)

    # Generate a mask of touching worms
    masks_touching = np.zeros(images[0].shape,np.uint8)
    temp_masks = np.any((masks_worms>pix_conf_thresh) & (touch_count > 0),axis=3).astype(np.uint8)*255
    masks_touching[:,:,1] = temp_masks[:,:,0]
    masks_touching[:,:,2] = temp_masks[:,:,0] # yellow in BGR

    weighted_img_touching = cv2.addWeighted(images[0],alpha,masks_touching,1-alpha,None)
    if show_timing: start = printElapsedTime(start,"Output image generation" ,print_elapsed_time)   
    return masks_worms, masks_worms>pix_conf_thresh, weighted_img_worms,\
                weighted_img_touching, touch_count 


# Testing main func
if __name__ == "__main__":

    # Load image
    fimg = r"C:\WormWatcher\Samples\raw images\2020-09-09 (00-02-20)_roi1.png"
    image = cv2.imread(fimg,cv2.IMREAD_COLOR)

    # Load network
    fpth_worm = r"C:\Dropbox\Autogans Supporting Files\Worm RCNN\resnet50_2classes_Combo_aug_00-02.pth"

    # Auto-select GPU or CPU
    print("Loading model...")
    use_cuda = torch.cuda.is_available() # Automatically decide whether to use CUDA
    model_worm = loadRCNN(fpth_worm,2,use_cuda)

    print("Initializing segmentation: CUDA = ",use_cuda)
    confidence_masks, threshed_masks, weighted_img_worms, weighted_img_touching = RCNN_Segment(model_worm,[image],use_cuda,
                                                pix_conf_thresh=0.4, obj_conf_thresh = 0.00,
                                                ol_thresh = 1.0, clump_thresh = 0.001)
    # Confidence masks needs to be thresholded if looking for binary

    fout = r'C:\WormWatcher\Samples\original_image.png'
    cv2.imwrite(fout,image)    
    fout = r'C:\WormWatcher\Samples\worm_image.png'
    cv2.imwrite(fout,weighted_img_worms)
    fout = r'C:\WormWatcher\Samples\touching_image.png'
    cv2.imwrite(fout,weighted_img_touching)    
    #cv2.imshow("Preview",weighted_img_touching), cv2.waitKey(0)
    pass


