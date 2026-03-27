# Worm.py
# 
# Class for storing an individual worm's information and various single-worm tracking and 
# processing operations

import numpy as np
import cv2
import os
import OpenCV_Helpers as cv2h
from enum import Enum



class lm(Enum):
    BF = 0
    DF = 1

class Worm:

    # Initialize a worm object, optionally providing the "global mask", e.g. the mask 
    # Showing where the worm is in the entire image. Global mask should be only
    # ones and zeros, it should not be the raw Mask RCNN output showing all worms 
    # as different numbers, just a single worm.
    #
    # To save memory, the mask is always converted to a contour.
    # To translate it to real coordinates, draw the contour 
    #
    # global_mask should be the mask of the worm in the entire image, and ONLY
    # that one worm. It will be converted to contour
    def __init__(self,single_mask = np.array([]), light_mode = lm.DF):
        self.light_mode = light_mode
        
        # Secondary classification attributes
        self.classification_status = "pending"
        self.primary_confidence = 0.0
        self.secondary_confidence = 0.0
        self.secondary_classification = None
        
        # Only contourize if we have a valid mask
        if single_mask.size > 0:
            self.contourize(single_mask)
        else:
            # Initialize with empty values
            self.contours = np.zeros((2,1,2),np.int32)
            self.area = 0
            self.centroid = [0,0]
        
        return

    # TODO: Try this on an empty no worm image
    def contourize(self,single_mask = np.array([])):

        # If image is wrong type, fix it
        if single_mask.dtype != np.uint8: single_mask = single_mask.astype(np.uint8)

        # If image is empty or has no non-zero values, return with empty contours
        if single_mask.size == 0 or np.max(single_mask) == 0:
            self.contours = np.zeros((2,1,2),np.int32)
            self.area = 0
            self.centroid = [0,0]

        # Otherwise return largest contour
        else:
            contours, hierarchy = cv2.findContours(single_mask.astype(np.uint8),cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_NONE)
            self.contours = cv2h.selectLongestContour(contours) # Sometimes there are little dots in the segmentation?
            self.area = cv2.contourArea(self.contours)
            self.centroid = cv2h.contourCentroid(self.contours)
        
        return

    

    # Update this worm's position, e.g. using the flow track method
    #   Optional all_worms_mask - if provided, flow is not allowed into other worms
    #   search areas. MUST BE DILATED SAME SEARCH_TOL AS INPUT ARG
    def track(self,img_gray,all_worms_mask = np.ndarray((0,0)), search_tol=3):

        # Enforce grayscale
        img_gray = img_gray[:,:,0]

        # If all_worms_mask is not provided, use all zeros (assume no other worms)
        if all_worms_mask.shape[0] == 0: 
            all_worms_mask = np.ones(img_gray.shape,np.uint8)

        # If it is provided make a local copy
        else:
            all_worms_mask = all_worms_mask*1

        # Generate a mask and dilate it (must match all_worms_mask if in use)
        mask = np.zeros(img_gray.shape[0:2],np.uint8)
        cv2.drawContours(mask,[self.contours],0,(1,1,1),-1)
        mask = cv2.dilate(mask,np.ones((search_tol,search_tol),np.uint8))

        # MOdify all_worms_mask to NOT include the current worm
        # (includes only all other worms)
        all_worms_mask = all_worms_mask & ~mask

        # Exclude all other worms from the search area
        mask = mask * ~all_worms_mask

        # Generate list of all pixels values in the mask and sort
        L = img_gray[mask>0]
        L = np.sort(L)


        # We will find self.area brightest or darkest pixels in the current frame. 
        # TODO: Evaluate performance of segmenting and multiplying entire image for each worm...
        # TODO: What if lost worm?
        # TODO: SHrinking worms
        A = int(self.area)
        A2 = int(A*1.1)
        if A2 >= len(L)-1: # Case 1: If the area of the worm is larger than search use all pixels (shouldn't happen...)
            new_mask = mask

        elif self.light_mode == lm.BF: # Case 2 BF: Take darkest pixels within search mask
            thresh = L[A2]
            new_mask = img_gray>thresh * mask

        elif self.light_mode == lm.DF: # Case 3 DF: Take brightest pixels within search mask
            thresh = L[-A2]
            new_mask = (img_gray>thresh).astype(np.uint8) * mask

        # Update the worm data
        self.contourize(new_mask)
        self.area = A # Don't allow area to change during tracking


        return



    # Draw worm on an image, IN PLACE, MUST BE COLOR. Also returns it
    def draw(self,img_color):

        # Dynamically set thickness and geometry params
        thick = 1+int(img_color.shape[1]/2000) # e.g. 2 if size is >2000 or 3 if >3000 etc
        rad = 4+int(img_color.shape[1]/1000)

        # Draw image
        img_color = cv2.drawContours(img_color,[self.contours],0,(0,255,0),thick)
        img_color = cv2.circle(img_color,tuple(self.centroid),rad,(255,0,0),thickness=-1)
        return img_color

    # Return a representation of the worm object as a dict e.g. for saving to disk
    def toDict(self):
        D = {}
        D["contours"] = self.contours.tolist()
        D["centroid"] = tuple(self.centroid)
        D["classification_status"] = self.classification_status
        D["primary_confidence"] = self.primary_confidence
        D["secondary_confidence"] = self.secondary_confidence
        D["secondary_classification"] = self.secondary_classification
        return D

    # Populate worm object from a dict, e.g. which was saved to disk as JSON
    def fromDict(self, D):
        if "contours" in D:
            self.contours = np.array(D["contours"])
        if "centroid" in D:
            self.centroid = list(D["centroid"])
        if "classification_status" in D:
            self.classification_status = D["classification_status"]
        if "primary_confidence" in D:
            self.primary_confidence = D["primary_confidence"]
        if "secondary_confidence" in D:
            self.secondary_confidence = D["secondary_confidence"]
        if "secondary_classification" in D:
            self.secondary_classification = D["secondary_classification"]

