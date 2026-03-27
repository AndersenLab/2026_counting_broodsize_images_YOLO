# WormTracker
#
# Class for Fining all worms, e.g. using mask RCNN or another method, then tracks them over time.
# Operates on Worm objects.

# Standard inputs
import numpy as np
import cv2
import os
import OpenCV_Helpers as cv2h
from enum import Enum
from Worm import *
from Plate import *
import torch


# Search for RCNN_Segment (different directories for repo vs deployment environments)
import sys
sys.path.insert(1, os.path.realpath('..\ActivityAnalyzer\ActivityAnalyzer\mask_rcnn')) # Autogans
sys.path.insert(1, os.path.realpath('..\WWAnalyzerPy\mask_rcnn')) #Deployed systems (WWUpdater.nsi)
from RCNN_Segment import RCNN_Segment
from RCNN_Helper import loadRCNN

class WormTracker:

    def __init__(self):
        self.worms = []

    # Separate out individual worms in a mask-RCNN output and contourize them
    # 
    # OPTION 1: Label encoding by pixel value
    #   In this format, individual worms have unique pixel numbers from 1 to 255 
    #   in the RED channel (2) in BGR format. postSegmentWorms clears 
    #   the list of worms and remakes it. Contrast with trackWorms, which
    #   attempts to update the positions of each worm, starting from
    #   a known worms list
    #
    # OPTION 2: One Mask per layer
    #   In this format, the input array has size [R,C,1,N], where R,C is the rows
    #   and columns size of the image, there is only one channel (grayscale) and 
    #   N is the number of worm masks. 
    #
    #   This option does not have flow protection for neighboring worms

    def postSegmentWorms(self,mask_rcnn_out):
        
        # Clear worm list
        self.worms = []

        # Route request depending on input option used
        if len(mask_rcnn_out.shape) == 3 and mask_rcnn_out.shape[2] == 3: 
            self.__postSegmentIndexed(mask_rcnn_out)
        elif len(mask_rcnn_out.shape) == 4: 
            self.__postSegmentLayered(mask_rcnn_out)
        else: 
            raise Exception("Invalid input mask_rcnn_out with shape: " + str(mask_rcnn_out.shape))


    # Private method for Option 1
    def __postSegmentIndexed(self,mask_rcnn_out):
        
        # Mask expected to be 3D BGR With red channel = worms. 
        mask = mask_rcnn_out[:,:,2] # Assume OpenCV BGR format

        # Find the minimum and maximum mask values
        min_worm = np.min(mask[mask>0])
        max_worm = np.max(mask[mask>0])

        # Generate worm objects from each, but don't allow flow track to flow into
        # other worms
        for i in range(min_worm, max_worm+1):
            this_mask = (mask==i) # This worm
            others_mask = (mask>0) & (~this_mask) # All other worms
            search_mask = this_mask & (~others_mask) # This worm excluding other worms
            self.worms.append(Worm(search_mask))


    # Private method for Option 2
    def __postSegmentLayered(self,mask_rcnn_out):
        for i in range(mask_rcnn_out.shape[3]):
            if np.sum(mask_rcnn_out[:,:,0,i]) == 0:
                #print("DELETE ME - skipping eliminated worm")
                continue
            self.worms.append(Worm(mask_rcnn_out[:,:,0,i]))


    # Draw all the worms on a color image. Modifies image in place.
    def drawWorms(self,img_color):
        for worm in self.worms:
            worm.draw(img_color)
        return

    # Track worm paths 
    def trackWorms(self,img_gray,search_tol = 3):

        # To handle worm collisions in the search, first generate a mask of
        # all other worms. Dilate it so this process doesn't need to be repeated
        # For each worm
        all_worms_mask = self.allWormsMask(img_gray.shape)
        all_worms_mask = cv2.dilate(all_worms_mask,
                                    np.ones((search_tol,search_tol),np.uint8))

        # Now loop through and update each worm
        for worm in self.worms:
            worm.track(img_gray,all_worms_mask,search_tol)

    # Draw all the worms 
    def allWormsMask(self,shape,multiplier = 1):

        all_worms_mask = np.zeros(shape[0:2], np.uint8)
        for i,worm in enumerate(self.worms):
            cv2.drawContours(all_worms_mask,[worm.contours],0,(1,1,1),-1)
            
        return all_worms_mask*multiplier

    # Convert info about all the worms in this wormtracker object to a list of dicts e.g. for saving
    def toDict(self):
        D = []
        for worm in self.worms:
            D.append(worm.toDict())
        return D

        


# Test script
if __name__ == "__main__":
    pass
