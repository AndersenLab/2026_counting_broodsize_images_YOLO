# Plate.py
# Anthony D. Fouad, Ph.D.
# Classes for segmenting a multi-well plate (or similar format ) and applying various analysis to wells individually
#
# 

import cv2
import numpy as np
import os
from OpenCV_Helpers import *

# Search for RCNN_Segment (different directories for repo vs deployment environments)
import sys
sys.path.insert(1, os.path.realpath('..\ActivityAnalyzer\ActivityAnalyzer\mask_rcnn')) # Autogans
sys.path.insert(1, os.path.realpath('..\WWAnalyzerPy\mask_rcnn')) #Deployed systems (WWUpdater.nsi)
from RCNN_Segment import RCNN_Segment
from RCNN_Helper import loadRCNN
from scipy.optimize import minimize

WINDOW_NAME = "Preview"

# Create 
class Well:
    def __init__(self,img_gray,circle, well_num, lawn_params:dict):

        # Store a crop of the segmentation image (probably darkfield?) that was used to generate this well
        circle = circle.astype(np.int16) # Theoretically possible to get negatives. They will be cropped later.
        self.global_circle = circle*1
        self.radius = circle[2]*1
        self.local_circle = circle*1 
        self.local_circle[0] = self.radius
        self.local_circle[1] = self.radius
        self.img = self.cropWell(img_gray)
        self.well_num = well_num

        # Attempt to segment bacterial lawn in the well image
        self.lawn_params = lawn_params
        self.segmentLawn()


    # Segment a bacterial lawn from the darkfield image
    def segmentLawn(self):
        
        if not self.lawn_params["enable"]:
            self.global_lawn_circle = np.array([0,0,0])
            self.lawn_radius = 0
            self.lawn_circle = np.array([0,0,0])
            return

        # Generate local masked image that excludes the edges
        img = self.img*1
        img = cv2.GaussianBlur(img,(99,99),55)
        well_mask = np.zeros(img.shape,np.uint8)
        well_mask = cv2.circle(well_mask,(self.radius,self.radius),self.radius,(1,1,1),-1)
        img = img * well_mask.astype(np.float32) # set edges to black.

        # Filter it with a circular kernel
        R = int(self.lawn_params["relative size"]*self.radius)
        D = 2*R
        ker = np.zeros((D,D),np.uint8)
        ker = cv2.circle(ker,(R,R),R,(255,255,255),-1)
        img_filt = cv2.filter2D(img,cv2.CV_32F,ker)
        img_filt = img_filt/np.max(img_filt) # normalize 2 1

        # Blur the filtered image again. This has the side effect of de-weighting edges
        img_filt = cv2.GaussianBlur(img_filt,(55,55),49)

        # Find the maxima and store it 
        y,x = np.where(img_filt == np.max(img_filt))
        self.lawn_circle = self.local_circle*1
        self.lawn_circle = np.array([x[0],y[0],R],np.int32)
        self.lawn_radius = R

        # Store the local lawn circle as well
        self.global_lawn_circle = self.lawn_circle*1
        self.global_lawn_circle[0] += self.global_circle[0] - self.global_circle[2]
        self.global_lawn_circle[1] += self.global_circle[1] - self.global_circle[2]
        
        # DEBUG
        # cv2.imshow(WINDOW_NAME,img_filt)
        # cv2.waitKey(0)

    def toDict(self):
        
        return


    # DRAW FUNCTIONS 
    # - If an image is NOT supplied, uses the well image.
    # - If an image IS supplied, draws the well structures on the supplied image - XY coordinate of the well center stored in well.circle
    def drawWell(self, img = None, show_debug = False, main_color = (0,255,0)):
        
        # Get image if not supplied - handles single well images or full plate images
        if img is None: 
            img = self.img*1
            circle = self.local_circle
            lawn_circle = self.lawn_circle
        else:
            circle = self.global_circle
            lawn_circle = self.global_lawn_circle
            # lawn_circle[0] += circle[0] - circle[2]
            # lawn_circle[1] += circle[1] - circle[2]

        # Assign secondary draw colors
        if main_color == (0,255,0): 
            lawn_color = (0,155,0)
            dot_color = (0,155,255)
        else: 
            lawn_color = main_color
            dot_color = main_color


        # Get color image
        if len(img.shape) < 3: img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        else: img_color = img*1

        # Draw local well outline
        cv2.circle(img_color, (circle[0], circle[1]), circle[2], main_color, 4)

        # Print text label
        r = int(0.9*circle[2])
        cv2.putText(img_color,"%d" % self.well_num, (circle[0]-r, circle[1]-r), cv2.FONT_HERSHEY_SIMPLEX,2.2,main_color,3)

        # Draw lawn outline
        cv2.circle(img_color, tuple(lawn_circle[:2]),lawn_circle[2],lawn_color,2)
        cv2.circle(img_color, tuple(lawn_circle[:2]),int(lawn_circle[2]*0.05),dot_color,-1)

        # DEBUG - WEL VIEW
        if show_debug:
            cv2.imshow(WINDOW_NAME,img_color)
            cv2.waitKey(0)

        return img_color


    # Crop an arbitrary image according to this well's position on the plate
    def cropWell(self, img):

        x0 = self.global_circle[0] - self.global_circle[2]
        x1 = self.global_circle[0] + self.global_circle[2]
        y0 = self.global_circle[1] - self.global_circle[2]
        y1 = self.global_circle[1] + self.global_circle[2]

        x0 = max((0,x0))
        x1 = min((img.shape[1],x1))
        y0 = max((0,y0))
        y1 = min((img.shape[0], y1))
        if len(img.shape) == 2:
            img_crop = img[y0:y1,x0:x1]*1 # deep copy
        else:
            img_crop = img[y0:y1,x0:x1,:]*1 # deep copy
        return img_crop



class Plate:

    def __init__(self, params, construct_method = "from segmentation", use_file_idx = None):
        if construct_method == "from segmentation": return self.constructFromSegmentation(params, use_file_idx)
        elif construct_method == "from data dump": return self. constructFromDataDump(params)
        else: raise Exception("invalid construct method for Plate")
        pass

    def constructFromDataDump(self, params:dict):
        self.params = {"N wells": len(params)}
        self.wells = []
        for i,key in enumerate(params): 
            if not "Well" in key: continue
            self.wells.append(Well(self.img_gray,np.array(params[key]['well circle']),i+1))
        pass

    def constructFromSegmentation(self,params:dict,use_file_idx:int):

        # If constructing from a new segmentable image, must specify the file idx
        if use_file_idx is None:
            raise Exception("Cannot construct from segmentation without supply idx of file to use!")

        # Load a darkfield image for analysis
        self.params = params
        fnames = os.listdir(params["paths"]["dark field"])
        fnames = [i for i in fnames if "png" in i]
        if len(fnames) < 1: raise Exception("No PNG images found in the dark field path: %s" % params["paths"]["dark field"])
        img  = cv2.imread(os.path.join(params["paths"]["dark field"], fnames[use_file_idx]))

        # Store a color image (for display) and a gray image (for analysis)
        self.img_color = img
        if len(self.img_color.shape) < 3:
            self.img_color = cv2.cvtColor(self.img_color, cv2.COLOR_GRAY2BGR)
        self.img_gray = cv2.cvtColor(self.img_color,cv2.COLOR_BGR2GRAY)

        # Find circles for the arenas. Param1 = canny upper thresh. Param2 = sensitivity (lower = more sens)
        if True: #params["circle method"] == "hough":
            params["Hough circles"]["N wells"] = params["N wells"] # Hough  needs access to N wells
            self.circles = constrainedHoughCircles(self.img_gray,params["Hough circles"])
        else:
            raise Exception("Invalid circle method: ", params["circle method"])

        if params["enforce well count"] and self.circles.shape[0] != params["N wells"]:
            raise Exception("Found %d wells (expected %d)" % (self.circles.shape[0],params["N wells"]))

        # Sort the circles by row/col (roughly), starting with top left corner
        self.circles = sortCircles(self.circles)

        # Generate well objects for each circle
        self.wells = []
        for i,circle in enumerate(self.circles): 
            self.wells.append(Well(self.img_gray,circle,i+1,params["lawn"]))
            #self.wells[-1].drawWell(show_debug = True)

        # Within each arena, attempt to segment a bacteria lawn in case this is needed



    def drawWells(self, wait_ms = 1000):

        img_color = self.img_color*1

        main_color = (0,255,0) if len(self.wells) == self.params["N wells"] else (0,0,255)

        for well in self.wells:
            img_color = well.drawWell(img_color,main_color = main_color)

        # Shrink image to fit screens if needed
        if img_color.shape[0] > 900: img_color = cv2.resize(img_color,(0,0),fx=0.33,fy=0.33)

        # Text label
        img_color = cv2.putText(img_color,"press any key to continue",(10,45),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0,255,0),2)

        # Display
        cv2.imshow(WINDOW_NAME, img_color)
        k = cv2.waitKey(wait_ms)
        return k, img_color

    def toDict(self):

        wells_dicts = []
        for well in self.wells:
            wells_dicts.append(well.toDict())
        return wells_dicts

    # Static func to draw a well from a dictionary e.g. which has been saved to disk
    def drawFromDict(img_color: np.ndarray, D:dict, local = False,main_color = (0,255,0)):
    
        # Draw well outlines
        for key in D:
            if not "Well" in key: continue

            # Select local or global coordinates
            if local:
                circle = D[key]["local well circle"]
                lawn_circle = D[key]["local lawn circle"]
            else:
                circle = D[key]["well circle"]
                lawn_circle = D[key]["lawn circle"]
            lawn_color = (0,155,0)
            dot_color = (0,155,255)
            well_num = key.replace("Well ","")

            # Draw local well outline
            cv2.circle(img_color, (circle[0], circle[1]), circle[2], main_color, 2)

            # Print text label
            r = int(0.9*circle[2])
            cv2.putText(img_color,"%s" % well_num, (circle[0]-r, circle[1]-r), cv2.FONT_HERSHEY_SIMPLEX,2.2,main_color,3)

            # Draw lawn outline
            cv2.circle(img_color, tuple(lawn_circle[:2]),lawn_circle[2],lawn_color,2)
            cv2.circle(img_color, tuple(lawn_circle[:2]),int(lawn_circle[2]*0.05),dot_color,-1)

        return img_color






# Test script
if __name__ == "__main__":
    
    # Parameters
    params = loadJSON

    # Generate plate info
    P = Plate(params["plate"])






