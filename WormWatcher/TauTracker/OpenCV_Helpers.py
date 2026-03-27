import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from scipy.optimize import minimize


# Show a matplotlib image OR list of images for debugging (up to 4)
def showDebugImage(images:list):

    # Force list
    if type(images) != list:  images = [images]

    # Show debug with as many subplots as needed up to 4
    plt.figure(1)

    for i,img in enumerate(images):

        if len(images) > 1: plt.subplot(2,2,i+1)
        plt.imshow(img)
        plt.colorbar

    plt.show(block = True)


def applyTargetMean(img, target_mean):
    if target_mean is None: return img
    img = img.astype(np.float64) * target_mean / np.mean(img)
    img[img > 255] = 255
    img = img.astype(np.uint8)
    return img

def autoContrast(img, clip_pct = 1):

    # Automatic brightness and contrast optimization with optional histogram clipping
    if img.ndim ==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_pct *= (maximum/100.0)
    clip_pct /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_pct:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_pct):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return auto_result


# Pick and return the longest contour in contours, which is a list of lists
# If there are no contours (just a list?) then return empty
def selectLongestContour(contours):
    longest_contour = []
    for contour in contours:
        if len(contour) > len(longest_contour):
            longest_contour = contour

    return longest_contour


# Return the centroid [x,y] of the contour. This may differ from the center of 
# mass of the mask or the center of gravity of an image. It's a simple approximation
# of the worm centroid
#
# Contours must be a SINGLE np.ndarray of shape (N,1,2)  - a single worm contour.

def contourCentroid(contours):

    xbar = int(np.mean(contours[:,0,0]))
    ybar = int(np.mean(contours[:,0,1]))
    return xbar,ybar
    
# Normalize an image to uint8 format

def normalize2uint8(img:np.ndarray):
    img = img.astype(np.float32)
    img -= np.min(img)
    img = img / np.max(img)
    img = img*255
    img = img.astype(np.uint8)
    return img



import numpy as np


# Find the XY point in pt_list [Nx2 array] that is closest to Ref
def closestPoint(ref:np.ndarray,pt_list:np.ndarray):

    ref = ref.astype(np.float64)
    pt_list = pt_list.astype(np.float64)

    dist = np.sqrt((pt_list[:,0] - ref[0])**2 + (pt_list[:,1]-ref[1])**2)
    idx = np.where(dist == np.min(dist))
    closest = pt_list[idx,:]

    return idx,closest,np.min(dist)

def ptDist(pt0, pt1):

    if type(pt0) is not np.ndarray: pt0 = np.array(pt0)
    if type(pt1) is not np.ndarray: pt1 = np.array(pt1)

    pt0 = pt0.astype(np.float64)*1.0
    pt1 = pt1.astype(np.float64)*1.0 # deep copy

    dist = np.sqrt((pt1[0] - pt0[0])**2 + (pt1[1]-pt0[1])**2)
    return dist

def sortCircles(circles:np.ndarray):

    # Setup output array
    circles = circles *1 # deep copy
    new_circles = np.zeros(circles.shape,circles.dtype)
    ysearch = 0
    i=-1
    r = np.mean(circles[:,2])


    while i < circles.shape[0]-1:

        # Seed the row by finding the top left
        i+=1
        idx0,_,_ = closestPoint(np.array([0,ysearch]),circles[:,:2])
        idx0 = idx0[0] # Require to start with first one
        new_circles[i,:] = circles[idx0,:]*1
        circles[idx0,:] = 50000 # Remove from further consideration

        # Find all remaining circles in this row
        # First, try to find a new circle that is similar in level but to the right of the last
        xmin = new_circles[i,0]+1
        ymin = new_circles[i,1]-r
        ymax = new_circles[i,1]+r

        # Sort this row's points according to x pos
        idx_this_row = (circles[:,0] > xmin) & (circles[:,1] > ymin) & (circles[:,1] < ymax)
        this_circles = circles[idx_this_row,:]
        pos_this_row = np.where(idx_this_row)[0]
        pos_this_row = pos_this_row[this_circles[:,0].argsort()]

        for pos_this_row in pos_this_row:
            i+=1
            new_circles[i,:] = circles[pos_this_row,:]*1
            circles[pos_this_row,:] = 50000 # remove from further consideration

        ysearch += 2*r

    return new_circles

# Load image and convert to grayscale
def constrainedHoughCircles(image: np.ndarray,params):

    # Unpack parameters
    ideal_spacing = params["well spacing (fraction of diameter)"] * params["well diameter (pixels)"]
    dsfactor = params["dsfactor"]
    N_wells = params["N wells"]
    r = int(params["well diameter (pixels)"] / 2) * dsfactor
    param1 = params["canny thresh"]
    param2 = params["sensitivity"]

    # Downsample image for speed
    image = cv2.resize(image.copy(),(0,0),fx=dsfactor,fy=dsfactor)

    # Blur image
    img_blur = cv2.GaussianBlur(image,(params["blur"],params["blur"]),params["blur"])

    # Detect circles using HoughCircles on the blurred image
    print("Circle detection...")
    circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, minDist=int(1.9*r),
                            param1=param1, param2=param2, minRadius=int(0.9*r), maxRadius=int(1.1*r))

    if circles is None or (circles is not None and circles.shape[1] < N_wells):
        # If N_wells is 1, create a single circle encompassing the entire image
        if N_wells == 1:
            print("Circle detection failed, but using entire image as single well (only allowed for 1-well images)")
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            # Calculate radius to fully enclose the image (distance to farthest corner)
            # Use the smaller dimension to ensure the circle fits within the image
            radius = min(width, height) // 2
            # Create a single circle array in the expected format (1, 1, 3)
            circles = np.array([[[center_x, center_y, radius]]], dtype=np.float32)
        else:
            raise Exception("Could not find enough circles to segment wells!")

    circles = np.uint16(np.around(circles[0, :]))
    
    # Fundamental validation before optimization
    if len(circles) == 0:
        raise Exception("No circles detected - cannot proceed with optimization")
    
    if len(circles) < N_wells:
        print(f"Warning: Only found {len(circles)} circles but need {N_wells}. Using all available circles.")
        N_wells = len(circles)

    def error_function(selected_indices):
        # Check for NaN or infinite values in input
        if np.any(np.isnan(selected_indices)) or np.any(np.isinf(selected_indices)):
            # Return a gradient-friendly penalty that increases with distance from valid region
            return 1e6 + np.sum(np.abs(selected_indices))
        
        # Robust bounds checking to prevent invalid indices
        valid_indices = np.clip(selected_indices, 0, len(circles) - 1).astype(np.int32)
        
        # Additional safety check for invalid indices
        if np.any(valid_indices < 0) or np.any(valid_indices >= len(circles)):
            # Return a gradient-friendly penalty that increases with distance from valid region
            return 1e6 + np.sum(np.abs(selected_indices - np.clip(selected_indices, 0, len(circles) - 1)))
        
        selected_circles = circles[valid_indices]
        distances = np.sqrt(np.sum((selected_circles[:, None, :2] - selected_circles[None, :, :2])**2, axis=2))
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        error = np.sum((min_distances - ideal_spacing)**2)
        return error


    # Initial guess (first N_wells circles, ensuring they are within the range)
    print("Optimizing circle combos")
    initial_guess = np.clip(np.arange(N_wells), 0, len(circles) - 1)
    
    # Validate initial guess
    if len(initial_guess) == 0 or np.any(initial_guess < 0) or np.any(initial_guess >= len(circles)):
        print("Warning: Invalid initial guess, using first available circles")
        initial_guess = np.arange(min(N_wells, len(circles)))

    # Optimization
    if params["eliminate extra circles"]:
        result = minimize(error_function, initial_guess, method='L-BFGS-B', bounds=[(0, len(circles) - 1) for _ in range(N_wells)])
        best_circles_indices = np.round(result.x).astype(int)
        best_circles = circles[best_circles_indices]
    else:
        best_circles = circles

    # Undo downsampling from circles
    best_circles_ds = best_circles.copy()
    best_circles = (best_circles / dsfactor).astype(np.int32)

    return best_circles


def drawCircles(image,circles, thickness):
    # Draw the selected circles on the image
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for circle in circles:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv2.circle(output, center, radius, (0, 255, 0), thickness)

    return output        

# JSON handling
def loadJSON(fname):
    with open(fname,"r") as infile:
        return json.load(infile)

def saveJSON(fname, jdata, indent = 4):
    jdata_out = json.dumps(jdata, indent=indent)
    with open(fname,'w') as out_file: 
        out_file.write(jdata_out)


# Force file dialogs to front (script mode only)
def forceFileDialogsToFront():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()  # Hide the root window
    root.wm_attributes('-topmost', 1)  # Make it the topmost window temporarily










    