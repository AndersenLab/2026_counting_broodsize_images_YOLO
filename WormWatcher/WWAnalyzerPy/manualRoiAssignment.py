"""
    Manually specify ROIs on a plate, e.g. when it's not obvious or easy for auto recognition

"""
from verifyDirlist import verifyDirlist
import csvio
import cv2
import numpy as np
import os
from tkinter import filedialog
import tkinter as tk
from scipy.signal import medfilt2d, correlate2d



""" HELPER FUNCTIONS """


# Helper function mouse callback
global xclick, yclick, rtclick
xclick = 0;
yclick = 0;
rtclick = False;

def onMouse(event, x, y, flags, params):

    global xclick, yclick, rtclick

    if event == cv2.EVENT_LBUTTONDOWN:
        xclick = x
        yclick = y
    elif event == cv2.EVENT_RBUTTONDOWN:
        rtclick = True

    return

def drawAllcircles(img, coords, ri = 95, ro = 150):

    # Draw all circles on image
    if len(img.shape) == 2:
        img_display = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    else:
        img_display = np.copy(img)

    
    for coord in coords:
        img_display = cv2.circle(img_display,coord,ri,(0,255,0),2)
        img_display = cv2.circle(img_display,coord,ro,(0,255,0),2)

    return img_display

# Select an image to use
root = tk.Tk()
root.withdraw()
fimg = filedialog.askopenfilename(parent = root,title = "Select an image to open",filetypes = (("PNG images", "*.png"),("all files","*.*")))

# Load the image
img = cv2.imread(fimg)

# Click on first ROI
print("INSTRUCTIONS:\nClick ROI Centers, in order, down columns first\nRight click to delete last\nEsc to finish")

idx = 0;
coords = []
cv2.namedWindow("Preview",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Preview",onMouse)

while True:
    
    # Draw all circles
    img_display = drawAllcircles(img,coords)

    # Display image
    cv2.imshow("Preview",img_display)
    k = cv2.waitKey(50)

    # Ask user to click
    idx += 1
    print("Click on ROI #",idx);


    # Wait for click
    while xclick == 0 and yclick == 0 and rtclick == False and k != 27:
        k = cv2.waitKey(50)

    # If it's a right click, delete last element and decrement
    if rtclick and idx > 1:
        rtclick = False
        coords.pop()
        idx -= 2
        

    elif rtclick:
        None

    # If it's an esc keypress, exit
    elif k == 27 and len(coords) > 2:
        print("Annotation completed - saving coordinates")
        break

    elif k == 27:
        print("Quitting without saving anything")
        break

    else:
        coords.append((xclick,yclick))
        xclick = 0;
        yclick = 0;



if len(coords) > 2:
    # Calculate radius 
    dist = ((coords[1][0] - coords[0][0])**2 + (coords[1][1] - coords[0][1])**2)**0.5
    r_total = dist / 2
    ro = int(r_total * 0.9)
    ri = int(ro * 0.8)

    # Draw final circles

    img_display = drawAllcircles(img,coords,ri,ro)

    # Display image
    cv2.imshow("Preview",img_display)
    k = cv2.waitKey(50)
    
    # Save the data
    pout,__ = os.path.split(fimg)
    pout,__ = os.path.split(pout)
    fout = os.path.join(pout,"ROI manual coordinates.csv")

    # Print CSV file
    with open(fout, "w") as outfile:
        for coord in coords:
            outfile.write(str(coord[0]) + "," + str(coord[1]) + "," + str(ri) + "," + str(ro) + "\n")

    print("Coordinates saved as:\n",fout)
    print("To use these in a STEP 1 activity calculation, leave the file there.")
    

