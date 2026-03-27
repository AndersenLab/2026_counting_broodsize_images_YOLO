"""
    Export Cropped ROI IMages, sorted according to whether they are contaminated (group 6) or not.
    Save images of each individual ROIs to use for user marking, training or testing.
    
    Adapted from zscriptSampleImages.py

"""
from verifyDirlist import *
from csvio import *
import os
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import cv2
import pandas as pd


# Parameters
pparent = r'F:\Matt eLIfe paper data (merged)'
labelparent = r'F:\Matt eLIfe paper data (merged)\_Analysis'
#pout = os.path.join(r'F:\Matt eLIfe paper data (merged)\WMseriescolor_trainimage',plate)
sessions_per_day = 2
days_to_sample = [21,25] #[4,10,20]    # only do last day for now
days_to_sample = np.asarray(days_to_sample)
outsize = 85
separate_train_test_dead_alive = True
crop_middle_flag = False

#create output folders for trianing and test data
opfolder = os.path.join(pparent,'Processed Data')
trainalive = os.path.join(opfolder,'train','alive')
traindead = os.path.join(opfolder,'train','dead')
testalive = os.path.join(opfolder,'test','alive')
testdead = os.path.join(opfolder,'test','dead')
for folder in [opfolder,trainalive,traindead,testalive,testdead]:
    if not os.path.exists(folder):
        os.makedirs(folder)

#labelsorig = dataorig[:,1].astype(np.str)
#dataorig = dataorig[:,2:].astype(np.float)

# Find the sessions
d_plates, plate_names = verifyDirlist(pparent,True,"","_Analysis")

# Going to keep a list of all fout images
foutlist = list()

cnt_alive = 0 #a counter for counting the total number of alive cases
cnt_dead = 0 #a counter for counting the total number of dead cases
for (pplate,name) in zip(d_plates,plate_names):

    # Find the Sessions for this plate
    d_sessions_full,d_sessions = verifyDirlist(pplate,True,"")

    # Skip plate if no excel file exist
    pAnalysis = os.path.join(pparent,'_Analysis',name,'ManualWMLifespan.csv')

    if not os.path.exists(pAnalysis):
        print("-----------------------------> Skipping (not enough sessions): ",name)
        continue
    


    # Load the excel file
    #create a list to temperately store all csv labels
    csvlist = []
    with open(pAnalysis, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        for row in datareader:
            csvlist.append(row)
        #remove the first and last row in csv which are 'manual' and 'end'
        csvlist.pop(0) 
        csvlist.pop(-1)
        labelarr = np.array(csvlist)
        #loop over different sessions
        for exp in range(np.shape(labelarr)[0]):
            #extract image paths under the current session folder
            img_folder = d_sessions_full[exp]
            #print(img_folder)
            d_img_full,d_img = verifyDirlist(img_folder,False,'png')
            if len(d_img_full) < 1:
                print("-----------------------------> Skipping (no images!!): ",d_sessions_full)
                continue
            # Get the ROI coordinates
            d_roi = verifyDirlist(img_folder,False,'ROI_Coord')[0]

            if len(d_roi) != 1:
                print("-----------------------------> Skipping (no ROI file!!): ",d_sessions_full)
                continue

            rois = readArrayFromCsv(d_roi[0],"<ROI Coordinates>")
            x = rois[0,:].astype(np.int)
            y = rois[1,:].astype(np.int)
            ri = rois[2,:].astype(np.int)
            ro = rois[3,:].astype(np.int)
            #load frames (three in total)
            N = len(d_img) #total number of frames inside this folder
            N_half = int(N/2)
            imglis = [] #a frame holder
            for frame_idx in [N_half+4, N_half+12, N_half+24]:
                fimg = d_img_full[frame_idx]
                imglis.append(cv2.imread(fimg,0))
            #then loop over different wells within one big image, -1 due to csv loading func loads one extra element
            for well in range(np.shape(labelarr)[1]-1):
                if (labelarr[exp,well] in ('nan','7.0')):
                    #skip since not labelled or corrupted
                    continue
                elif(labelarr[exp,well] == '1.0'):
                    #alive case, load corresponding images
                    cnt_alive = cnt_alive + 1
                    #Crop individual images
                    croplis = []
                    for img in imglis:
                        # Crop to ROI
                        crop_img = img[y[well]-ro[well]:y[well]+ro[well], x[well]-ro[well]: x[well]+ro[well]]

                        # Resize to standard small size
                        crop_img = cv2.resize(crop_img,(outsize,outsize))
                        croplis.append(crop_img)
                    # merge three frames into a 3d array
                    img_merged = np.array(croplis)#shape is 3,85,85
                    #then save the merged image into train or test folder
                    # Determine whether this plate will be training or testing
                    if cnt_alive%5 == 0:
                        subfolder = 'Test'
                    else:
                        subfolder = 'Train'
                    pout =  os.path.join(opfolder,subfolder,'Alive')
                    np.save(os.path.join(pout,str(cnt_alive)),img_merged)

                elif(labelarr[exp,well] in ('0.0','4,0','5.0')):
                    #dead case, not sure what '5' means, currently classified as dead
                    cnt_dead = cnt_dead + 1
                    #Crop individual images
                    croplis = []
                    for img in imglis:
                        # Crop to ROI
                        crop_img = img[y[well]-ro[well]:y[well]+ro[well], x[well]-ro[well]: x[well]+ro[well]]

                        # Resize to standard small size
                        crop_img = cv2.resize(crop_img,(outsize,outsize))
                        croplis.append(crop_img)
                    # merge three frames into a 3d array
                    img_merged = np.array(croplis)#shape is 3,85,85
                    #then save the merged image into train or test folder
                    # Determine whether this plate will be training or testing
                    if cnt_alive%5 == 0:
                        subfolder = 'Test'
                    else:
                        subfolder = 'Train'
                    pout =  os.path.join(opfolder,subfolder,'Dead')
                    np.save(os.path.join(pout,str(cnt_dead)),img_merged)

print("Program finished")