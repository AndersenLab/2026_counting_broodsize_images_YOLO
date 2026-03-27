import numpy as np
import os
import cv2
from verifyDirlist import *

path = r'F:\Matt eLIfe paper data (merged)\Processed Data\Train\Alive'
Img_path = verifyDirlist(path,False,'npy')[0]
totalcnt = 0
whitecnt = 0
for i in Img_path:
    
    example = np.load(i)
    for frame in range(3):
        totalcnt = totalcnt +1
        frame_mean = np.mean(example[frame,:,:])
        if frame_mean > 250:            # High intensity, i.e. white img
            whitecnt = whitecnt + 1
            print(i)
    
print("total frame is ",totalcnt)
print("white frame is ",whitecnt)
example_path = r'F:\Matt eLIfe paper data (merged)\Processed Data\Test\Alive\7035.npy'
exp = np.load(example_path)
example2 = np.zeros((85,85,3),dtype = np.uint8)

"""
for i in range(3):
    example2[:,:,i] = exp[i,:,:]
    print(exp[i,:,:])

#cv2.imshow("Debug",example2)
#cv2.waitKey()

print(example.shape)
"""