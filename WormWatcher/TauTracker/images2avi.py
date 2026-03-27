# images2avi
#
# Convert all images within the source folder to a video

import cv2
import os

# Parameters
img_type = "png"
pdata = r'C:\Users\antho\Downloads\22-20220504T013458Z-001\96welltestimagesfrom4-28-22\2022-04-28 (16-09-01)'

# Find images
fnames = os.listdir(pdata)
fnames = [i for i in fnames if "png" in i]

# Create output video file
fout = os.path.join(pdata,"_video.avi")
img = cv2.imread(os.path.join(pdata,fnames[0]))
out = cv2.VideoWriter(fout,cv2.VideoWriter_fourcc('M','J','P','G'), 10, 
                        (img.shape[1],img.shape[0]))
                        
# out = cv2.VideoWriter()
# out.open(fout, -1, 25, (img.shape[1],img.shape[0]), True)

for i,fname in enumerate(fnames):

    # Load image
    img = cv2.imread(os.path.join(pdata,fnames[0]))
    out.write(img)

out.release()


