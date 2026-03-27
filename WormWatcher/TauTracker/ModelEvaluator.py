# ModelEvaluator.py
# Anthony D. Fouad, Ph.D.
#
# Test script for evaluating a Mask-RCNN model on images from labeled datasets. Not for use on raw images of 
# whole plates, only for use on images in the same format as the training images

import os

import cv2
from cv2 import EVENT_LBUTTONDBLCLK
from Plate import *
from MultiWellAnalysis import *
from numpy import random
 

# ------------------------ Files ---------------------- #
ptest = r"H:\Tau worm mask RCNN\Model 3a - worms aggregation rev 1\Data for refinement\original"
i0 = 0


# ------------------------ RCNN ---------------------- #

seg_params =  {
        "model path": "C:\\WormWatcher\\NeuralNets\\model_2023-10-07_03-45-06_resnet50_2classes_10epochs_10Done_AggWorms3a.pth",
        "object conf thresh": None, # not used
        "pix conf thresh": 0.22, # this needs to be up at 0.45 to keep the blobs small, but if it's that high it neglects to keep a lot of clustered worms. 
        "overlap thresh": 0.55,
        "clump thresh": 0.001,
        "min object area": 80,
        "target mean": None,
        "show timing": False,
        "show worm segmentation images": False
    }
seg_params["use cuda"] = torch.cuda.is_available()

def process_manual_mask(manual_mask):
    """
    Process a 3-channel RGB manual mask to produce binary masks for each worm.
    
    Args:
    - manual_mask (numpy.ndarray): The manual mask image with 3 channels.
    
    Returns:
    - binary_masks (numpy.ndarray): A stack of binary masks where each worm is represented by a different mask with shape [R, C, 1, N].
    """
    # Identify unique colors in the image
    h, w, _ = manual_mask.shape
    flat_image = manual_mask.reshape(-1, 3)
    unique_colors = np.unique(flat_image, axis=0)
    
    # Filter out the black color [0, 0, 0]
    unique_colors = unique_colors[np.sum(unique_colors, axis=1) > 0]
    
    # For each unique color, create a binary mask
    masks = []
    for color in unique_colors:
        binary_mask = np.all(manual_mask == color, axis=2).astype(np.uint8) * 255
        binary_mask = binary_mask[:, :, np.newaxis]  # Add an additional dimension
        masks.append(binary_mask)
    
    return np.stack(masks, axis=-1)


# Composite image for two segmentations, one in red, one in green
def compositeImageTwoSegmentations(img,seg1,seg2):

    # First mask in red
    mask = copy.deepcopy(seg1)
    mask[:,:,:2] = 0

    # Second mask in green
    mask[:,:,1] = seg2[:,:,0]

    # combine images
    fused = cv2.addWeighted(img,0.6,mask,0.4,1.0)
    return fused

# Composite image for the RCNN worms, where each worm is shown in a random color to aid in
# identifying discrete worms
def compositeImageIndivWorms(img, masks):
    mask = np.zeros(img.shape, dtype=np.float64)
    half_width = img.shape[1] // 2

    for i in range(masks.shape[-1]):
        this_mask = masks[:, :, 0, i]
        color = (random.rand(3)*255).astype(np.uint8)
        shape = list(this_mask.shape) + [3]
        this_color_mask = np.zeros(shape, dtype=np.uint8)
        for clr in range(3):
            this_color_mask[:, :, clr] = color[clr] * this_mask
        
        # Apply the color only where the accumulated mask is still black
        if i < masks.shape[-1]//2:
            no_color_applied = np.all(mask == 0, axis=2)
            no_color_applied = no_color_applied[:,:half_width] # Crop mask to only the rcnn side
            this_color_mask[~no_color_applied] = 0  # Set regions with color to black
        
        if i < masks.shape[-1]//2:
            mask[:, :half_width] += this_color_mask
        else:
            mask[:, half_width:] += this_color_mask

    fused = cv2.addWeighted(img, 0.5, mask.astype(np.uint8), 0.5, 1.0)
    return fused



def getRcnnResults(model, img_crop, seg_params):
    img_crop = cv2.GaussianBlur(img_crop,(5,5),3)
    masks, masks_thresh, weighted, weighted_touch, touch_count = RCNN_Segment(model,img_crop,seg_params)

    N_touching = sum(touch_count>0)
    N_alone = sum(touch_count == 0)
    N_total = len(touch_count)

    return np.max(masks,3)*255, N_touching / N_total, masks_thresh

def getManualResults(mask):
    mask32 = mask.astype(np.int32)
    total_worms = 0
    touching_worms = 0
    for i in range(1,255):
        if np.sum(mask==i) == 0: continue
        total_worms += 1

        plt.imshow(mask32[:,:,2])
        plt.show()
        #for j in range(i+1,255):

        # TODO: increment touching worms if sum of this masks area with the other is higher than the mask value (touches ANY other object)
        mask==i
        
    pass

# Double click to zoom/unzoom
global zoom, zoom_coord, show_overlays
zoom = False
show_overlays = True
zoom_coord = (0,0)

def onClick(event,x,y,flags,param):
    global zoom, zoom_coord, show_overlays
    if event == cv2.EVENT_LBUTTONDBLCLK:
        zoom = not zoom
        zoom_coord = (x,y)

    elif event == cv2.EVENT_RBUTTONDOWN:
        show_overlays = not show_overlays


def showImageInteractive(img,masks):
    global zoom, zoom_coord, show_overlays
    #cv2.imwrite("C:\\WormWatcher\\Samples\\" + fimgs[i],fused)
    print("ESC to quick image view")
    k=-1
    cv2.namedWindow("Preview")
    cv2.setMouseCallback('Preview',onClick)

    while k != 27:

        # Apply overlays or not
        if show_overlays:
            fused = compositeImageIndivWorms(img,masks)
        else:
            fused = copy.deepcopy(img)

        # Zoom or not
        
        if zoom:

            # Safe coordinate extract
            x,y = zoom_coord
            dsfactor = 1.0
            x = int(x/dsfactor)
            y = int(y/dsfactor)
            hd2 = int(dsfactor*img.shape[0]/4) # HALF of height of 1/3 image size
            wd2 = int(dsfactor*img.shape[1]/4) # HALF of width of 1/3 image size

            x0 = max((0,x-wd2))
            y0 = max((0,y-hd2))
            x1 = x0+2*wd2
            y1 = y0+2*hd2

            # Crop img_color, but keep it the same size
            fused = fused[y0:y1, x0:x1, :]
            fused = cv2.resize(fused, (img.shape[1], img.shape[0]))

        # Add labels
        half_width = img.shape[1] // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        rcnn_position = (10, 30)
        manual_position = (half_width + 10, 30)        
        cv2.putText(fused, "RCNN", rcnn_position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(fused, "Manual", manual_position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Downsample a bit so it fits on screens
        fused_ds = cv2.resize(fused,(0,0),fx=0.75,fy=0.75)
        cv2.imshow("Preview",fused_ds)
        k = cv2.waitKey(100)

    cv2.destroyAllWindows()

# load model
model = loadRCNN(seg_params["model path"],2,seg_params["use cuda"])   

# Load images
pimg = os.path.join(ptest,"DatasetImages")
pmask = os.path.join(ptest,"DatasetMasks")
fimgs = [i for i in os.listdir(pimg)]
fmasks = [i for i in os.listdir(pmask)]
if len(fimgs) != len(fmasks):
    raise Exception()

for i in range(len(fimgs)):
    if i < i0: continue
    img = cv2.imread(os.path.join(pimg,fimgs[i]))
    manual_mask = cv2.imread(os.path.join(pmask,fmasks[i]))
    rcnn_mask, rcnn_touch_frac, masks = getRcnnResults(model,img,seg_params)
    
    #manual_touch_frac = getManualResults(manual_mask)
    print(f"{fimgs[i]} - RCNN touch frac = {rcnn_touch_frac:0.2f}")

    # Process the manual mask
    manual_masks = process_manual_mask(manual_mask)

    # Display RCNN masks and manual masks side-by-side
    combined_img = np.hstack([img, img])
    combined_masks = np.concatenate([masks, manual_masks], axis=-1)

    showImageInteractive(combined_img, combined_masks)



