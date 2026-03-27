"""
    RCNN_Server and/or RCNN_Segment_Image.py

    Python-based server engine for segmenting worms using Mask-RCNN. 
    Supply full file names of network and image as command line arguments
    Output masks are saved to disk
    Load an image and segment it using PyTorch Mask R-CNN

    Resource for directly sending /receiving images via socket connection
    https://github.com/TheMarvelousWhale/Python-Socket-For-Image-Transfer/blob/master/OptimizedServer.py


    Modified on 6/19/2021 to also run the gender finder net and return coordinates of heads,
    herm tails, and male tails

"""

# Hardcoded params
print_elapsed_time = False
MAX_LENGTH = 4500*3500*4 # up to 4 digits per
HOST = '127.0.0.1'

# Softcoded params
import sys
if len(sys.argv) >=4:
    fpth_worm = sys.argv[1]
    PORT = int(sys.argv[2])
    conf_thresh = float(sys.argv[3])

else:
    # Peter/John, please don't change these, just put your nets in the NeuralNets folder
    fpth_worm = r"C:\WormPicker\NeuralNets\model_2021-04-16_22-06-40_100epochs_15Done.pth"
    fpth_gender = r'C:\WormPicker\NeuralNets\model_2021-06-20_15-10-50_resnet50_4classes_9epochs_7Done_HTGender.pth'
    PORT = 10000
    conf_thresh= 0.7

# Global variables
buf = ""

# Imports
import time
import socket
from threading import Thread
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from RCNN_Helper import loadRCNN
import cv2
import numpy as np
import numpy.matlib
import os


def printElapsedTime(start,label,print_elapsed_time):
    end = time.time()
    print("-- server processing - ", label, " took ", round(end-start,3), "seconds")
    return end

def decodeImageFromSocket(buf):

    # Split comma separated values
    buf = str(buf)
    if buf[0] == 'b':
        buf = buf[2:] # remove b and ' at beginning after conversion  to string
    if buf[-1] == "'": # Remove ' from end
        buf = buf[:-1]

    vals = buf.split(",")

    pixel_start_idx = int(vals[0]) # Starting index in the list of encoded string integers



    # Preallocate image
    R = int(vals[1])
    C = int(vals[2])
    img = np.zeros((R,C),np.uint8)

    # Enforce that the received image size matches the header
    N_pix_actual = len(vals)-pixel_start_idx
    N_pix_requested=R*C

    # Enforce that values were split
    if len(vals) < 10:
        print("DECODE ERROR: Only received %d values" % (N_pix_actual))
        return None





    if N_pix_actual < N_pix_requested:
        print("DECODE ERROR: Expected %d x %d = %d pixels - received %d pixels" % (R,C,N_pix_requested,N_pix_actual))
        return None

    ## TODO: add decoding of gender information


    # Populate pixel values
    for r in range(R):
        for c in range(C):
            img[r,c] = int(vals[pixel_start_idx])
            pixel_start_idx+=1

    # Convert the grayscale image to "color" which is required by CNN
    img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    return img_color


def encodeImageForSocket(img, worm_gender_results = None):
   
    gender_results = ""
    if not worm_gender_results:
        first_pixel_idx = 3
    else:
        head_tail_count = 0
        #Encode the gender information as follows:
        #   For each head and tail we have a set of 5 values consisting of Worm number, cX, cY, head/tail type, conf score.
        for i, worm in worm_gender_results.items():
            if worm['head-index'] != -1:
                head_tail_count += 1
                gender_items = [i, worm['head-centroid'][0], worm['head-centroid'][1], 1, round(worm['head-score'].item(),4)]
                gender_results += ",".join(map(str,gender_items)) + ','
            if worm['tail-index'] != -1:
                head_tail_count += 1 
                gender_items = [i, worm['tail-centroid'][0], worm['tail-centroid'][1], worm['tail-type'].item(), round(worm['tail-score'].item(),4)]
                gender_results += ",".join(map(str,gender_items)) + ','

        first_pixel_idx = 3 + 5 * head_tail_count

    #Encode rows and columns first
    encoded_image = f'{first_pixel_idx},'
    encoded_image+= str(img.shape[0]) + ","
    encoded_image+= str(img.shape[1]) + ","
    encoded_image+= gender_results

    #print(f'pre pixel encoded string: {encoded_image}')

    # Populate the rest of the pixel values into the str
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            encoded_image += str(int(img[r,c])) + ","

    # Add newline print_elapsed_time to trigger end for boost asio
    encoded_image+="\n"

    return encoded_image

# Main server function
def handle(clientsocket):
    while 1:

        # Get data
        start = time.time()
        buf = clientsocket.recv(MAX_LENGTH)

        
        if buf == b'': return #client terminated connection
        if "terminate" in str(buf.lower()): return # client asked server to terminate
        print("Buffer received: ", len(buf), "\tmax: ", MAX_LENGTH)

        print('-------------- Connection received: Decoding image for inference --------------')

        # Decode image
        image = decodeImageFromSocket(buf)

        # Failed to decode: Abort prediction and send back empty masks
        if image is None:
            clientsocket.send("DECODE_ERROR".encode())
            return

        #cv2.imshow("DEBUG IMAGE",image)
        #cv2.waitKey(0)

        # Load image
        start = printElapsedTime(start,"Reading and decoding image" ,print_elapsed_time)

        pil2tensor = torchvision.transforms.ToTensor()
        image_tensor = pil2tensor(image)
        if use_cuda:
            image_tensor = image_tensor.cuda()

        start = printElapsedTime(start,"Image Conversion" ,print_elapsed_time)

        ## PERFORM WORM IDENTIFICATION
        # Perform inference
        model_worm.eval() # For inference
        with torch.no_grad():
            predictions_worms = model_worm([image_tensor])           # Returns prediction

        start = printElapsedTime(start,"Worm Inference" ,print_elapsed_time)    

        # Extract masks from predictions_worms
        predictions_worms = predictions_worms[0] # Always 1 element if 1 element list of images is supplied
        #print(f'worm predictions: {predictions_worms}')
        masks_worms = predictions_worms['masks'].permute([2,3,1,0]).cpu().numpy()
        N_masks_worms = masks_worms.shape[3]

        # Consolidate masks that are actually for the same worm
        ol_thresh = 0.75
        eliminated_idx = []
        for n in range(N_masks_worms):
            if n in eliminated_idx: continue
            this_mask = np.mean(masks_worms[:,:,:,n],axis=2)>conf_thresh
            #print(f'mean this mask: \n{this_mask}')
            A1 = np.sum(this_mask)
            #print(f'sum this mask: \n{A1}')

            # Compare it with all other masks
            for m in range(N_masks_worms):

                # Skip self comparisons
                if n==m: continue

                # Skip masks already eliminated
                if m in eliminated_idx: continue

                # Check how much the smaller mask overlaps with the larger mask
                test_mask = np.mean(masks_worms[:,:,:,m],axis=2)>conf_thresh
                A2 = np.sum(test_mask)
                A_min = float(min((A1,A2)))
                #print(f'n: {n}, m: {m}, a1: {A1}, a2: {A2}')
                ol_mask = this_mask & test_mask
                A_ol = float(np.sum(ol_mask))

                # If overlaps by 75% or more, merge the two as union
                if A_ol/A_min > ol_thresh:
                    new_mask = this_mask | test_mask
                    new_mask = np.stack((new_mask,)*3,axis=-1).shape
                    masks_worms[:,:,:,n] # Replace nth mask
                    masks_worms[:,:,:,m] = np.zeros(masks_worms[:,:,:,n].shape,np.uint8)
                                    # Set mth mask to empty
                    eliminated_idx.append(m) # Mark mth mask as eliminated
                #print(f'eliminated: {eliminated_idx}')

        ## PERFORM GENDER IDENTIFICATION
        # Perform inference
        model_gender.eval() # For inference
        with torch.no_grad():
            predictions_gender = model_gender([image_tensor])           # Returns prediction

        #print(f"gender predictions info: {predictions_gender}")

        start = printElapsedTime(start,"Gender Inference" ,print_elapsed_time)    
        #print(f'gender predictions: {predictions_gender}')
        # Extract masks from predictions_worms
        predictions_gender = predictions_gender[0] # Always 1 element if 1 element list of images is supplied
        masks_gender = predictions_gender['masks'].permute([2,3,1,0]).cpu().numpy()

        #print('initial masks gender')
        #for x in masks_gender:
        #        print(np.amax(x))
        N_masks_gender = masks_gender.shape[3]

        #print(f'gender predictions: \n{predictions_gender}')
        #print(f'num gender masks: {N_masks_gender}')

        worm_gender_results = {}

        overlap_threshold = 0.75
        ## Now for each worm mask we must look through the head and tail masks and find the masks that overlap with the worm, and take the head and tail with the highest overlap.
        for n in range (N_masks_worms):
            worm_gender_results[n] = {'head-score':0, 'head-index':-1, 'tail-score':0, 'tail-index':-1, 'tail-type': -1}
            if n in eliminated_idx:
                continue

            this_worm = np.mean(masks_worms[:,:,:,n],axis=2)>conf_thresh
            #print(f'mean this mask: \n{this_mask}')
            #A_worm = np.sum(this_worm)

            for m in range(N_masks_gender):
                if predictions_gender['scores'][m] < 0.4: continue # Ignore gender masks with very low confidence 
                gender_mask = np.mean(masks_gender[:,:,:,m],axis=2)>conf_thresh
                A_gender = np.sum(gender_mask)
                ol_mask = this_worm & gender_mask
                A_ol = float(np.sum(ol_mask))

                if A_ol/A_gender >= overlap_threshold:
                    if predictions_gender['labels'][m] == 1: # Head
                        if predictions_gender['scores'][m] > worm_gender_results[n]['head-score']:
                            worm_gender_results[n]['head-score'] = predictions_gender['scores'][m]
                            worm_gender_results[n]['head-index'] = m
                    elif predictions_gender['labels'][m] == 2 or predictions_gender['labels'][m] == 3: # Tail
                        if predictions_gender['scores'][m] > worm_gender_results[n]['tail-score']:
                            worm_gender_results[n]['tail-score'] = predictions_gender['scores'][m]
                            worm_gender_results[n]['tail-index'] = m
                            worm_gender_results[n]['tail-type'] = predictions_gender['labels'][m]

        #print(f'Gender results: \n{worm_gender_results}')

        for i in worm_gender_results:
            if worm_gender_results[i]['head-index'] != -1:

                curr_mask = masks_gender[:,:,0,worm_gender_results[i]['head-index']]
                #print(f'pre conv shape: {curr_mask.shape}')
                #curr_mask = cv2.cvtColor(curr_mask,cv2.COLOR_RGB2GRAY)
                #print(f'post conv shape: {curr_mask.shape}')
                curr_mask = curr_mask > conf_thresh


                curr_mask = np.array(curr_mask)
                #print(f'max: {np.amax(curr_mask)}')
                #print(curr_mask)
                y_ind, x_ind = np.where(curr_mask)
                #print(f'mask indices \n{mask_indices}')
                #print(f'mask indices: {len(mask_indices)}')

                cX = int(np.sum(x_ind)/len(x_ind))
                cY = int(np.sum(y_ind)/len(y_ind))

                centroid = (cX, cY)
                #print(f'centroid: {centroid}')

                ## convert image to grayscale image
                #gray_image = cv2.cvtColor(masks_gender[worm_gender_results[i]['head-index']], cv2.COLOR_RGB2GRAY)
                ## convert the grayscale image to binary image
                #ret,thresh = cv2.threshold(gray_image,127,255,0)
                ## calculate moments of binary image
                #M = cv2.moments(thresh)
                
                ## calculate x,y coordinate of center
                #cX = int(M["m10"] / M["m00"])
                #cY = int(M["m01"] / M["m00"])



                #head_mask = np.mean(masks_gender[worm_gender_results[i]['head-index']],axis=2)>conf_thresh
                #head_mask = np.array(head_mask)
                #head_mask = head_mask.astype(np.float32)
                ##print(head_mask)
                ##print(head_mask.shape)
                ##head_mask = np.array(head_mask)
                ## calculate moments of binary image
                #M = cv2.moments(head_mask)
                ## calculate x,y coordinate of center
                #cX = int(M["m10"] / M["m00"])
                #cY = int(M["m01"] / M["m00"])
                worm_gender_results[i]['head-centroid'] = centroid

                #cv2.circle(image, centroid, 5,(255, 255, 255), -1)
                #cv2.imshow("centroid", image)
                #cv2.waitKey(0)
            
            if worm_gender_results[i]['tail-index'] != -1:
                curr_mask = masks_gender[:,:,0,worm_gender_results[i]['tail-index']]
                #print(f'pre conv shape: {curr_mask.shape}')
                #curr_mask = cv2.cvtColor(curr_mask,cv2.COLOR_RGB2GRAY)
                #print(f'post conv shape: {curr_mask.shape}')
                curr_mask = curr_mask > conf_thresh


                curr_mask = np.array(curr_mask)
                #print(f'max: {np.amax(curr_mask)}')
                #print(curr_mask)
                y_ind, x_ind = np.where(curr_mask)
                #print(f'mask indices \n{mask_indices}')
                #print(f'mask indices: {len(mask_indices)}')

                cX = int(np.sum(x_ind)/len(x_ind))
                cY = int(np.sum(y_ind)/len(y_ind))

                centroid = (cX, cY)
                #print(f'centroid: {centroid}')
                worm_gender_results[i]['tail-centroid'] = centroid

                #cv2.circle(image, centroid, 5,(255, 255, 255), -1)
                #cv2.imshow("centroid", image)
                #cv2.waitKey(0)


        #print(f'Gender results: \n{worm_gender_results}')



        ########

        # Add masks_worms to a single image for returning to client
        all_masks_worms = np.zeros(image.shape,np.uint8) # Preallocate allmasks

        for n in range(N_masks_worms):

            # Extract current mask
            this_mask = masks_worms[:,:,:,n] # Get the n'th prediciton mask

            # If not, add it as a new mask with a unique ID
            this_mask = (this_mask>conf_thresh).astype(np.uint8)*(n+1)
            #cv2.imshow("this masks", this_mask)
            #cv2.waitKey(0)
            all_masks_worms += this_mask

        start = printElapsedTime(start,"Mask extraction" ,print_elapsed_time)    


        # Send grayscale mask image back to C++

        #cv2.imshow("worm masks", all_masks_worms)
        #cv2.waitKey(0)
        encoded_image = encodeImageForSocket(all_masks_worms[:,:,0], worm_gender_results)
        clientsocket.send(encoded_image.encode())
        start = printElapsedTime(start,"Sending image back to C++" ,print_elapsed_time)    

        
if __name__ == "__main__":
    # Auto-select GPU or CPU
    use_cuda = torch.cuda.is_available() # Automatically decide whether to use CUDA
    model_worm = loadRCNN(fpth_worm,2,use_cuda)
    model_gender = loadRCNN(fpth_gender, 4, use_cuda)

    # Setup the server socket to listen
    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((HOST, PORT))
    serversocket.listen(10)


    print("Server set up and listening: CUDA = ",use_cuda)
    # MAIN LOOP: SERVER LISTENS FOR IMAGE INPUT TO ANALYZE
    while 1:

        # It "terminate" was sent to the buffer at any point, terminate
        if "terminate" in buf.lower():
            clientsocket.send('closing server'.encode())
            time.sleep(1.5)
            exit(0)

        #accept connections from outside
        (clientsocket, address) = serversocket.accept()
        time.sleep(0.15) # This little pause seems to prevent partial-buffer transfer errors.

        ct = Thread(target=handle, args=(clientsocket,))
        ct.start()











pass


