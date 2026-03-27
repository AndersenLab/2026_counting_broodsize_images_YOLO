# exportFramesForTraining.py
# Anthony D. Fouad, Ph.D.
#
# Exports images of individual wells in a WormTracker dataset so that they can be labeled by myvision.ai and used
# for AI training

from Plate import *
import cv2
import numpy as np
import copy


# collect data folders (containing images) which might be all in one folder or in timepoint subfolders
def datacrawler(params, N_select):

    pparent = params["export"]["parent"]
    subdirs = os.listdir(pparent)
    datasets = {}
    pout = params["export"]["pout"]

    if not os.path.exists(pout): os.makedirs(pout)

    # Save json data to disk
    saveJSON(os.path.join(pout,"exported_params.json"),params)    

    for p in subdirs:

        # Set up dictionary for datasets of various formats
        datasets[p] = {}
        datasets[p]["dir"] = os.path.join(pparent,p) # the main folder name of the dataset, could contain images or subfolders
        datasets[p]["all images"] = [] # all png images, regardless of which subfolder if any

        # Crawl through the subcontents to at most 1 more subdir level and add images
        contents = os.listdir(datasets[p]["dir"])

        for item in contents:

            # Image, add it directly. 
            if ".png" in item: datasets[p]["all images"].append(os.path.join(pparent,p,item))

            # Subfolder, crawl one more level and add images
            elif not os.path.isfile(os.path.join(pparent,p,item)):
                subcontents = os.listdir(os.path.join(pparent,p,item))
                for subitem in subcontents:
                    if ".png" in subitem: datasets[p]["all images"].append(os.path.join(pparent,p,item,subitem))

        if len(datasets[p]["all images"]) == 0:
            print("WARNING: No images found for dataset: ", p)
            del(datasets[p])
            continue

        # For each dataset select some images that can be used for chopping
        idx = np.linspace(0,len(datasets[p]["all images"])-1,N_select).astype(np.int64)
        datasets[p]["selected images"] = []
        datasets[p]["outnames"] = []
        for i in idx: 
            datasets[p]["selected images"].append(datasets[p]["all images"][i])

            # Store the output name too
            outname = copy.deepcopy(datasets[p]["all images"][i]) # strip the path info
            outname = outname.replace(os.path.join(pparent),"") 
            outname = outname.replace("\\","_") # remove slashes
            outname = outname.replace("/","_")
            outname = os.path.join(pout,outname) # replace out path

            datasets[p]["outnames"].append(outname)

        # Remove duplicates (can happen if not enough images)
        datasets[p]["selected images"] = np.unique(datasets[p]["selected images"]).tolist()
        datasets[p]["outnames"] = np.unique(datasets[p]["outnames"]).tolist()


    return datasets
    

# Test script
if __name__ == "__main__":
    
    # Plate semgentation parameters
    params = loadJSON("exportFramesForLabelingParams.json")

    # Export parameters
    target_mean = params["worm"]["target mean"] # set null/None to not do target mean

    # Before starting, preview all well segmentations to make sure they are correct
    
    datasets = datacrawler(params,params["export"]["frames per dataset"])
    for pn,key in enumerate(datasets):
        print("Previewing dataset %d of %d" % (pn,len(datasets)-1))
        pdata,_ = os.path.split(datasets[key]["selected images"][0])
        params["plate"]["paths"]["dark field"] = pdata
        P = Plate(params["plate"])
        if params["export"]["show well segmentation"]:
            P.drawWells(0)
        datasets[key]["plate"] = P



    # Chop up all images
    for n,setname in enumerate(datasets):

        # Generate plate info and display the well segmentation to ensure accuracy before continuing
        P = datasets[setname]["plate"]
        fnames = datasets[setname]["selected images"]


        # Get names of PNG images therein and subsample them. 
        print("Chopping dataset %d of %d" % (n,len(datasets)-1))
        for i, fname in enumerate(fnames):

            # Load image
            print('frame=',i)
            img = cv2.imread(fname,cv2.IMREAD_COLOR)

            # Split it up and save individual images
            # Crop the image of the well from the big image
            for j,well in enumerate(P.wells):
                img_crop = well.cropWell(img)

                if target_mean is not None:
                    img_crop = img_crop.astype(np.float64) * target_mean / np.mean(img_crop)
                    img_crop[img_crop > 255] = 255
                    img_crop = img_crop.astype(np.uint8)

                #img_crop = normalize2uint8(img_crop)
                if params["export"]["blur"]["enable"]:
                    ksize = params["export"]["blur"]["kernel size"]
                    sigma = params["export"]["blur"]["sigma"]
                    img_crop = cv2.GaussianBlur(img_crop,(ksize,ksize),sigma)

                # Split up the crop images
                pout, fout = os.path.split(datasets[setname]["outnames"][i])
                pout = os.path.join(pout,"well%0.2d" % j)
                if not os.path.exists(pout): os.makedirs(pout)
                cv2.imwrite(os.path.join(pout,fout), img_crop)
            pass

    


