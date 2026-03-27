from verifyDirlist import *
from CNN_Helper import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import datetime
from csvio import *
from datetime import date
import numpy as np


def differ_days(date1, date2):
    a = date1
    b = date2
    return (a-b).days


pdata = r'F:\Matt eLIfe paper data (merged)\Processed Data'
pparent = r'F:\Matt eLIfe paper data (merged)'
pnet = os.path.join(pdata,'checkpoint')
presult = os.path.join(pdata, 'results')
load_net = os.path.join(pnet,'trainedmodel_85.pth')
use_loaded_net = True

# Parameters - training and testing data locations
    
ptrain = os.path.join(pdata,'train')
ptest = os.path.join(pdata,'test')
pvalid = ptest
classes = ('alive', 'dead') # Must match folder names

# Parameters - training details

img_size = 85
ker_size = 5
batch_size = 32;
use_cuda = False
learn_rate = 0.0005

normtype = 1 # 0 = none, 1= normalize21, 2= subtract mean

#load and initialize network
net = Net(img_size,ker_size)
if use_loaded_net:
        
    # Load the overall checkpoint
    checkpoint = torch.load(load_net,map_location=torch.device('cpu') )

    # Load the network state
    net.load_state_dict(checkpoint['model_state_dict'])
  

    # Load the optimizer state
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load the starting epoch
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']



""" STEP 0: VERIFY CUDA DEVICE """

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if use_cuda and torch.cuda.is_available() :
    device = torch.device('cuda:0')
    torch.set_default_tensor_type('torch.cuda.FloatTensor') # IF using cuda create all new tensors on cuda
    # Seems to avoid an error when load state dict loads to CPU for some reason




elif use_cuda:
    print('CUDA requested but CUDA Device not detected. Training Aborted...')
    exit(0)
else:
    device = torch.device('cpu')

print(device)

#define the worm class
class Worm:
    def __init__(self, state, startdate, deaddate):
        self.state = state#0 is alive and 1 is dead
        self.startdate = startdate
        self.deaddate = deaddate

""" STEP 1: Loop through plates """
#need to include filter 1 to select only plate folders, as Processed Data folder has been added
d_plates, plate_names = verifyDirlist(pparent,True,"1","")

total_lifespan = 0
wormcnt = 0
platelist = []
for (pplate,name) in zip(d_plates,plate_names):
    csvname = name + ".csv"
    platedataholder = []
    print("current plate:", name)
    lifespanlist = []
    d_sessions_full,d_sessions = verifyDirlist(pplate,True,"")#sessions for the plate
    init_date_extract = d_sessions[0].split()[0].split("-")#read and extract the current date
    init_date = date(int(init_date_extract[0]),int(init_date_extract[1]),int(init_date_extract[2]))

    print(init_date)

    wormslist = [Worm(0,init_date, init_date) for i in range(240)]
    #Then loop over sessions to perform inference
    for (image_folder,datafoldername) in zip(d_sessions_full,d_sessions):
        print("current session:", datafoldername)
        current_date_extract = datafoldername.split()[0].split("-")#read and extract the current date
        current_date = date(int(current_date_extract[0]),int(current_date_extract[1]),int(current_date_extract[2]))
        d_img_full,d_img = verifyDirlist(image_folder,False,'png')
        if len(d_img_full) < 1:
            print("-----------------------------> Skipping (no images!!): ",image_folder)
            continue
        # Get the ROI coordinates
        d_roi = verifyDirlist(image_folder,False,'ROI_Coord')[0]
        if len(d_roi) != 1:
            print("-----------------------------> Skipping (no ROI file!!): ",image_folder)
            labels_np = 6*np.ones(240)
            platedataholder.append(labels_np)
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
        testholder = torch.zeros(240,3,85,85)
        for well in range(240):
            #Crop individual image
            croplis = []
            for img in imglis:
                # Crop to ROI
                crop_img = img[y[well]-ro[well]:y[well]+ro[well], x[well]-ro[well]:x[well]+ro[well]]

                # Resize to standard small size
                crop_img = cv2.resize(crop_img,(img_size,img_size))
                croplis.append(crop_img)
            # merge three frames into a 3d array
            img_merged = np.array(croplis)#shape is 3,85,85
            testholder[well,:,:,:] = torch.from_numpy(img_merged)
        #do the inference and get the predicted labels
        labels = Inference(testholder,net,device,nn.CrossEntropyLoss())
        labels_np = labels.numpy()
        platedataholder.append(labels_np)
        #update the worm status, if worm is already dead then skip
        for well in range(240):
            if wormslist[well].state == 0: #alive 
                label = labels[well]
                wormslist[well].state = label
                wormslist[well].deaddate = current_date
            #dead worm's status won't be updateded so the deaddate is their dead date
            #else: #dead case, but still need to be considered since there are 'fake death'


        #loop over sessions finishes here
    labelresult = np.array(platedataholder)
    print(labelresult.shape)
    np.savetxt(os.path.join(presult, csvname), labelresult, delimiter = ',')
    #now process the data and calculate the lifespan
    for worm in wormslist:
        lifespan = differ_days(worm.deaddate, worm.startdate)
        lifespanlist.append(lifespan)
        total_lifespan = total_lifespan + lifespan
        wormcnt = wormcnt + 1
    platelist.append(lifespanlist)


#loop of all plates finishes
lifespan_avg = total_lifespan/wormcnt
print("The average lifespan of all worms is ",lifespan_avg)
lifespandata = np.array(platelist)
outfile = "lifespandata"
np.save(outfile, lifespandata)




        
                










""" STEP 2: CREATE OR LOAD CNN, LOSS FCN, OPTIMIZER """

'''
# Create basic or load
net = Net(img_size,ker_size)

if use_loaded_net:
        
    # Load the overall checkpoint
    checkpoint = torch.load(load_net)

    # Load the network state
    net.load_state_dict(checkpoint['model_state_dict'])

    # Set a basic optimizer
    #optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=_momentum)
    optimizer = optim.Adam(net.parameters())
    #optimizer = optim.Adagrad(net.parameters(),lr=learn_rate)
    #optimizer = optim.ASGD(net.parameters(),lr=learn_rate)

    # Load the optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load the starting epoch
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']

else:
    # Use the basic optimizer and set start epoch to 0
    optimizer = optim.Adam(net.parameters(),lr = 0.0002)
    #optimizer = optim.Adamax(net.parameters())


    start_epoch = 0

# Set criterion
criterion = nn.CrossEntropyLoss()

# Transfer to CUDA device
net = net.float()
net.to(device)
'''


