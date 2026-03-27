
# Torch
import torch
import torchvision
import torchvision.transforms as transforms

# Torch sub-modules
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

# Other
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
import time

# Anthony's import
from verifyDirlist import *
from CNN_Helper import *


if __name__ == '__main__':

    # Parameters - load or create network
    pdata = r'F:\Matt eLIfe paper data (merged)\Processed Data'
    pnet = os.path.join(pdata,'checkpoint')
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
    load_limit_train = 9286 # Set maximum image set size to load for each type
    load_limit_test = 2321
    batch_size = 32;
    num_epochs = 20;
    use_cuda = False
    learn_rate = 0.0005
    _momentum = 0.9 # SGD only

    normtype = 1 # 0 = none, 1= normalize21, 2= subtract mean

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

    """ STEP 1: LOAD DATA """

    # Define training and testing sets and classes
    trainset = ImageDataset(ptrain,classes,load_limit_train,normtype)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
   
    testset = ImageDataset(ptest,classes,load_limit_test,normtype)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    validset = ImageDataset(pvalid,classes,min((20000,load_limit_test)),normtype)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=2)

    


    # Show some images for fun
    # showSampleImages(trainloader)

    """ STEP 2: CREATE OR LOAD CNN, LOSS FCN, OPTIMIZER """


    # Create basic or load
    net = Net(img_size,ker_size)

    if use_loaded_net:
        
        # Load the overall checkpoint
        checkpoint = torch.load(load_net,map_location=torch.device('cpu'))

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
        #optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=_momentum)
        optimizer = optim.Adam(net.parameters(),lr = 0.0002)
        #optimizer = optim.Adamax(net.parameters())
        #optimizer = optim.Adagrad(net.parameters(),lr=learn_rate)
        #optimizer = optim.ASGD(net.parameters(),lr=learn_rate)


        start_epoch = 0

    # Set criterion
    criterion = nn.CrossEntropyLoss()

    # Transfer to CUDA device
    net = net.float()
    net.to(device)

    # Pre-test
    testCNN(validloader,net,device,criterion)

    """ STEP 3: TRAIN THE NETWORK """

    '''
    for epoch in range(start_epoch,num_epochs):  # loop over the dataset multiple times
        start_time = time.time()
        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels], but transfer them to GPU
            # inputs, labels, activations = data # CPU version
            inputs, labels, activations = data[0].to(device), data[1], data[2].to(device)  # GPU version

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterion(outputs, activations)
            loss.backward()
            optimizer.step()

            # print loss statistics every 25% of an epoch
            running_loss += loss.item()
            iternum = int(len(trainset)/batch_size*0.25)
            if i % iternum == iternum-1:   

                # Loss on training data
                print('[%d, %5d] loss: %.4f' %(epoch + 1, i + 1, running_loss / iternum))
                running_loss = 0.0

        # Print the validation accuracy every epoch
        testCNN(validloader,net,device,criterion,start_time)

        # Save network every epoch
        outpath =os.path.join(pnet,'latest_net.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss},
            outpath)



    print('Finished Training')



    """ STEP 4: TEST ON TEST DATA """
    testCNN(testloader,net,device,criterion)


    '''
