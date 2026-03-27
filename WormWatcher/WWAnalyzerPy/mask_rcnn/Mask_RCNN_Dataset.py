import os
import numpy as np
import torch
import cv2
from PIL import Image
import random

import sys
# Add the utils folder to the path so we can import the helper modules
# Note that this is done relative to the mask_rcnn folder so it should work regardless of your repo directory
utils_path = sys.path[0].split("\\")[:-1] # Get the current path to the script and remove the mask_rcnn directory
base_path = '\\'.join(utils_path) # Create the path to the base folder
utils_path.append('mask_rcnn_utils') # append the utils folder to the path
utils_path = '\\'.join(utils_path) # create the final path string to the utils folder
sys.path.append(base_path) # Add the base folder to the system's path
sys.path.append(utils_path) # Add the utils folder to the system's path

import utils as U





class WormDataset(torch.utils.data.Dataset):
    '''
    Defines a dataset to be used with the Mask RCNN network
    '''


    # Constructor, 
    def __init__(self, root, transforms, num_classes, class_parameters=None, 
                 use_cropping=False, crop_size=(512, 512), overlap_ratio=0.1, 
                 min_objects_per_crop=1, max_crops_per_image=16):
        '''
        root: root folder of the dataset containing the DatasetImages and DatasetMasks folders.
        transforms: any transforms to be performed on the dataset images. We have been doing our own augmentations so this typically only converts the images to tensors.
        num_classes: the number of classes for the datset. Right now this only matters if its 1 because if it is then we flatten the output labels to all be the same.
        class_parameters: the encoding parameters for the masks. Should be in the format: {ClassId: (label, start_val, end_val, channel)}. If no params are provided then the dataset will default to the parameters from util.py "get_labeling_params" method.
        use_cropping: whether to crop large images into smaller patches
        crop_size: (width, height) of each crop
        overlap_ratio: ratio of overlap between adjacent crops (0.0 to 0.5)
        min_objects_per_crop: minimum number of objects required to keep a crop
        max_crops_per_image: maximum number of crops to generate per image
        '''
        self.root = root
        self.transforms = transforms
        # find all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "DatasetImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "DatasetMasks"))))
        self.num_classes = num_classes
        if class_parameters is None:
            self.class_params = U.get_labeling_params()
        else:
            self.class_params = class_parameters

        # Cropping parameters
        self.use_cropping = use_cropping
        self.crop_size = crop_size
        self.overlap_ratio = overlap_ratio
        self.min_objects_per_crop = min_objects_per_crop
        self.max_crops_per_image = max_crops_per_image
        
        # Pre-compute valid crops for each image if cropping is enabled
        self.crop_info = []
        if self.use_cropping:
            self._precompute_valid_crops()
        
        print(f'label params: {self.class_params}')
        if self.use_cropping:
            print(f'Cropping enabled: {len(self.crop_info)} valid crops generated from {len(self.imgs)} images')
            print(f'Crop size: {self.crop_size}')
            print(f'This will significantly reduce memory usage!')
        else:
            print(f'Cropping disabled: using full {len(self.imgs)} images')

    def _precompute_valid_crops(self):
        '''Pre-compute valid crop locations for all images'''
        print("Pre-computing valid crop locations...")
        
        for img_idx, (img_name, mask_name) in enumerate(zip(self.imgs, self.masks)):
            if img_idx % 10 == 0:
                print(f"Processing image {img_idx + 1}/{len(self.imgs)}")
                
            # Load mask to determine object locations
            mask_path = os.path.join(self.root, "DatasetMasks", mask_name)
            mask = np.array(Image.open(mask_path))
            img_height, img_width = mask.shape[:2]
            
            # Generate potential crop locations
            step_x = int(self.crop_size[0] * (1 - self.overlap_ratio))
            step_y = int(self.crop_size[1] * (1 - self.overlap_ratio))
            
            valid_crops = []
            for y in range(0, img_height - self.crop_size[1] + 1, step_y):
                for x in range(0, img_width - self.crop_size[0] + 1, step_x):
                    # Check if this crop contains enough objects
                    crop_mask = mask[y:y+self.crop_size[1], x:x+self.crop_size[0]]
                    
                    # Count objects in this crop
                    object_count = 0
                    for channel in range(3):
                        channel_mask = crop_mask[:,:,channel]
                        unique_ids = np.unique(channel_mask)
                        # Remove background (0) and count valid objects
                        valid_ids = []
                        for obj_id in unique_ids[1:]:  # Skip background
                            # Check if this ID corresponds to a valid class
                            channel_labels = [lbl for lbl in self.class_params if self.class_params[lbl][3] == channel]
                            for lbl in channel_labels:
                                if self.class_params[lbl][1] <= obj_id <= self.class_params[lbl][2]:
                                    valid_ids.append(obj_id)
                                    break
                        object_count += len(valid_ids)
                    
                    if object_count >= self.min_objects_per_crop:
                        valid_crops.append((x, y, x + self.crop_size[0], y + self.crop_size[1]))
                    
                    if len(valid_crops) >= self.max_crops_per_image:
                        break
                
                if len(valid_crops) >= self.max_crops_per_image:
                    break
            
            # Store crop info for this image
            for crop_coords in valid_crops:
                self.crop_info.append((img_idx, crop_coords))

    def _extract_crop_data(self, img, mask, crop_coords):
        '''Extract and process a crop from the full image and mask'''
        x1, y1, x2, y2 = crop_coords
        
        # Crop image and mask
        img_crop = img.crop((x1, y1, x2, y2))
        mask_crop = mask[y1:y2, x1:x2]
        
        # Process mask to extract objects and labels
        mask_labels = []
        label_names = []
        channel_masks = []
        
        for channel in range(3):
            this_channel = mask_crop[:,:,channel]
            this_obj_ids = np.unique(this_channel)
            this_obj_ids = this_obj_ids[1:]  # Remove background
            
            # Assign labels for each instance according to this channel
            this_channel_labels = [x for x in self.class_params if self.class_params[x][3] == channel]
            ids_to_remove = []
            
            for id in this_obj_ids:
                for index, lbl in enumerate(this_channel_labels):
                    if id >= self.class_params[lbl][1] and id <= self.class_params[lbl][2]:
                        mask_labels.append(lbl)
                        label_names.append(self.class_params[lbl][0])
                        break
                    if index == len(this_channel_labels) - 1:
                        ids_to_remove.append(id)
            
            # Remove excluded labels
            this_obj_ids = np.array([x for x in this_obj_ids if x not in ids_to_remove])
            
            if len(this_obj_ids) > 0:
                this_channel_masks = this_channel == this_obj_ids[:, None, None]
                channel_masks.append(this_channel_masks)
        
        # Concatenate all channel masks
        if len(channel_masks) > 0:
            masks = channel_masks[0]
            for m in range(1, len(channel_masks)):
                masks = np.concatenate((masks, channel_masks[m]))
        else:
            masks = np.empty((0, mask_crop.shape[0], mask_crop.shape[1]), dtype=bool)
        
        return img_crop, masks, mask_labels

    def __getitem__(self, idx):
        '''
        This method is implicitly called by PyTorch during training. It will take an image and its corresponding mask from the dataset, decode the mask according to the labeling parameters of the dataset, and return the image along with its targets.
        '''

        if self.use_cropping:
            # Get crop information
            original_img_idx, crop_coords = self.crop_info[idx]
            
            # Load full images and masks
            img_path = os.path.join(self.root, "DatasetImages", self.imgs[original_img_idx])
            mask_path = os.path.join(self.root, "DatasetMasks", self.masks[original_img_idx])
            img = Image.open(img_path).convert("RGB")
            mask = np.array(Image.open(mask_path))
            
            # Extract crop
            img, masks, mask_labels = self._extract_crop_data(img, mask, crop_coords)
            
            # Process the cropped data
            num_objs = masks.shape[0]
            boxes = []
            for i in range(num_objs):
                pos = np.where(masks[i])
                if len(pos[0]) == 0:  # Skip empty masks
                    continue
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])

                if xmax - xmin < 1 or ymax - ymin < 1:
                    xmax += 1
                    ymax += 1

                boxes.append([xmin, ymin, xmax, ymax])

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            
            if self.num_classes == 1:
                labels = torch.ones((len(boxes),), dtype=torch.int64)
            else:
                labels = torch.as_tensor(mask_labels[:len(boxes)], dtype=torch.int64)

            masks = torch.as_tensor(masks[:len(boxes)], dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target
        
        else:
            # Original behavior - load full images
            img_path = os.path.join(self.root, "DatasetImages", self.imgs[idx])
            mask_path = os.path.join(self.root, "DatasetMasks", self.masks[idx])
            img = Image.open(img_path).convert("RGB")
            # See constructor comments for encoding of masks.
            mask = Image.open(mask_path)
            #mask = Image.open(r'C:\Dropbox\FramesData\FramesForAnnotation\train\Renamed\021 - Copy_mask.png')
            # convert the PIL Image into a numpy array
            mask = np.array(mask) # RGB shape (512, 612, 3)
            #print(f'\t\t$$$$$$$$$$$$$$$$$ Shape: {mask.shape}')

            mask2 = Image.open(mask_path)
            mask2 = np.array(mask2)

            #exclude.append('Egg') # For now just hard code to exclude eggs because the PyTorch training is calling getitem so I can't add the exclude filter to the call.

            #class_params = {0 : ["L1-Larva",1,51,0], 1: ["L2-Larva",52,102,0], 2 : ["L3-Larva",103,153,0], 3 : ["L4-Herm",154,204,0], 4 : ["L4-Male",205,255,0], 5 : ["Adult-Herm",1,51,1], 6 : ["Adult-Male",52,102,1], 7: ["Unkown-Stage", 205, 255, 2]} # Class ID : (Class Name, Start Value, Max Value, Channel)
            #class_params = U.get_labeling_params()
            #class_params = {x : class_params[x] for x in class_params if class_params[x][0] not in exclude} # Filter out the unwanted labels
            #print(f'params after exclude: {class_params}')

            mask_labels = []
            label_names = []
            channel_masks = []
            # The masks are contained within each channel of the image, so loop through each channel
            # get the labels for each mask, and then generate a binary mask for each instance in the channel.
            for channel in range(3):
                this_channel = mask[:,:,channel]
                this_obj_ids = np.unique(this_channel) # Get the instance ids for this channel
                this_obj_ids = this_obj_ids[1:] # First id is the background, so remove it

                # Assign the labels for each instance according to this channel
                this_channel_labels = [x for x in self.class_params if self.class_params[x][3] == channel]
                #print(f'Channel: {channel}, labels: {this_channel_labels}')
                ids_to_remove = []
                for id in this_obj_ids:
                    for index, lbl in enumerate(this_channel_labels):
                        if id >= self.class_params[lbl][1] and id <= self.class_params[lbl][2]:
                            mask_labels.append(lbl)
                            label_names.append(self.class_params[lbl][0])
                            break
                        if index == len(this_channel_labels) - 1:
                            #print(f'Excluded label found. Id: {id}, channel: {channel}')
                            ids_to_remove.append(id) # If we reach here then the id was not found in the class parameters (which means its been excluded)

                # Remove the excluded labels:
                this_obj_ids = np.array([x for x in this_obj_ids if x not in ids_to_remove])

                this_channel_masks = this_channel == this_obj_ids[:, None, None]
                channel_masks.append(this_channel_masks)

                #print(f'mask_labels: {mask_labels}')
                #print(f'labels names: {label_names}')

            # Now that we have all the masks, concatenate them into one numpy array
            masks = channel_masks[0]
            for m in range(1,len(channel_masks)):
                masks = np.concatenate((masks, channel_masks[m]))

        #print(f'Final masks shape after concat: {masks.shape}')

        ##print(f"&&&&&&&&&&&& Shape: {mask.shape}")
        ## instances are encoded as different colors
        #obj_ids = np.unique(mask)
        ##print(f"****************************** image: {self.imgs[idx]}")
        ##print(f"****************************** mask: {self.masks[idx]}")
        ##print(f"****************************************************************Unique IDs for image {idx}: {obj_ids}")

        ## first id is the background, so remove it
        #obj_ids = obj_ids[1:]

        ## split the color-encoded mask into a set
        ## of binary masks
        #masks = mask == obj_ids[:, None, None]
        #print(f'\t\t$$$$$$$$$$$$$$$$$ Shape: {masks.shape}')
        #print(f'\t\t$$$$$$$$$$$$$$$$$ masks: {masks}')

        # get bounding box coordinates for each mask
        #num_objs = len(obj_ids)
        num_objs = masks.shape[0]
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            if xmax - xmin < 0.001 or ymax - ymin < 0.001:
                print(f'\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Bad bounding box found for mask {i} in image: \n\t{img_path}')
                print('\tBounding box height or width is 0, so adding 1 to the dimensions so it is a valid box.\n')
                xmax += 1
                ymax += 1

            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if self.num_classes == 1:
            labels = torch.ones((num_objs,), dtype=torch.int64) # For use with models with only 1 class
        else:
            labels = torch.as_tensor(mask_labels, dtype=torch.int64) # For use with multi-class models

        #print(f'image {idx} path: {img_path}')
        #print(f'image {idx} labels: {labels}')
        #print(f'\t\t$$$$$$$$$$$$$$$$$ labels: {labels.shape}')
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd


        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


    def __len__(self):
        if self.use_cropping:
            return len(self.crop_info)
        else:
            return len(self.imgs)

    #@staticmethod
    #def get_labeling_params(exclude=[]):
    #    class_params = {1 : ["L1-Larva",1,51,0], 2: ["L2-Larva",52,102,0], 3 : ["L3-Larva",103,153,0], 4 : ["L4-Herm",154,204,0], 5 : ["L4-Male",205,255,0], 6 : ["Adult-Herm",1,51,1], 7 : ["Adult-Male",52,102,1], 8 : ["Dauer",103,153,1] , 9: ["Egg", 154, 204, 1], 10: ["Unknown-Stage", 205, 255, 2]} # Class ID : (Class Name, Start Value, Max Value, Channel)

    #    class_params = {x : class_params[x] for x in class_params if class_params[x][0] not in exclude} # Filter out the unwanted labels
    #    return class_params
