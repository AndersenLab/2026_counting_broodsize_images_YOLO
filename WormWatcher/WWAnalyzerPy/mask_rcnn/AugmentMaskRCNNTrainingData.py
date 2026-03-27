'''
This script loads a set of images and their corresponding masks that are to be used in training the Mask RCNN network and augments them with
various image manipulation techniques. Any augmentations on an image are similarly applied to that images mask to ensure the image and masks
continue to accurately represent one another.

The images and their masks are to be in two seperate folders, and the augmented images and masks will be saved in the same folders, but with
modified filenames representing the applied augmentations.

Each image must have a corresponding mask.
'''

import sys, os
sys.path.insert(1, os.path.realpath('.'))
from verifyDirlist import *

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import random
import numpy.random
import itertools

# AUGMENTATION CONFIGURATION
# Set which augmentations to apply (True/False)
APPLY_FLIPS = True
APPLY_BLUR = True
APPLY_NOISE = True
APPLY_BRIGHTNESS = True
APPLY_ROTATION = True
APPLY_SCALING = True
APPLY_TRANSLATION = True
APPLY_CONTRAST = True
APPLY_SHEAR = True
APPLY_ELASTIC = True  # Computationally expensive
APPLY_COMBINATIONS = True

# Number of variations to create for each augmentation type
NUM_ROTATION_VARIATIONS = 3
NUM_BRIGHTNESS_VARIATIONS = 3
NUM_NOISE_VARIATIONS = 2
NUM_BLUR_VARIATIONS = 2
NUM_SCALE_VARIATIONS = 2
NUM_CONTRAST_VARIATIONS = 2
NUM_SHEAR_VARIATIONS = 2
NUM_TRANSLATION_VARIATIONS = 2

# Probability of applying expensive augmentations
ELASTIC_PROBABILITY = 0.3  # Apply elastic deformation to 30% of images

# Validation parameters for mask quality
MIN_OBJECTS_PER_MASK = 1        # Minimum number of objects required in mask
MIN_PIXELS_PER_OBJECT = 10      # Minimum pixels required per object

# Parameter ranges for augmentations
ROTATION_RANGE = (-45, 45)  # degrees - expanded from ±30 to ±45 (rotation doesn't distort object shape)
SCALE_RANGE = (0.7, 1.3)    # scale factors - expanded from 0.8-1.2 to 0.7-1.3 (scaling doesn't distort object shape)
TRANSLATION_PERCENT = 12    # percentage of image size for max translation - expanded from 8% to 12% (translation doesn't distort object shape)
CONTRAST_RANGE = (0.7, 1.3) # contrast factors
SHEAR_RANGE = (-0.1, 0.1)   # shear factors - kept reduced from ±0.2 to ±0.1 (shear DOES distort object shape)
NOISE_RANGE = (5, 15)       # noise intensity
BRIGHTNESS_RANGE = (30, 70) # brightness change amount
BLUR_KERNEL_RANGE = (7, 14) # gaussian blur kernel size range (odd numbers only)

def flipImage(image, mask, augmentations = [], v_flip = False):
    if v_flip:
        flip_direction = 0
        augmentations.append("vflip")
    else:
        flip_direction = 1
        augmentations.append("hflip")
    
    image_flip = cv2.flip(image, flip_direction)
    mask_flip = cv2.flip(mask, flip_direction)

    return image_flip, mask_flip, augmentations

def blurImage(image, mask, augmentations = [], blur_kernel = None):
    if blur_kernel is None:
        blur_kernel = random.randrange(7, 14, 2)
    # ONLY blur the image, NOT the mask (mask must preserve discrete values)
    b_img = cv2.GaussianBlur(image, (blur_kernel,blur_kernel), 0)
    augmentations.append(f'gblur{blur_kernel}')
    return b_img, mask, augmentations  # mask unchanged

def addRandomNoise(image, mask, augmentations = [], noise_amount = None):
    # ONLY add noise to image, NOT to mask (mask must preserve discrete values)
    if noise_amount is None:
        noise_amount = random.randint(5, 15)
    noise = np.random.normal(0, noise_amount, image.shape).astype(np.float32)
    
    # Convert image to float, add noise, then convert back and clip
    img_float = image.astype(np.float32)
    img_noise = img_float + noise
    img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)

    augmentations.append(f'rnoise{noise_amount}')
    return img_noise, mask, augmentations  # mask unchanged

def changeBrightness(image, mask, augmentations = [], brightness_change = None):
    # ONLY change brightness of image, NOT mask (mask must preserve discrete values)
    if brightness_change is None:
        brightness_change = random.randint(30,70)
    lighting = np.full(image.shape, brightness_change).astype('uint8')
    if random.randint(0,1) == 1:
        # Brighten
        img_lighting = cv2.add(image, lighting)
        augmentations.append(f'brighten{brightness_change}')
    else:
        # Darken
        img_lighting = cv2.subtract(image, lighting)
        augmentations.append(f'darken{brightness_change}')
    
    return img_lighting, mask, augmentations  # mask unchanged

def rotateImage(image, mask, augmentations = [], angle = None):
    """Rotate image and mask by a random angle between -30 and 30 degrees"""
    if angle is None:
        angle = random.randint(-30, 30)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Apply rotation with different interpolation methods
    # Image: use linear interpolation for smooth results
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                  borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
    # Mask: use nearest neighbor to preserve discrete values
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (width, height), 
                                 borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    
    augmentations.append(f'rot{angle}')
    return rotated_image, rotated_mask, augmentations

def scaleImage(image, mask, augmentations = [], scale_factor = None):
    """Scale image and mask by a random factor between 0.8 and 1.2"""
    if scale_factor is None:
        scale_factor = random.uniform(0.8, 1.2)
    height, width = image.shape[:2]
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize with different interpolation methods
    # Image: use linear interpolation for smooth results
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    # Mask: use nearest neighbor to preserve discrete values
    scaled_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    
    # If scaled up, crop to original size; if scaled down, pad to original size
    if scale_factor > 1.0:
        # Crop from center
        start_x = (new_width - width) // 2
        start_y = (new_height - height) // 2
        scaled_image = scaled_image[start_y:start_y+height, start_x:start_x+width]
        scaled_mask = scaled_mask[start_y:start_y+height, start_x:start_x+width]
    else:
        # Pad to original size
        pad_x = (width - new_width) // 2
        pad_y = (height - new_height) // 2
        scaled_image = cv2.copyMakeBorder(scaled_image, pad_y, height-new_height-pad_y, 
                                         pad_x, width-new_width-pad_x, cv2.BORDER_REFLECT)
        scaled_mask = cv2.copyMakeBorder(scaled_mask, pad_y, height-new_height-pad_y, 
                                        pad_x, width-new_width-pad_x, cv2.BORDER_REFLECT)
    
    augmentations.append(f'scale{scale_factor:.2f}')
    return scaled_image, scaled_mask, augmentations

def translateImage(image, mask, augmentations = [], tx = None, ty = None):
    """Translate image and mask by random amounts"""
    height, width = image.shape[:2]
    if tx is None:
        tx = random.randint(-width//8, width//8)
    if ty is None:
        ty = random.randint(-height//8, height//8)
    
    # Create translation matrix
    translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
    
    # Apply translation with different interpolation methods
    # Image: use linear interpolation for smooth results
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height), 
                                     borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
    # Mask: use nearest neighbor to preserve discrete values
    translated_mask = cv2.warpAffine(mask, translation_matrix, (width, height), 
                                    borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    
    augmentations.append(f'trans{tx},{ty}')
    return translated_image, translated_mask, augmentations

def adjustContrast(image, mask, augmentations = [], contrast_factor = None):
    """Adjust contrast of the image ONLY (mask must preserve discrete values)"""
    if contrast_factor is None:
        contrast_factor = random.uniform(0.7, 1.3)
    
    # Apply contrast adjustment ONLY to image, NOT mask
    contrasted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
    
    augmentations.append(f'contrast{contrast_factor:.2f}')
    return contrasted_image, mask, augmentations  # mask unchanged

def shearImage(image, mask, augmentations = [], shear_x = None, shear_y = None):
    """Apply shear transformation to image and mask"""
    height, width = image.shape[:2]
    
    # Random shear factors
    if shear_x is None:
        shear_x = random.uniform(-0.2, 0.2)
    if shear_y is None:
        shear_y = random.uniform(-0.2, 0.2)
    
    # Create shear matrix
    shear_matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
    
    # Apply shear with different interpolation methods
    # Image: use linear interpolation for smooth results
    sheared_image = cv2.warpAffine(image, shear_matrix, (width, height), 
                                  borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
    # Mask: use nearest neighbor to preserve discrete values
    sheared_mask = cv2.warpAffine(mask, shear_matrix, (width, height), 
                                 borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
    
    augmentations.append(f'shear{shear_x:.2f},{shear_y:.2f}')
    return sheared_image, sheared_mask, augmentations

def elasticDeformation(image, mask, augmentations = []):
    """Apply elastic deformation to image and mask"""
    height, width = image.shape[:2]
    
    # Parameters for elastic deformation - reduced for more subtle distortion
    alpha = random.uniform(20, 60)   # Strength of deformation - reduced from 50-150 to 20-60
    sigma = random.uniform(6, 10)    # Smoothness of deformation - reduced from 8-12 to 6-10
    
    # Generate random displacement fields
    dx = np.random.randn(height, width) * alpha
    dy = np.random.randn(height, width) * alpha
    
    # Smooth the displacement fields
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply displacement
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    # Apply deformation with different interpolation methods
    # Image: use linear interpolation for smooth results
    deformed_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # Mask: use nearest neighbor to preserve discrete values
    deformed_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    
    augmentations.append(f'elastic{alpha:.0f},{sigma:.1f}')
    return deformed_image, deformed_mask, augmentations

def generateAugmentationCombinations(image, mask):
    """Generate multiple augmentation combinations for a single image"""
    augmented_images = []
    
    # Start with the original image
    augmented_images.append((image, mask, []))
    
    # Single augmentations with multiple variations
    if APPLY_FLIPS:
        # Horizontal flip
        h_flip, mask_h_flip, h_augmentations = flipImage(image, mask, augmentations=[], v_flip=False)
        augmented_images.append((h_flip, mask_h_flip, h_augmentations.copy()))
        
        # Vertical flip
        v_flip, mask_v_flip, v_augmentations = flipImage(image, mask, augmentations=[], v_flip=True)
        augmented_images.append((v_flip, mask_v_flip, v_augmentations.copy()))
        
        # Both flips
        vh_flip, mask_vh_flip, vh_augmentations = flipImage(h_flip, mask_h_flip, augmentations=h_augmentations.copy(), v_flip=True)
        augmented_images.append((vh_flip, mask_vh_flip, vh_augmentations.copy()))
    
    if APPLY_ROTATION:
        # Multiple rotation variations - using expanded angles (rotation doesn't distort object shape)
        rotation_angles = [-45, -30, -15, 15, 30, 45]
        for angle in rotation_angles[:NUM_ROTATION_VARIATIONS]:
            rotated_image, rotated_mask, rotated_augmentations = rotateImage(image, mask, augmentations=[], angle=angle)
            augmented_images.append((rotated_image, rotated_mask, rotated_augmentations.copy()))
    
    if APPLY_BRIGHTNESS:
        # Multiple brightness variations
        brightness_changes = [30, 50, 70]
        for brightness in brightness_changes[:NUM_BRIGHTNESS_VARIATIONS]:
            bright_image, bright_mask, bright_augmentations = changeBrightness(image, mask, augmentations=[], brightness_change=brightness)
            augmented_images.append((bright_image, bright_mask, bright_augmentations.copy()))
    
    if APPLY_BLUR:
        # Multiple blur variations
        blur_kernels = [7, 11, 13]
        for kernel in blur_kernels[:NUM_BLUR_VARIATIONS]:
            blur_image, blur_mask, blur_augmentations = blurImage(image, mask, augmentations=[], blur_kernel=kernel)
            augmented_images.append((blur_image, blur_mask, blur_augmentations.copy()))
    
    if APPLY_NOISE:
        # Multiple noise variations
        noise_amounts = [5, 10, 15]
        for noise in noise_amounts[:NUM_NOISE_VARIATIONS]:
            noise_image, noise_mask, noise_augmentations = addRandomNoise(image, mask, augmentations=[], noise_amount=noise)
            augmented_images.append((noise_image, noise_mask, noise_augmentations.copy()))
    
    if APPLY_SCALING:
        # Multiple scale variations - using expanded scaling (scaling doesn't distort object shape)
        scale_factors = [0.7, 0.85, 1.15, 1.3]
        for scale in scale_factors[:NUM_SCALE_VARIATIONS]:
            scaled_image, scaled_mask, scaled_augmentations = scaleImage(image, mask, augmentations=[], scale_factor=scale)
            augmented_images.append((scaled_image, scaled_mask, scaled_augmentations.copy()))
    
    if APPLY_TRANSLATION:
        # Multiple translation variations - using expanded translation amounts (translation doesn't distort object shape)
        height, width = image.shape[:2]
        translations = [(width//8, height//8), (-width//8, -height//8), (width//10, -height//10), (-width//10, height//10)]
        for tx, ty in translations[:NUM_TRANSLATION_VARIATIONS]:
            trans_image, trans_mask, trans_augmentations = translateImage(image, mask, augmentations=[], tx=tx, ty=ty)
            augmented_images.append((trans_image, trans_mask, trans_augmentations.copy()))
    
    if APPLY_CONTRAST:
        # Multiple contrast variations
        contrast_factors = [0.7, 0.85, 1.15, 1.3]
        for contrast in contrast_factors[:NUM_CONTRAST_VARIATIONS]:
            contrast_image, contrast_mask, contrast_augmentations = adjustContrast(image, mask, augmentations=[], contrast_factor=contrast)
            augmented_images.append((contrast_image, contrast_mask, contrast_augmentations.copy()))
    
    if APPLY_SHEAR:
        # Multiple shear variations - using conservative shear values
        shear_combinations = [(0.05, 0.025), (-0.05, -0.025), (0.075, -0.05), (-0.075, 0.05)]
        for shear_x, shear_y in shear_combinations[:NUM_SHEAR_VARIATIONS]:
            shear_image, shear_mask, shear_augmentations = shearImage(image, mask, augmentations=[], shear_x=shear_x, shear_y=shear_y)
            augmented_images.append((shear_image, shear_mask, shear_augmentations.copy()))
    
    if APPLY_ELASTIC and random.random() < ELASTIC_PROBABILITY:
        # Elastic deformation (applied less frequently)
        elastic_image, elastic_mask, elastic_augmentations = elasticDeformation(image, mask, augmentations=[])
        augmented_images.append((elastic_image, elastic_mask, elastic_augmentations.copy()))
    
    if APPLY_COMBINATIONS:
        # Two-way combinations
        base_images = augmented_images[1:min(8, len(augmented_images))]  # Use first few augmented images as base
        
        for base_img, base_mask, base_augs in base_images:
            # Add rotation to existing augmentations
            if APPLY_ROTATION and 'rot' not in str(base_augs):
                combo_img, combo_mask, combo_augs = rotateImage(base_img, base_mask, augmentations=base_augs.copy(), angle=15)
                augmented_images.append((combo_img, combo_mask, combo_augs.copy()))
            
            # Add brightness to existing augmentations
            if APPLY_BRIGHTNESS and 'brighten' not in str(base_augs) and 'darken' not in str(base_augs):
                combo_img, combo_mask, combo_augs = changeBrightness(base_img, base_mask, augmentations=base_augs.copy(), brightness_change=40)
                augmented_images.append((combo_img, combo_mask, combo_augs.copy()))
            
            # Add contrast to existing augmentations
            if APPLY_CONTRAST and 'contrast' not in str(base_augs):
                combo_img, combo_mask, combo_augs = adjustContrast(base_img, base_mask, augmentations=base_augs.copy(), contrast_factor=1.2)
                augmented_images.append((combo_img, combo_mask, combo_augs.copy()))
            
            # Add noise to existing augmentations
            if APPLY_NOISE and 'rnoise' not in str(base_augs):
                combo_img, combo_mask, combo_augs = addRandomNoise(base_img, base_mask, augmentations=base_augs.copy(), noise_amount=8)
                augmented_images.append((combo_img, combo_mask, combo_augs.copy()))
        
        # Three-way combinations (selected combinations to avoid explosion)
        def create_rotation_brightness_contrast(img, mask):
            rot_img, rot_mask, rot_augs = rotateImage(img, mask, [], 20)  # Expanded rotation
            bright_img, bright_mask, bright_augs = changeBrightness(rot_img, rot_mask, rot_augs, 40)
            contrast_img, contrast_mask, contrast_augs = adjustContrast(bright_img, bright_mask, bright_augs, 1.1)
            return contrast_img, contrast_mask, contrast_augs
        
        def create_scale_translation_noise(img, mask):
            scale_img, scale_mask, scale_augs = scaleImage(img, mask, [], 1.2)  # Expanded scaling
            trans_img, trans_mask, trans_augs = translateImage(scale_img, scale_mask, scale_augs, img.shape[1]//8, img.shape[0]//8)  # Expanded translation
            noise_img, noise_mask, noise_augs = addRandomNoise(trans_img, trans_mask, trans_augs, 8)
            return noise_img, noise_mask, noise_augs
        
        def create_flip_blur_brightness(img, mask):
            flip_img, flip_mask, flip_augs = flipImage(img, mask, [], False)
            blur_img, blur_mask, blur_augs = blurImage(flip_img, flip_mask, flip_augs, 9)
            bright_img, bright_mask, bright_augs = changeBrightness(blur_img, blur_mask, blur_augs, 50)
            return bright_img, bright_mask, bright_augs
        
        def create_shear_rotation_contrast(img, mask):
            shear_img, shear_mask, shear_augs = shearImage(img, mask, [], 0.05, 0.025)  # Keep reduced shear (distorts shape)
            rot_img, rot_mask, rot_augs = rotateImage(shear_img, shear_mask, shear_augs, -15)  # Expanded rotation (doesn't distort shape)
            contrast_img, contrast_mask, contrast_augs = adjustContrast(rot_img, rot_mask, rot_augs, 1.15)
            return contrast_img, contrast_mask, contrast_augs
        
        popular_combinations = [
            create_rotation_brightness_contrast,
            create_scale_translation_noise,
            create_flip_blur_brightness,
            create_shear_rotation_contrast,
        ]
        
        for combo_func in popular_combinations:
            try:
                combo_img, combo_mask, combo_augs = combo_func(image, mask)
                augmented_images.append((combo_img, combo_mask, combo_augs.copy()))
            except:
                pass  # Skip if combination fails
    
    return augmented_images

# Add Gaussian noise to the image. Adapted from stackoverflow:
# https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv
# Note that we don't need to change the mask at all because the position of the worm is not changing with this augmentation
def addNoise(image):
    # TODO: Make it work properly. THIS IS CURRENTLY NOT WORKING.
    print(f"image: {image}")
    row,col,ch= image.shape
    mean = 0
    sigma = 5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch) 
    print(f"gauss: {gauss}")
    gauss = (gauss).astype(int)
    print(f"gauss int: {gauss}")
    noisy = image + gauss
    return noisy

def validateMask(mask, min_objects=1, min_pixels_per_object=10, debug=False):
    """
    Validate that the mask contains viable objects.
    REJECTS the entire image if ANY object is smaller than min_pixels_per_object.
    
    Args:
        mask: numpy array of the mask
        min_objects: minimum number of objects required
        min_pixels_per_object: minimum pixels required per object
        debug: if True, print detailed debugging information
    
    Returns:
        bool: True if mask is valid, False otherwise
    """
    # Convert to grayscale if needed
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # Get unique values (excluding background)
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values != 0]
    
    if debug:
        print(f"🔍 Mask validation: {len(unique_values)} objects found")
        print(f"   Unique values: {unique_values}")
    
    if len(unique_values) < min_objects:
        if debug:
            print(f"❌ Not enough objects: {len(unique_values)} < {min_objects}")
        return False
    
    # Check if ALL objects have sufficient pixels
    # If ANY object is too small, reject the entire image
    tiny_objects = []
    for value in unique_values:
        object_pixels = np.sum(mask == value)
        if debug:
            print(f"   Object {value}: {object_pixels} pixels")
        if object_pixels < min_pixels_per_object:
            tiny_objects.append((value, object_pixels))
    
    if tiny_objects:
        if debug:
            print(f"❌ Tiny objects found (min required: {min_pixels_per_object}):")
            for value, pixels in tiny_objects:
                print(f"     Object {value}: {pixels} pixels")
        return False
    
    # All objects are valid size and we have enough objects
    if debug:
        print(f"✅ All objects valid (min: {min_pixels_per_object} pixels)")
    return True

def saveAugmentations(path_image, image, path_mask, mask, augmentations = []):
    # Validate mask before saving
    if not validateMask(mask, MIN_OBJECTS_PER_MASK, MIN_PIXELS_PER_OBJECT):
        print(f"Skipping augmentation {'+'.join(augmentations)} - no valid objects in mask")
        return False
    
    if ".png" in path_image:
        path_image = path_image[:-4]
        path_mask = path_mask[:-4]
    else:
        print("Only supports png filetypes.")
        return False
    
    for aug in augmentations:
        path_image = path_image + "_" + aug
        path_mask = path_mask + "_" + aug

    path_image = path_image + ".png"
    path_mask = path_mask + ".png"

    #print(f"Saving image: {path_image}")

    # Save image with default compression
    cv2.imwrite(path_image, image)
    
    # Save mask with NO compression to prevent artifacts
    cv2.imwrite(path_mask, mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # POST-SAVE VALIDATION: Check mask after save/load cycle
    # This catches compression artifacts that create tiny objects
    # CRITICAL: Load mask exactly the same way the dataset loader does!
    try:
        from PIL import Image
        saved_mask = np.array(Image.open(path_mask))  # Same as dataset loader
        
        if not validateMask(saved_mask, MIN_OBJECTS_PER_MASK, MIN_PIXELS_PER_OBJECT):
            print(f"🚨 POST-SAVE VALIDATION FAILED: {'+'.join(augmentations)} - compression artifacts detected!")
            print(f"   File: {os.path.basename(path_mask)}")
            print(f"   Running detailed validation to identify the issue:")
            # Run validation again with debug mode to see what went wrong
            validateMask(saved_mask, MIN_OBJECTS_PER_MASK, MIN_PIXELS_PER_OBJECT, debug=True)
            print(f"   Deleting corrupted files: {os.path.basename(path_image)}, {os.path.basename(path_mask)}")
            # Delete the corrupted files
            try:
                os.remove(path_image)
                os.remove(path_mask)
            except:
                pass
            return False
    except Exception as e:
        print(f"⚠️  Could not load saved mask for validation: {e}")
        # If we can't load it, it's definitely corrupted
        try:
            os.remove(path_image)
            os.remove(path_mask)
        except:
            pass
        return False
    
    return True


#if __name__ == "main":

pbase = r'C:\Users\pdb69\Desktop\WormCamp Cluster Assays'


root = tk.Tk()
root.withdraw()
# Fixed: Import filedialog correctly
pimgs = filedialog.askdirectory(parent=root, initialdir=pbase, title='Please select the DatasetImages folder')
pmasks = pimgs.replace("DatasetImages","DatasetMasks")

#pimgs = r'C:\Users\pdb69\Desktop\WormTrainingDataTEST\Images'
#pmasks = r'C:\Users\pdb69\Desktop\WormTrainingDataTEST\Masks'

image_paths, _ = verifyDirlist(pimgs,False,".png") 
mask_paths, _ = verifyDirlist(pmasks,False,".png")

print(f'{len(image_paths)} images found.')
print(f'{len(mask_paths)} masks found.')

# Create a mapping from image basename to full paths
def get_base_name(filepath):
    """Get base filename without _mask suffix and extension"""
    filename = os.path.basename(filepath)
    # Remove extension
    base = os.path.splitext(filename)[0]
    # Remove _mask suffix if present
    if base.endswith('_mask'):
        base = base[:-5]
    return base

# Create mappings
image_map = {}
mask_map = {}

for img_path in image_paths:
    base_name = get_base_name(img_path)
    image_map[base_name] = img_path

for mask_path in mask_paths:
    base_name = get_base_name(mask_path)
    mask_map[base_name] = mask_path

# Find matching pairs
matched_pairs = []
unmatched_images = []
unmatched_masks = []

for base_name, img_path in image_map.items():
    if base_name in mask_map:
        matched_pairs.append((img_path, mask_map[base_name]))
    else:
        unmatched_images.append(img_path)

for base_name, mask_path in mask_map.items():
    if base_name not in image_map:
        unmatched_masks.append(mask_path)

print(f'Matched pairs: {len(matched_pairs)}')
print(f'Unmatched images: {len(unmatched_images)}')
print(f'Unmatched masks: {len(unmatched_masks)}')

if unmatched_images:
    print("Unmatched images:")
    for img in unmatched_images[:5]:
        print(f"  - {os.path.basename(img)}")

if unmatched_masks:
    print("Unmatched masks:")
    for mask in unmatched_masks[:5]:
        print(f"  - {os.path.basename(mask)}")

print(f'Configuration: Generating approximately {NUM_ROTATION_VARIATIONS + NUM_BRIGHTNESS_VARIATIONS + NUM_BLUR_VARIATIONS + NUM_NOISE_VARIATIONS + NUM_SCALE_VARIATIONS + NUM_TRANSLATION_VARIATIONS + NUM_CONTRAST_VARIATIONS + NUM_SHEAR_VARIATIONS + 10} augmentations per image')

# Initialize overall statistics
total_saved = 0
total_skipped = 0

# Process matched pairs only
for idx, (image_path, mask_path) in enumerate(matched_pairs):
    
    if (idx != 0 and idx % 10 == 0):  # Reduced frequency due to more augmentations
        print(f'{idx}/{len(matched_pairs)} image pairs augmented.')

    print(f'Augmenting pair {idx+1}: {os.path.basename(image_path)} + {os.path.basename(mask_path)}')
    
    image = cv2.imread(image_path,cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    
    if image is None:
        print(f"ERROR: Could not load image: {image_path}")
        continue
        
    if mask is None:
        print(f"ERROR: Could not load mask: {mask_path}")
        continue

    # Generate all augmentation combinations
    augmented_images = generateAugmentationCombinations(image, mask)

    # Save all augmented images (skip the original at index 0)
    saved_count = 0
    skipped_count = 0
    
    for (img, msk, augs) in augmented_images[1:]:
        if saveAugmentations(image_path, img, mask_path, msk, augs):
            saved_count += 1
        else:
            skipped_count += 1

    # Update overall statistics
    total_saved += saved_count
    total_skipped += skipped_count

    print(f'{idx + 1}/{len(matched_pairs)} image pairs augmented. Generated {len(augmented_images)-1} variations. Saved: {saved_count}, Skipped: {skipped_count}')

# Print overall summary
print(f'\n=== AUGMENTATION COMPLETE ===')
print(f'Total image pairs processed: {len(matched_pairs)}')
print(f'Total augmentations saved: {total_saved}')
print(f'Total augmentations skipped: {total_skipped}')
print(f'Success rate: {total_saved/(total_saved + total_skipped)*100:.1f}%' if (total_saved + total_skipped) > 0 else 'N/A')
print(f'Average augmentations per image: {total_saved/len(matched_pairs):.1f}' if len(matched_pairs) > 0 else 'N/A')

if total_skipped > 0:
    print(f'\nNote: {total_skipped} augmentations were skipped because they resulted in empty masks.')
    print('Consider adjusting augmentation parameters if too many are being skipped.')
    print('Current validation criteria:')
    print(f'  - Minimum objects per mask: {MIN_OBJECTS_PER_MASK}')
    print(f'  - Minimum pixels per object: {MIN_PIXELS_PER_OBJECT}')
