"""
Debug version of the augmentation script to identify where image/mask saving failures occur.
"""

import cv2
import numpy as np
import os

def debug_saveAugmentations(path_image, image, path_mask, mask, augmentations = []):
    """Debug version of saveAugmentations with comprehensive error checking"""
    
    print(f"DEBUG: Attempting to save augmentation: {augmentations}")
    
    # Check input validity
    if image is None:
        print(f"ERROR: Image is None for augmentation {augmentations}")
        return False
    if mask is None:
        print(f"ERROR: Mask is None for augmentation {augmentations}")
        return False
    
    # Check image properties
    print(f"DEBUG: Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"DEBUG: Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"DEBUG: Image min/max: {np.min(image)}/{np.max(image)}")
    print(f"DEBUG: Mask min/max: {np.min(mask)}/{np.max(mask)}")
    
    # Process paths
    if ".png" in path_image:
        path_image = path_image[:-4]
        path_mask = path_mask[:-4]
    else:
        print("ERROR: Only supports png filetypes.")
        return False
    
    for aug in augmentations:
        path_image = path_image + "_" + aug
        path_mask = path_mask + "_" + aug

    path_image = path_image + ".png"
    path_mask = path_mask + ".png"

    print(f"DEBUG: Saving to:")
    print(f"  Image: {path_image}")
    print(f"  Mask:  {path_mask}")
    
    # Ensure images are in correct format for saving
    if image.dtype != np.uint8:
        print(f"WARNING: Converting image from {image.dtype} to uint8")
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    if mask.dtype != np.uint8:
        print(f"WARNING: Converting mask from {mask.dtype} to uint8")
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    
    # Try to save image
    try:
        success_img = cv2.imwrite(path_image, image)
        if not success_img:
            print(f"ERROR: cv2.imwrite failed for image {path_image}")
            return False
        else:
            print(f"SUCCESS: Image saved to {path_image}")
    except Exception as e:
        print(f"EXCEPTION: Failed to save image {path_image}: {str(e)}")
        return False
    
    # Try to save mask
    try:
        success_mask = cv2.imwrite(path_mask, mask)
        if not success_mask:
            print(f"ERROR: cv2.imwrite failed for mask {path_mask}")
            return False
        else:
            print(f"SUCCESS: Mask saved to {path_mask}")
    except Exception as e:
        print(f"EXCEPTION: Failed to save mask {path_mask}: {str(e)}")
        return False
    
    print(f"DEBUG: Successfully saved both image and mask for {augmentations}")
    return True

def debug_augmentation_function(func, name, image, mask):
    """Debug wrapper for augmentation functions"""
    print(f"\nDEBUG: Testing {name}")
    print(f"Input image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Input mask shape: {mask.shape}, dtype: {mask.dtype}")
    
    try:
        aug_image, aug_mask, aug_list = func(image.copy(), mask.copy(), [])
        
        print(f"Output image shape: {aug_image.shape}, dtype: {aug_image.dtype}")
        print(f"Output mask shape: {aug_mask.shape}, dtype: {aug_mask.dtype}")
        print(f"Augmentation list: {aug_list}")
        
        # Check for None returns
        if aug_image is None:
            print(f"ERROR: {name} returned None for image")
            return None, None, []
        if aug_mask is None:
            print(f"ERROR: {name} returned None for mask")
            return None, None, []
        
        # Check for shape mismatches
        if aug_image.shape != image.shape:
            print(f"WARNING: {name} changed image shape from {image.shape} to {aug_image.shape}")
        if aug_mask.shape != mask.shape:
            print(f"WARNING: {name} changed mask shape from {mask.shape} to {aug_mask.shape}")
        
        print(f"SUCCESS: {name} completed successfully")
        return aug_image, aug_mask, aug_list
        
    except Exception as e:
        print(f"EXCEPTION: {name} failed with error: {str(e)}")
        return None, None, []

def test_specific_augmentation():
    """Test a specific augmentation to identify the issue"""
    
    # Create test image and mask
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    test_mask = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    print("=== TESTING AUGMENTATION FAILURE ===")
    
    # Test the simplest augmentation first
    from AugmentMaskRCNNTrainingData import flipImage
    
    result_img, result_mask, result_augs = debug_augmentation_function(
        flipImage, "Horizontal Flip", test_image, test_mask
    )
    
    if result_img is not None and result_mask is not None:
        # Test saving
        print("\n=== TESTING SAVE FUNCTION ===")
        success = debug_saveAugmentations(
            "test_image.png", result_img, "test_mask.png", result_mask, result_augs
        )
        
        if success:
            print("✓ Save function works correctly")
            # Check if files actually exist
            if os.path.exists("test_image_hflip.png") and os.path.exists("test_mask_hflip.png"):
                print("✓ Files were actually created on disk")
                # Clean up
                os.remove("test_image_hflip.png")
                os.remove("test_mask_hflip.png")
            else:
                print("✗ Files were not created on disk despite success return")
        else:
            print("✗ Save function failed")
    else:
        print("✗ Augmentation function failed")

if __name__ == "__main__":
    test_specific_augmentation() 