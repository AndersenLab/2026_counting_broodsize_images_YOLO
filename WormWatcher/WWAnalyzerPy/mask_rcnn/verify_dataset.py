"""
Verification script to check the 1:1 relationship between images and masks.
This script helps identify any mismatches in your dataset.
"""

import os
import sys
from verifyDirlist import *

def verify_dataset(image_folder, mask_folder):
    """Verify that images and masks have a 1:1 relationship"""
    
    print(f"Verifying dataset:")
    print(f"Images folder: {image_folder}")
    print(f"Masks folder: {mask_folder}")
    print("-" * 50)
    
    # Get lists of files
    image_paths, image_names = verifyDirlist(image_folder, False, ".png")
    mask_paths, mask_names = verifyDirlist(mask_folder, False, ".png")
    
    print(f"Found {len(image_paths)} images")
    print(f"Found {len(mask_paths)} masks")
    
    # Check if counts match
    if len(image_paths) != len(mask_paths):
        print(f"❌ MISMATCH: {len(image_paths)} images vs {len(mask_paths)} masks")
        return False
    
    # Check if filenames match (without extensions)
    image_basenames = [os.path.splitext(name)[0] for name in image_names]
    mask_basenames = [os.path.splitext(name)[0] for name in mask_names]
    
    # Sort both lists to ensure they're in the same order
    image_basenames.sort()
    mask_basenames.sort()
    
    mismatches = []
    for i, (img_name, mask_name) in enumerate(zip(image_basenames, mask_basenames)):
        if img_name != mask_name:
            mismatches.append((img_name, mask_name))
    
    if mismatches:
        print(f"❌ Found {len(mismatches)} filename mismatches:")
        for img_name, mask_name in mismatches[:10]:  # Show first 10
            print(f"   Image: {img_name} | Mask: {mask_name}")
        if len(mismatches) > 10:
            print(f"   ... and {len(mismatches) - 10} more")
        return False
    
    print("✅ All images have corresponding masks with matching names")
    return True

def count_augmented_files(folder):
    """Count files with augmentation suffixes"""
    _, filenames = verifyDirlist(folder, False, ".png")
    
    original_files = []
    augmented_files = []
    
    for filename in filenames:
        basename = os.path.splitext(filename)[0]
        if any(suffix in basename for suffix in ['_hflip', '_vflip', '_gblur', '_rnoise', 
                                                '_brighten', '_darken', '_rotate', '_scale', 
                                                '_contrast', '_gaussian_noise', '_salt_pepper',
                                                '_elastic', '_shadow', '_motion_blur', 
                                                '_intensity_variation', '_perspective']):
            augmented_files.append(filename)
        else:
            original_files.append(filename)
    
    return len(original_files), len(augmented_files)

def main():
    """Main verification function"""
    
    # Get folder paths
    if len(sys.argv) > 2:
        image_folder = sys.argv[1]
        mask_folder = sys.argv[2]
    else:
        # Default paths
        image_folder = input("Enter path to DatasetImages folder: ").strip()
        mask_folder = image_folder.replace("DatasetImages", "DatasetMasks")
    
    print("=" * 60)
    print("DATASET VERIFICATION TOOL")
    print("=" * 60)
    
    # Verify original dataset
    print("\n1. VERIFYING ORIGINAL DATASET")
    original_valid = verify_dataset(image_folder, mask_folder)
    
    if not original_valid:
        print("\n❌ Original dataset has issues. Please fix before running augmentation.")
        return
    
    # Count augmented files
    print("\n2. COUNTING AUGMENTED FILES")
    orig_images, aug_images = count_augmented_files(image_folder)
    orig_masks, aug_masks = count_augmented_files(mask_folder)
    
    print(f"Original images: {orig_images}")
    print(f"Augmented images: {aug_images}")
    print(f"Total images: {orig_images + aug_images}")
    print()
    print(f"Original masks: {orig_masks}")
    print(f"Augmented masks: {aug_masks}")
    print(f"Total masks: {orig_masks + aug_masks}")
    
    # Check augmented file balance
    if aug_images != aug_masks:
        print(f"\n❌ AUGMENTATION MISMATCH: {aug_images} augmented images vs {aug_masks} augmented masks")
        print("This suggests the augmentation process didn't maintain 1:1 relationship.")
    else:
        print(f"\n✅ Augmented files are balanced: {aug_images} images and {aug_masks} masks")
    
    # Check total balance
    total_images = orig_images + aug_images
    total_masks = orig_masks + aug_masks
    
    if total_images != total_masks:
        print(f"\n❌ TOTAL MISMATCH: {total_images} total images vs {total_masks} total masks")
    else:
        print(f"\n✅ Perfect 1:1 relationship maintained: {total_images} total files")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")

if __name__ == "__main__":
    main() 