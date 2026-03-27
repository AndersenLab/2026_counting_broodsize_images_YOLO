#!/usr/bin/env python3
"""
Generate Background Images for Mask-RCNN Training

This script generates background-only images by filtering out annotated worms
from image/mask pairs and patching them together.

Usage:
    python generate_maskrcnn_backgrounds.py

Author: AI Assistant
Date: 2024
"""

import os
import sys
import numpy as np
import cv2
from pathlib import Path

# ============================================================================
# CONFIGURATION - Hardcoded paths
# ============================================================================

# Debug mode - when True, shows OpenCV images for inspection
DEBUG = True

# Number of background images to show in debug mode
DEBUG_NUM_IMAGES = 3

# Folder containing DatasetImages and DatasetMasks subfolders
DATASET_FOLDER = r"I:\Tau worm mask RCNN\Model 4c - combinaed 4a+4b\Augmented"

# Number of image pairs to select for background generation
NUM_IMAGE_PAIRS = 5

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def load_image_mask_pair(images_dir, masks_dir, image_filename):
    """Load an image and its corresponding mask file"""
    image_path = os.path.join(images_dir, image_filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Find corresponding mask file by searching through all mask files
    # Mask files have "_mask" suffix: image "name.ext" -> mask "name_mask.ext"
    # Match by removing "_mask" from mask filename and comparing with image base name
    img_base = os.path.splitext(image_filename)[0]
    
    # Get all mask files
    mask_files = [f for f in os.listdir(masks_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    # Search for matching mask
    mask_candidates = []
    for mask_file in mask_files:
        mask_base = os.path.splitext(mask_file)[0]
        # Remove _mask from mask name and check if it matches image name
        mask_without_suffix = mask_base.replace('_mask', '')
        if mask_without_suffix == img_base:
            mask_candidates.append(mask_file)
    
    if not mask_candidates:
        raise FileNotFoundError(
            f"Mask not found for image: {image_filename}\n"
            f"   Image base name: {img_base}\n"
            f"   Searched in: {masks_dir}\n"
            f"   Found {len(mask_files)} mask files total"
        )
    
    # Use the first matching mask (if multiple, take first)
    mask_filename = mask_candidates[0]
    mask_path = os.path.join(masks_dir, mask_filename)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    return image, mask


def create_background_only_image(image, mask):
    """Filter out annotated worms (where mask > 0) to create background-only image"""
    # Create a copy of the image
    background = image.copy()
    
    # Set pixels where mask > 0 to a placeholder value (we'll patch over these)
    # For now, we'll use NaN or a sentinel value, but since we're patching,
    # we can just mark them
    worm_mask = mask > 0
    
    # Create background-only image by setting worm regions to 0 (will be patched later)
    background[worm_mask] = 0
    
    return background, worm_mask


def patch_background_images(background_images, worm_masks):
    """Patch multiple background images together to fill in worm regions"""
    if len(background_images) == 0:
        raise ValueError("No background images provided")
    
    # Use the first image as the base
    base_image = background_images[0].copy()
    base_mask = worm_masks[0].copy()
    
    # For each subsequent image, patch over worm regions in the base
    for i in range(1, len(background_images)):
        bg_img = background_images[i]
        bg_mask = worm_masks[i]
        
        # Find regions in base that need patching (where base has worms)
        # and where the current image doesn't have worms
        needs_patch = base_mask
        has_background = ~bg_mask
        
        # Patch regions: where base needs patching AND current image has background
        patch_regions = needs_patch & has_background
        
        # Copy background pixels from current image to base
        base_image[patch_regions] = bg_img[patch_regions]
        base_mask[patch_regions] = False  # Mark as patched
    
    return base_image, base_mask


def show_debug_images(background_images, final_image=None):
    """Display images using OpenCV for debugging"""
    print(f"\n🔍 DEBUG MODE: Showing first {min(DEBUG_NUM_IMAGES, len(background_images))} background images...")
    
    for i in range(min(DEBUG_NUM_IMAGES, len(background_images))):
        img = background_images[i]
        
        # Resize if too large for display (max 800px width)
        display_img = img.copy()
        if display_img.shape[1] > 800:
            scale = 800 / display_img.shape[1]
            new_width = int(display_img.shape[1] * scale)
            new_height = int(display_img.shape[0] * scale)
            display_img = cv2.resize(display_img, (new_width, new_height))
        
        window_name = f"Background Image {i+1}"
        cv2.imshow(window_name, display_img)
        print(f"   Showing {window_name} (press any key to continue)")
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
    
    # Show final patched image if provided
    if final_image is not None:
        display_img = final_image.copy()
        if display_img.shape[1] > 800:
            scale = 800 / display_img.shape[1]
            new_width = int(display_img.shape[1] * scale)
            new_height = int(display_img.shape[0] * scale)
            display_img = cv2.resize(display_img, (new_width, new_height))
        
        cv2.imshow("Final Patched Background", display_img)
        print(f"   Showing Final Patched Background (press any key to continue)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    """Main execution function"""
    print("🖼️  Background Image Generator for Mask-RCNN")
    print("=" * 60)
    
    # Validate dataset folder structure
    images_dir = os.path.join(DATASET_FOLDER, "DatasetImages")
    masks_dir = os.path.join(DATASET_FOLDER, "DatasetMasks")
    
    if not os.path.exists(images_dir):
        raise ValueError(f"DatasetImages directory not found: {images_dir}")
    if not os.path.exists(masks_dir):
        raise ValueError(f"DatasetMasks directory not found: {masks_dir}")
    
    # Find image files
    image_files = sorted([f for f in os.listdir(images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    
    print(f"📁 Found {len(image_files)} images in dataset")
    print(f"   Images directory: {images_dir}")
    print(f"   Masks directory: {masks_dir}")
    
    # Select image pairs
    num_to_select = min(NUM_IMAGE_PAIRS, len(image_files))
    selected_files = image_files[:num_to_select]
    
    print(f"\n📋 Processing {num_to_select} image pairs...")
    
    # Load images and create background-only versions
    background_images = []
    worm_masks = []
    
    for i, image_file in enumerate(selected_files):
        print(f"   [{i+1}/{num_to_select}] Processing {image_file}...")
        
        image, mask = load_image_mask_pair(images_dir, masks_dir, image_file)
        background, worm_mask = create_background_only_image(image, mask)
        
        background_images.append(background)
        worm_masks.append(worm_mask)
        
        print(f"      Image shape: {image.shape}, Mask shape: {mask.shape}")
        print(f"      Worm pixels: {np.sum(worm_mask)} ({100*np.sum(worm_mask)/worm_mask.size:.1f}%)")
    
    if len(background_images) == 0:
        raise ValueError("No background images were successfully created")
    
    print(f"\n✅ Created {len(background_images)} background-only images")
    
    # Patch images together
    print(f"\n🔧 Patching background images together...")
    final_background, remaining_mask = patch_background_images(background_images, worm_masks)
    
    remaining_worm_pixels = np.sum(remaining_mask)
    total_pixels = remaining_mask.size
    print(f"   Remaining worm pixels after patching: {remaining_worm_pixels} ({100*remaining_worm_pixels/total_pixels:.1f}%)")
    
    # Debug mode: show images
    if DEBUG:
        show_debug_images(background_images, final_background)
    
    print(f"\n✅ Background generation complete!")
    print(f"   Final image shape: {final_background.shape}")
    
    return final_background, background_images


if __name__ == "__main__":
    main()

