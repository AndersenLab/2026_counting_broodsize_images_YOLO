"""
Test script for visualizing data augmentations
Press any key to see new random augmentations
Press 'q' or ESC to quit
"""

import cv2
import numpy as np
import os
import sys
import json
from pathlib import Path
from typing import Tuple, Optional, List, Dict

try:
    from .data_augmentation import DataAugmentation
except ImportError:
    from data_augmentation import DataAugmentation

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================
CONFIG = {
    "image_path": r"I:\Tau worm mask RCNN\Model 4c - combinaed 4a+4b\Original\DatasetImages\BRC20241_A1_D1_R8 Parmar Nikita_tile_0056_460_2300.png",  # Path to test image
    "vgg_json_path": r"I:\Tau worm mask RCNN\Model 4c - combinaed 4a+4b\Original\merged_labels_20251207_234814.json",  # Path to VGG JSON with annotations
}
# ============================================================================


def load_vgg_json_masks(vgg_json_path: str, image_filename: str, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Load mask from VGG JSON annotations for a specific image.
    
    Args:
        vgg_json_path: Path to VGG JSON file
        image_filename: Filename of the image (must match JSON entry)
        image_shape: Shape of the image (height, width)
    
    Returns:
        Combined mask array or None if not found
    """
    if not os.path.exists(vgg_json_path):
        return None
    
    with open(vgg_json_path, 'r') as f:
        vgg_data = json.load(f)
    
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Parse VGG JSON structure
    img_metadata = None
    if isinstance(vgg_data, dict):
        if '_via_img_metadata' in vgg_data:
            img_metadata = vgg_data['_via_img_metadata']
        else:
            img_metadata = vgg_data
    elif isinstance(vgg_data, list):
        # List format
        for item in vgg_data:
            if isinstance(item, dict) and item.get('filename') == image_filename:
                img_metadata = {image_filename: item}
                break
    
    if not img_metadata:
        return None
    
    # Find the image entry
    image_entry = None
    for img_id, img_data in img_metadata.items():
        filename = img_data.get('filename') or img_data.get('fname')
        if filename == image_filename or img_id == image_filename:
            image_entry = img_data
            break
    
    if not image_entry:
        return None
    
    # Extract regions/annotations
    regions = image_entry.get('regions') or image_entry.get('via_region_data', {})
    
    if isinstance(regions, list):
        region_list = regions
    elif isinstance(regions, dict):
        region_list = list(regions.values())
    else:
        return None
    
    # Draw each region as a separate object (different pixel value)
    object_id = 1
    for region in region_list:
        shape_attr = region.get('shape_attributes') or region
        
        # Parse polygon
        if 'all_points_x' in shape_attr and 'all_points_y' in shape_attr:
            all_points_x = shape_attr['all_points_x']
            all_points_y = shape_attr['all_points_y']
            
            if len(all_points_x) >= 3 and len(all_points_y) >= 3:
                # Create polygon points
                pts = np.array([[x, y] for x, y in zip(all_points_x, all_points_y)], dtype=np.int32)
                # Fill polygon with object ID
                cv2.fillPoly(mask, [pts], object_id)
                object_id += 1
        elif shape_attr.get('name') == 'rect':
            # Rectangle
            x = int(shape_attr.get('x', 0))
            y = int(shape_attr.get('y', 0))
            width = int(shape_attr.get('width', 0))
            height = int(shape_attr.get('height', 0))
            if width > 0 and height > 0:
                mask[y:y+height, x:x+width] = object_id
                object_id += 1
        elif shape_attr.get('name') == 'circle':
            # Circle
            cx = int(shape_attr.get('cx', 0))
            cy = int(shape_attr.get('cy', 0))
            r = int(shape_attr.get('r', 0))
            if r > 0:
                cv2.circle(mask, (cx, cy), r, object_id, -1)
                object_id += 1
    
    return mask if object_id > 1 else None


def draw_mask_outlines_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Draw mask outlines on grayscale image.
    
    Args:
        image: Input image (BGR)
        mask: Mask array (grayscale, 0=background, >0=objects)
    
    Returns:
        Grayscale image with green mask outlines
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Convert to BGR for colored outlines
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Find contours for each object in mask
    unique_ids = np.unique(mask)
    for obj_id in unique_ids:
        if obj_id == 0:  # Skip background
            continue
        
        # Create binary mask for this object
        obj_mask = (mask == obj_id).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw green outlines (thin)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 1)
    
    return result


def visualize_augmentations(image: np.ndarray, mask: Optional[np.ndarray], 
                           aug: DataAugmentation, num_samples: int = 4):
    """
    Visualize multiple augmented versions side by side with masks as green outlines.
    
    Args:
        image: Original image
        mask: Original mask (optional)
        aug: DataAugmentation instance
        num_samples: Number of augmented samples to show
    """
    # Create display image
    h, w = image.shape[:2]
    display_width = w * (num_samples + 1)  # +1 for original
    display_height = h
    
    display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
    
    # Place original image with mask outlines
    if mask is not None:
        original_display = draw_mask_outlines_on_image(image, mask)
    else:
        # Convert to grayscale
        if len(image.shape) == 3:
            original_display = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            original_display = cv2.cvtColor(original_display, cv2.COLOR_GRAY2BGR)
        else:
            original_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    display[0:h, 0:w] = original_display
    cv2.putText(display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Generate and place augmented samples with mask outlines
    for i in range(num_samples):
        x_offset = w * (i + 1)
        
        if mask is not None:
            aug_img, aug_mask, aug_name = aug.apply_random_augmentation(image, mask)
            
            # Draw mask outlines on augmented image
            aug_display = draw_mask_outlines_on_image(aug_img, aug_mask)
            display[0:h, x_offset:x_offset+w] = aug_display
            cv2.putText(display, aug_name, (x_offset + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # No mask - just augment image and convert to grayscale
            aug_img, _, aug_name = aug.apply_random_augmentation(image, image)  # Use image as dummy mask
            if len(aug_img.shape) == 3:
                aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
                aug_display = cv2.cvtColor(aug_gray, cv2.COLOR_GRAY2BGR)
            else:
                aug_display = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR)
            display[0:h, x_offset:x_offset+w] = aug_display
            cv2.putText(display, aug_name, (x_offset + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return display


def main():
    """Main visualization loop"""
    print("=" * 80)
    print("Data Augmentation Visualization Tool")
    print("=" * 80)
    
    # Load paths from CONFIG
    image_path = CONFIG["image_path"]
    vgg_json_path = CONFIG["vgg_json_path"]
    
    # Validate paths
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("   Please update CONFIG['image_path'] at the top of this script")
        return
    
    if not os.path.exists(vgg_json_path):
        print(f"❌ VGG JSON not found: {vgg_json_path}")
        print("   Please update CONFIG['vgg_json_path'] at the top of this script")
        return
    
    print(f"✅ Loading image: {image_path}")
    print(f"✅ Loading masks from: {vgg_json_path}")
    
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Extract image filename for VGG JSON lookup
    image_filename = os.path.basename(image_path)
    
    # Load mask from VGG JSON
    mask = load_vgg_json_masks(vgg_json_path, image_filename, image.shape[:2])
    
    if mask is not None:
        print(f"✅ Mask loaded from VGG JSON: {mask.shape}")
        print(f"   Found {len(np.unique(mask)) - 1} objects in mask")
    else:
        print("⚠️  No mask found in VGG JSON for this image")
        print("   Will show image augmentations only")
        mask = None
    
    # Create augmentation instance with default settings
    aug = DataAugmentation(
        apply_flips=True,
        apply_blur=True,
        apply_noise=True,
        apply_brightness=True,
        apply_rotation=True,
        apply_scaling=True,
        apply_translation=True,
        apply_contrast=True,
        apply_shear=True,
        apply_elastic=False,  # Disabled by default (expensive)
    )
    
    print("\n" + "=" * 80)
    print("Visualization Controls:")
    print("  - Press ANY KEY to generate new random augmentations")
    print("  - Press 'q' or ESC to quit")
    print("  - Press 'c' to show combination augmentations")
    print("=" * 80 + "\n")
    
    # Initial display
    num_samples = 4
    use_combinations = False
    
    while True:
        if use_combinations:
            # Show combination augmentations with mask outlines
            display = np.zeros((image.shape[0], image.shape[1] * (num_samples + 1), 3), dtype=np.uint8)
            
            # Original with mask outlines
            if mask is not None:
                original_display = draw_mask_outlines_on_image(image, mask)
            else:
                # Convert to grayscale
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    original_display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                else:
                    original_display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            display[0:image.shape[0], 0:image.shape[1]] = original_display
            cv2.putText(display, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            for i in range(num_samples):
                x_offset = image.shape[1] * (i + 1)
                if mask is not None:
                    aug_img, aug_mask, aug_names = aug.apply_combination(image, mask, num_augmentations=2)
                    aug_display = draw_mask_outlines_on_image(aug_img, aug_mask)
                    display[0:image.shape[0], x_offset:x_offset+image.shape[1]] = aug_display
                    aug_name_str = "+".join(aug_names) if aug_names else "none"
                    cv2.putText(display, aug_name_str[:20], (x_offset + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    aug_img, _, aug_names = aug.apply_combination(image, image, num_augmentations=2)
                    # Convert to grayscale
                    if len(aug_img.shape) == 3:
                        aug_gray = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
                        aug_display = cv2.cvtColor(aug_gray, cv2.COLOR_GRAY2BGR)
                    else:
                        aug_display = cv2.cvtColor(aug_img, cv2.COLOR_GRAY2BGR)
                    display[0:image.shape[0], x_offset:x_offset+image.shape[1]] = aug_display
                    aug_name_str = "+".join(aug_names) if aug_names else "none"
                    cv2.putText(display, aug_name_str[:20], (x_offset + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            # Show single augmentations
            display = visualize_augmentations(image, mask, aug, num_samples)
        
        # Show display
        cv2.imshow("Data Augmentation Visualization", display)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('c'):
            use_combinations = not use_combinations
            print(f"🔄 {'Combination' if use_combinations else 'Single'} augmentation mode")
        elif key == ord('+') or key == ord('='):
            num_samples = min(num_samples + 1, 8)
            print(f"📊 Showing {num_samples} augmented samples")
        elif key == ord('-'):
            num_samples = max(num_samples - 1, 2)
            print(f"📊 Showing {num_samples} augmented samples")
    
    cv2.destroyAllWindows()
    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()

