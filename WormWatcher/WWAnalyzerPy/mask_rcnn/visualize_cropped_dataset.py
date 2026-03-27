"""
Visualization script for debugging the cropped dataset.
This script loads the dataset with cropping enabled and displays each crop
with mask overlays to help debug training issues.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Add the current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.realpath('.'))

from Mask_RCNN_Dataset import WormDataset

# Add the utils folder to the path
utils_path = sys.path[0].split("\\")[:-1]
base_path = '\\'.join(utils_path)
utils_path.append('mask_rcnn_utils')
utils_path = '\\'.join(utils_path)
sys.path.append(base_path)
sys.path.append(utils_path)

# Import transforms
import transforms as T

def get_transform():
    """Simple transform that just converts to tensor"""
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def visualize_crop_with_masks(image_tensor, target, crop_idx, save_dir=None):
    """
    Visualize a cropped image with mask overlays
    
    Args:
        image_tensor: PyTorch tensor of the image
        target: Target dictionary containing masks, boxes, labels
        crop_idx: Index of the crop for display purposes
        save_dir: Optional directory to save images instead of displaying
    """
    # Convert tensor to numpy array
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.dim() == 3:  # C, H, W
            image_np = image_tensor.permute(1, 2, 0).numpy()
        else:
            image_np = image_tensor.numpy()
    else:
        image_np = np.array(image_tensor)
    
    # Ensure image is in [0, 255] range
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image_np.astype(np.uint8)
    
    # Create a copy for overlay
    overlay_image = image_np.copy()
    
    # Get masks and boxes
    masks = target['masks'].numpy() if isinstance(target['masks'], torch.Tensor) else target['masks']
    boxes = target['boxes'].numpy() if isinstance(target['boxes'], torch.Tensor) else target['boxes']
    labels = target['labels'].numpy() if isinstance(target['labels'], torch.Tensor) else target['labels']
    
    print(f"\nCrop {crop_idx}:")
    print(f"  Image shape: {image_np.shape}")
    print(f"  Number of objects: {len(masks)}")
    print(f"  Labels: {labels}")
    print(f"  Boxes shape: {boxes.shape if len(boxes) > 0 else 'No boxes'}")
    
    # Set all worm pixels to pure bright green (no transparency)
    for i, mask in enumerate(masks):
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()
        
        # Directly set mask pixels to pure bright green - no blending
        overlay_image[mask > 0] = [0, 255, 0]  # Pure bright green, no transparency
    
    # Display or save the image
    if save_dir:
        save_path = os.path.join(save_dir, f"crop_{crop_idx:04d}.png")
        cv2.imwrite(save_path, cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        print(f"  Saved to: {save_path}")
    else:
        # Display using matplotlib with synchronized zoom/pan
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
        
        # Original image
        ax1.imshow(image_np)
        ax1.set_title(f'Original Crop {crop_idx}', fontsize=12, color='black')
        ax1.axis('on')  # Show axes for better navigation
        ax1.grid(True, alpha=0.3)
        
        # Image with overlays
        ax2.imshow(overlay_image)
        ax2.set_title(f'Crop {crop_idx} with Masks & Boxes\n{len(masks)} objects detected', 
                     fontsize=12, color='green', weight='bold')
        ax2.axis('on')  # Show axes for better navigation
        ax2.grid(True, alpha=0.3)
        
        # Synchronize the view limits
        ax1.callbacks.connect('xlim_changed', lambda *args: ax2.set_xlim(ax1.get_xlim()))
        ax1.callbacks.connect('ylim_changed', lambda *args: ax2.set_ylim(ax1.get_ylim()))
        ax2.callbacks.connect('xlim_changed', lambda *args: ax1.set_xlim(ax2.get_xlim()))
        ax2.callbacks.connect('ylim_changed', lambda *args: ax1.set_ylim(ax2.get_ylim()))
        
        plt.tight_layout()
        plt.show()

def main():
    # Configuration - MODIFY THESE PATHS
    dataset_root = r'I:\Tau worm mask RCNN\Model 4 - worms single plate DF\Augmented'  # Update this path
    
    # Parameters for visualization
    visualize_mode = "display"  # "display" or "save"
    save_directory = "cropped_visualizations"  # Only used if visualize_mode == "save"
    max_crops_to_show = 20  # Maximum number of crops to visualize
    
    print("="*60)
    print("CROPPED DATASET VISUALIZATION")
    print("="*60)
    print(f"Dataset root: {dataset_root}")
    print(f"Mode: {visualize_mode}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_root):
        print(f"ERROR: Dataset root does not exist: {dataset_root}")
        print("Please update the dataset_root path in the script.")
        return
    
    if not os.path.exists(os.path.join(dataset_root, "DatasetImages")):
        print(f"ERROR: DatasetImages folder not found in {dataset_root}")
        return
    
    if not os.path.exists(os.path.join(dataset_root, "DatasetMasks")):
        print(f"ERROR: DatasetMasks folder not found in {dataset_root}")
        return
    
    # Create save directory if needed
    if visualize_mode == "save":
        os.makedirs(save_directory, exist_ok=True)
        print(f"Saving images to: {save_directory}")
    
    try:
        # Import utils (may fail due to import issues, but let's try)
        try:
            import utils as U
            class_params = U.get_worm_params()
            print(f"Using class parameters from utils: {len(class_params)} classes")
        except:
            print("Could not import utils, using single class mode")
            class_params = None
        
        # Create dataset with cropping enabled
        print("\nCreating dataset with cropping enabled...")
        dataset = WormDataset(
            root=dataset_root,
            transforms=get_transform(),
            num_classes=1,  # Single class for simplicity
            class_parameters=class_params,
            use_cropping=True,
            crop_size=(512, 512),
            overlap_ratio=0.1,
            min_objects_per_crop=1,
            max_crops_per_image=16
        )
        
        print(f"\nDataset created successfully!")
        print(f"Total crops available: {len(dataset)}")
        
        if len(dataset) == 0:
            print("ERROR: No crops were generated! Check your dataset and class parameters.")
            return
        
        # Visualize crops
        print(f"\nStarting visualization of first {min(max_crops_to_show, len(dataset))} crops...")
        print("Press Enter to continue to next crop, or 'q' + Enter to quit")
        
        for i in range(min(max_crops_to_show, len(dataset))):
            try:
                # Get the crop
                image, target = dataset[i]
                
                # Visualize
                visualize_crop_with_masks(image, target, i, 
                                        save_directory if visualize_mode == "save" else None)
                
                if visualize_mode == "display":
                    # Wait for user input
                    user_input = input(f"\nShowing crop {i+1}/{min(max_crops_to_show, len(dataset))}. Press Enter for next, 'q' to quit: ")
                    if user_input.lower() == 'q':
                        break
                
            except Exception as e:
                print(f"Error processing crop {i}: {str(e)}")
                continue
        
        print("\nVisualization complete!")
        
    except Exception as e:
        print(f"ERROR: Failed to create dataset: {str(e)}")
        print("\nThis might be due to:")
        print("1. Incorrect dataset path")
        print("2. Missing or empty DatasetImages/DatasetMasks folders")
        print("3. Import issues with utils module")
        print("4. Incompatible image/mask files")

if __name__ == "__main__":
    main() 