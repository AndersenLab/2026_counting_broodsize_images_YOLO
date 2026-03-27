"""
Custom YOLO Dataset with On-the-Fly Data Augmentation
Wraps YOLO's dataset to apply DataAugmentation class augmentations during training
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from .data_augmentation import DataAugmentation
except ImportError:
    from data_augmentation import DataAugmentation


def yolo_polygons_to_mask(polygons: List[List[float]], img_width: int, img_height: int) -> np.ndarray:
    """
    Convert YOLO format polygons (normalized coordinates) to binary mask.
    
    Args:
        polygons: List of polygons, each as list of [x1, y1, x2, y2, ...] normalized coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        Binary mask array (uint8, 0=background, >0=objects)
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    for obj_id, polygon in enumerate(polygons, start=1):
        # Convert normalized coordinates to pixel coordinates
        points = []
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x_norm = polygon[i]
                y_norm = polygon[i + 1]
                x = int(x_norm * img_width)
                y = int(y_norm * img_height)
                points.append([x, y])
        
        if len(points) >= 3:
            # Draw filled polygon
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], obj_id)
    
    return mask


def mask_to_yolo_polygons(mask: np.ndarray) -> List[List[float]]:
    """
    Convert mask to YOLO format polygons (normalized coordinates).
    
    Args:
        mask: Binary mask array (uint8, 0=background, >0=objects)
    
    Returns:
        List of polygons, each as list of normalized [x1, y1, x2, y2, ...] coordinates
    """
    h, w = mask.shape
    polygons = []
    
    # Find contours for each object
    unique_ids = np.unique(mask)
    for obj_id in unique_ids:
        if obj_id == 0:  # Skip background
            continue
        
        # Create binary mask for this object
        obj_mask = (mask == obj_id).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert each contour to normalized polygon
        for contour in contours:
            if len(contour) >= 3:
                # Normalize coordinates
                polygon = []
                for point in contour:
                    x_norm = point[0][0] / w
                    y_norm = point[0][1] / h
                    # Clamp to [0, 1]
                    x_norm = max(0.0, min(1.0, x_norm))
                    y_norm = max(0.0, min(1.0, y_norm))
                    polygon.extend([x_norm, y_norm])
                
                if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                    polygons.append(polygon)
    
    return polygons


class AugmentedYOLODataset:
    """
    Wrapper for YOLO dataset that applies custom DataAugmentation on-the-fly.
    
    This class wraps YOLO's internal dataset and applies augmentations from
    DataAugmentation class before returning samples.
    """
    
    def __init__(self, yolo_dataset, augmentation: Optional[DataAugmentation] = None, 
                 apply_augmentation: bool = True):
        """
        Initialize augmented dataset wrapper.
        
        Args:
            yolo_dataset: YOLO's internal dataset object
            augmentation: DataAugmentation instance (None to create default)
            apply_augmentation: Whether to apply augmentations (False for validation)
        """
        self.yolo_dataset = yolo_dataset
        self.apply_augmentation = apply_augmentation
        
        if augmentation is None and apply_augmentation:
            # Create default augmentation instance
            self.augmentation = DataAugmentation(
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
        else:
            self.augmentation = augmentation
    
    def __len__(self):
        return len(self.yolo_dataset)
    
    def __getitem__(self, idx):
        """
        Get item with optional augmentation.
        
        Returns:
            Same format as YOLO dataset (dict with 'img', 'labels', etc.)
        """
        # Get original sample from YOLO dataset
        sample = self.yolo_dataset[idx]
        
        if not self.apply_augmentation:
            return sample
        
        # Track that augmentation is being used (only log first few times to avoid spam)
        if not hasattr(self, '_augmentation_logged'):
            self._augmentation_logged = True
            print(f"✅ AugmentedYOLODataset.__getitem__() called - augmentation is active!")
        elif idx < 3:  # Log first 3 samples
            print(f"✅ AugmentedYOLODataset processing sample {idx}")
        
        # Extract image and labels
        img = sample['img']  # This is already a numpy array or tensor
        labels = sample.get('labels', None)  # Shape: [N, 6] where N is num objects, columns: [class, x1, y1, x2, y2, ...]
        
        # Convert to numpy if needed
        if hasattr(img, 'numpy'):
            img_np = img.numpy()
        elif hasattr(img, 'cpu'):
            img_np = img.cpu().numpy()
        else:
            img_np = np.array(img)
        
        # Handle different image formats
        if len(img_np.shape) == 3 and img_np.shape[0] == 3:
            # CHW format -> HWC
            img_np = img_np.transpose(1, 2, 0)
        
        # Normalize from [0, 1] to [0, 255] if needed
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
        
        # Convert BGR to RGB if needed (YOLO might use RGB)
        if img_np.shape[2] == 3:
            # Assume BGR, convert to RGB for consistency
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        
        h, w = img_np.shape[:2]
        
        # Detect label format: detection (5 values: class, x_center, y_center, width, height) 
        # vs segmentation (class + polygon points)
        is_detection_mode = False
        if labels is not None and len(labels) > 0:
            # Check first label to determine format
            first_label = labels[0]
            if len(first_label) == 5:
                # Detection format: [class_id, x_center, y_center, width, height]
                is_detection_mode = True
        
        # Convert labels to mask for augmentation
        if labels is not None and len(labels) > 0:
            if is_detection_mode:
                # Detection mode: convert bounding boxes to mask
                mask = np.zeros((h, w), dtype=np.uint8)
                for obj_id, label in enumerate(labels, start=1):
                    if len(label) == 5:
                        class_id, x_center_norm, y_center_norm, width_norm, height_norm = label
                        # Convert normalized bbox to pixel coordinates
                        x_center = int(x_center_norm * w)
                        y_center = int(y_center_norm * h)
                        bbox_width = int(width_norm * w)
                        bbox_height = int(height_norm * h)
                        # Calculate bbox corners
                        x1 = max(0, x_center - bbox_width // 2)
                        y1 = max(0, y_center - bbox_height // 2)
                        x2 = min(w, x_center + bbox_width // 2)
                        y2 = min(h, y_center + bbox_height // 2)
                        # Draw rectangle as mask
                        mask[y1:y2, x1:x2] = obj_id
            else:
                # Segmentation mode: convert polygons to mask
                polygons = []
                for label in labels:
                    if len(label) >= 6:  # Has class + at least 2 points
                        # Skip class ID (first element), get polygon coordinates
                        polygon = label[1:].tolist()
                        polygons.append(polygon)
                
                if polygons:
                    mask = yolo_polygons_to_mask(polygons, w, h)
                else:
                    mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
        
        # Apply augmentation
        aug_img, aug_mask, _ = self.augmentation.apply_random_augmentation(img_np, mask)
        
        # Convert augmented mask back to labels
        if labels is not None and len(labels) > 0:
            if is_detection_mode:
                # Detection mode: convert mask back to bounding boxes
                aug_labels = []
                unique_ids = np.unique(aug_mask)
                unique_ids = unique_ids[unique_ids != 0]  # Remove background
                
                for i, obj_id in enumerate(unique_ids):
                    if i < len(labels):
                        class_id = labels[i][0]  # Preserve class ID
                        # Find bounding box of this object in mask
                        obj_mask = (aug_mask == obj_id)
                        if np.any(obj_mask):
                            rows = np.where(obj_mask)[0]
                            cols = np.where(obj_mask)[1]
                            y_min, y_max = rows.min(), rows.max()
                            x_min, x_max = cols.min(), cols.max()
                            # Convert to normalized center + width/height format
                            x_center_norm = ((x_min + x_max) / 2.0) / w
                            y_center_norm = ((y_min + y_max) / 2.0) / h
                            width_norm = (x_max - x_min) / w
                            height_norm = (y_max - y_min) / h
                            # Clamp to [0, 1]
                            x_center_norm = max(0, min(1.0, x_center_norm))
                            y_center_norm = max(0, min(1.0, y_center_norm))
                            width_norm = max(0, min(1.0, width_norm))
                            height_norm = max(0, min(1.0, height_norm))
                            aug_labels.append([class_id, x_center_norm, y_center_norm, width_norm, height_norm])
            else:
                # Segmentation mode: convert mask back to polygons
                aug_polygons = mask_to_yolo_polygons(aug_mask)
                
                # Reconstruct labels with augmented polygons
                aug_labels = []
                for i, polygon in enumerate(aug_polygons):
                    if i < len(labels):
                        class_id = labels[i][0]  # Preserve class ID
                        aug_label = [class_id] + polygon
                        aug_labels.append(aug_label)
            
            # Update sample
            if aug_labels:
                sample['labels'] = np.array(aug_labels, dtype=np.float32)
            else:
                # No valid labels after augmentation - keep original to avoid breaking training
                pass
        else:
            # No labels to augment
            pass
        
        # Convert augmented image back to format expected by YOLO
        # RGB -> BGR
        aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
        
        # Normalize to [0, 1]
        aug_img_normalized = aug_img_bgr.astype(np.float32) / 255.0
        
        # HWC -> CHW
        aug_img_chw = aug_img_normalized.transpose(2, 0, 1)
        
        # Update sample image
        sample['img'] = aug_img_chw
        
        return sample

