"""
Dataset Converter: VGG JSON and CVAT XML to YOLO Format
Converts VGG Image Annotator (VIA) JSON format and CVAT XML format to YOLO segmentation format

Supported formats:
- VGG JSON: Polygon, rectangle, and circle annotations
- CVAT XML: Bounding box annotations (converted to rectangular polygons)

Note: CVAT XML bounding boxes are converted to rectangular polygons (4 points),
which provides rectangular mask approximations. For better segmentation accuracy,
consider using polygon/mask annotations instead of bounding boxes.
"""

import json
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random

try:
    from .data_augmentation import DataAugmentation
except ImportError:
    from data_augmentation import DataAugmentation

# Debug flag - set to True to enable debug printouts
DEBUG = False


def _yolo_annotations_to_mask(annotations: List[Dict], img_width: int, img_height: int, task: str) -> np.ndarray:
    """
    Convert YOLO format annotations to a mask for augmentation.
    
    Args:
        annotations: List of annotation dicts with either 'polygon' (seg) or 'x_center', 'y_center', 'width', 'height' (detect)
        img_width: Image width in pixels
        img_height: Image height in pixels
        task: Task mode - "seg" or "detect"
    
    Returns:
        Mask array (uint8, 0=background, >0=objects)
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    if DEBUG:
        print(f"DEBUG: _yolo_annotations_to_mask: input_annotations={len(annotations)}, task={task}, img_size={img_width}x{img_height}")
    
    for obj_id, ann in enumerate(annotations, start=1):
        if task == "detect":
            # Detection mode: convert bounding box to rectangle in mask
            x_center = int(ann['x_center'] * img_width)
            y_center = int(ann['y_center'] * img_height)
            bbox_width = int(ann['width'] * img_width)
            bbox_height = int(ann['height'] * img_height)
            
            x1 = max(0, x_center - bbox_width // 2)
            y1 = max(0, y_center - bbox_height // 2)
            x2 = min(img_width, x_center + bbox_width // 2)
            y2 = min(img_height, y_center + bbox_height // 2)
            
            mask[y1:y2, x1:x2] = obj_id
        else:
            # Segmentation mode: convert polygon to mask
            polygon = ann['polygon']
            points = []
            for x_norm, y_norm in polygon:
                x = int(x_norm * img_width)
                y = int(y_norm * img_height)
                points.append([x, y])
            
            if len(points) >= 3:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], obj_id)
    
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]
    if DEBUG:
        print(f"DEBUG: _yolo_annotations_to_mask: output_mask has {len(unique_ids)} objects")
    
    return mask


def _mask_to_yolo_annotations(mask: np.ndarray, annotations: List[Dict], task: str) -> List[Dict]:
    """
    Convert augmented mask back to YOLO format annotations.
    
    Args:
        mask: Augmented mask array (uint8, 0=background, >0=objects)
        annotations: Original annotations (for preserving class_id)
        task: Task mode - "seg" or "detect"
    
    Returns:
        List of annotation dicts in YOLO format
    """
    h, w = mask.shape
    augmented_annotations = []
    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]  # Remove background
    
    if DEBUG:
        print(f"DEBUG: _mask_to_yolo_annotations: input_annotations={len(annotations)}, unique_mask_ids={len(unique_ids)}, mask_shape={mask.shape}")
    
    for idx, obj_id in enumerate(unique_ids):
        if idx < len(annotations):
            class_id = annotations[idx]['class_id']
            obj_mask = (mask == obj_id)
            
            if not np.any(obj_mask):
                if DEBUG:
                    print(f"DEBUG: _mask_to_yolo_annotations: skipping empty mask for obj_id={obj_id}, idx={idx}")
                continue  # Skip empty masks
            
            if task == "detect":
                # Detection mode: convert mask to bounding box
                rows = np.where(obj_mask)[0]
                cols = np.where(obj_mask)[1]
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()
                
                x_center = ((x_min + x_max) / 2.0) / w
                y_center = ((y_min + y_max) / 2.0) / h
                width = (x_max - x_min) / w
                height = (y_max - y_min) / h
                
                # Clamp to [0, 1]
                x_center = max(0, min(1.0, x_center))
                y_center = max(0, min(1.0, y_center))
                width = max(0, min(1.0, width))
                height = max(0, min(1.0, height))
                
                augmented_annotations.append({
                    'class_id': class_id,
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height
                })
            else:
                # Segmentation mode: convert mask to polygon
                obj_mask_binary = (obj_mask).astype(np.uint8) * 255
                contours, _ = cv2.findContours(obj_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Use the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    if len(largest_contour) >= 3:
                        polygon = []
                        for point in largest_contour:
                            x_norm = point[0][0] / w
                            y_norm = point[0][1] / h
                            x_norm = max(0.0, min(1.0, x_norm))
                            y_norm = max(0.0, min(1.0, y_norm))
                            polygon.append((x_norm, y_norm))
                        
                        if len(polygon) >= 3:
                            augmented_annotations.append({
                                'class_id': class_id,
                                'polygon': polygon
                            })
    
    if DEBUG:
        print(f"DEBUG: _mask_to_yolo_annotations: output_annotations={len(augmented_annotations)}")
    
    return augmented_annotations


def convert_vgg_to_yolo(vgg_json_path: str, images_dir: str, output_dir: str, class_name: str = "worm", task: str = "seg", augmentation_per_image: int = 0):
    """
    Convert VGG JSON annotations to YOLO format.
    
    Args:
        vgg_json_path: Path to VGG JSON annotation file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO format (will create 'images' and 'labels' subdirectories)
        class_name: Name of the class (default: "worm")
        task: Task mode - "seg" for segmentation (polygon format) or "detect" for detection (bounding box format)
        augmentation_per_image: Number of augmented versions to create per original image (0 = no augmentation)
    
    Returns:
        Dictionary with conversion statistics
    """
    # Create output directories
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Load VGG JSON
    with open(vgg_json_path, 'r') as f:
        vgg_data = json.load(f)
    
    # Create class mapping (single class: worm = class 0)
    class_mapping = {class_name: 0}
    
    # Statistics
    stats = {
        'total_images': 0,
        'converted_images': 0,
        'augmented_images': 0,
        'skipped_images': 0,
        'total_annotations': 0,
        'errors': []
    }
    
    # Create augmentation instance if needed
    augmentation = None
    if augmentation_per_image > 0:
        augmentation = DataAugmentation(
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
    
    # Parse VGG JSON structure
    if isinstance(vgg_data, dict):
        if '_via_img_metadata' in vgg_data:
            img_metadata = vgg_data['_via_img_metadata']
        else:
            img_metadata = vgg_data
    elif isinstance(vgg_data, list):
        # Convert list to dict format
        img_metadata = {}
        for item in vgg_data:
            if isinstance(item, dict) and 'filename' in item:
                img_id = item.get('filename', 'unknown')
                img_metadata[img_id] = item
    else:
        raise ValueError("Unsupported VGG JSON format")
    
    # Process each image
    for img_id, img_data in img_metadata.items():
        stats['total_images'] += 1
        
        try:
            # Get filename
            filename = img_data.get('filename') or img_data.get('fname')
            if not filename:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {img_id}: No filename found")
                continue
            
            # Find image file
            image_path = os.path.join(images_dir, filename)
            if not os.path.exists(image_path):
                # Try different extensions
                base_name = os.path.splitext(filename)[0]
                found = False
                for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                    alt_path = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        filename = base_name + ext
                        found = True
                        break
                
                if not found:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {filename}: File not found")
                    continue
            
            # Load image to get dimensions
            img = cv2.imread(image_path)
            if img is None:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {filename}: Could not load")
                continue
            
            h, w = img.shape[:2]
            
            # Parse annotations
            annotations = []
            regions = img_data.get('regions') or img_data.get('via_region_data', {})
            
            if isinstance(regions, list):
                for region in regions:
                    ann = _parse_vgg_region_to_yolo(region, w, h, class_mapping, task)
                    if ann:
                        annotations.append(ann)
                        stats['total_annotations'] += 1
            elif isinstance(regions, dict):
                for region_id, region in regions.items():
                    ann = _parse_vgg_region_to_yolo(region, w, h, class_mapping, task)
                    if ann:
                        annotations.append(ann)
                        stats['total_annotations'] += 1
            
            # Skip images without annotations
            if len(annotations) == 0:
                if DEBUG:
                    print(f"DEBUG: {filename}: skipping, no annotations")
                stats['skipped_images'] += 1
                continue
            
            if DEBUG:
                print(f"DEBUG: Processing {filename}: annotations={len(annotations)}")
            
            # Copy original image to output directory
            output_image_path = os.path.join(output_images_dir, filename)
            if not os.path.exists(output_image_path):
                import shutil
                shutil.copy2(image_path, output_image_path)
            
            # Create YOLO label file for original image
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(output_labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    if task == "detect":
                        # YOLO detection format: class_id x_center y_center width height (normalized)
                        line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
                    else:
                        # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
                        line = f"{ann['class_id']}"
                        for x, y in ann['polygon']:
                            line += f" {x:.6f} {y:.6f}"
                    f.write(line + '\n')
            
            if DEBUG:
                print(f"DEBUG: {filename}: saved original with {len(annotations)} annotations")
            
            stats['converted_images'] += 1
            
            # Create augmented versions if requested
            if augmentation_per_image > 0 and augmentation is not None:
                if DEBUG:
                    print(f"DEBUG: Processing {filename}: original_annotations={len(annotations)}")
                
                # Convert annotations to mask
                mask = _yolo_annotations_to_mask(annotations, w, h, task)
                
                # Convert image from BGR to RGB for augmentation
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Generate augmented versions
                base_name = os.path.splitext(filename)[0]
                file_ext = os.path.splitext(filename)[1]
                
                for aug_idx in range(augmentation_per_image):
                    if DEBUG:
                        print(f"DEBUG: Augmenting {filename} aug{aug_idx:03d}: starting with {len(annotations)} annotations")
                    try:
                        # Apply random augmentation
                        aug_img_rgb, aug_mask, _ = augmentation.apply_random_augmentation(img_rgb.copy(), mask.copy())
                        
                        # Validate augmented mask
                        if not augmentation.validate_mask(aug_mask):
                            if DEBUG:
                                print(f"DEBUG: {filename} aug{aug_idx:03d}: mask validation failed, skipping")
                            continue
                        
                        # Convert mask back to annotations
                        aug_annotations = _mask_to_yolo_annotations(aug_mask, annotations, task)
                        
                        if DEBUG:
                            print(f"DEBUG: {filename} aug{aug_idx:03d}: after conversion, annotations={len(aug_annotations)}")
                        
                        if len(aug_annotations) == 0:
                            if DEBUG:
                                print(f"DEBUG: {filename} aug{aug_idx:03d}: no valid annotations after augmentation, skipping")
                            continue
                        
                        # Save augmented image
                        aug_filename = f"{base_name}_aug{aug_idx:03d}{file_ext}"
                        aug_image_path = os.path.join(output_images_dir, aug_filename)
                        aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(aug_image_path, aug_img_bgr)
                        
                        # Save augmented label file
                        aug_label_filename = f"{base_name}_aug{aug_idx:03d}.txt"
                        aug_label_path = os.path.join(output_labels_dir, aug_label_filename)
                        
                        with open(aug_label_path, 'w') as f:
                            for ann in aug_annotations:
                                if task == "detect":
                                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
                                else:
                                    line = f"{ann['class_id']}"
                                    for x, y in ann['polygon']:
                                        line += f" {x:.6f} {y:.6f}"
                                f.write(line + '\n')
                        
                        if DEBUG:
                            print(f"DEBUG: {filename} aug{aug_idx:03d}: saved with {len(aug_annotations)} annotations")
                        
                        stats['augmented_images'] += 1
                        
                    except Exception as aug_e:
                        # Continue with next augmentation if one fails
                        if DEBUG:
                            print(f"DEBUG: {filename} aug{aug_idx:03d}: exception - {str(aug_e)}")
                        stats['errors'].append(f"Image {filename} augmentation {aug_idx}: {str(aug_e)}")
                        continue
            
        except Exception as e:
            stats['skipped_images'] += 1
            stats['errors'].append(f"Image {img_id}: {str(e)}")
            continue
    
    # Save class names file
    names_path = os.path.join(output_dir, "classes.txt")
    with open(names_path, 'w') as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("YOLO Conversion Statistics")
    print("=" * 80)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successfully converted: {stats['converted_images']}")
    print(f"Skipped: {stats['skipped_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Output directory: {output_dir}")
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    print("=" * 80)
    
    return stats


def _parse_vgg_region_to_yolo(region: Dict, img_width: int, img_height: int, class_mapping: Dict, task: str = "seg") -> Optional[Dict]:
    """
    Parse a VGG region to YOLO format.
    
    Args:
        region: Region dictionary from VGG JSON
        img_width: Image width
        img_height: Image height
        class_mapping: Dictionary mapping class names to class IDs
        task: Task mode - "seg" for segmentation or "detect" for detection
    
    Returns:
        For segmentation: Dictionary with 'class_id' and 'polygon' (normalized coordinates)
        For detection: Dictionary with 'class_id', 'x_center', 'y_center', 'width', 'height' (normalized)
        Returns None if invalid
    """
    # Get shape attributes
    shape_attr = region.get('shape_attributes') or region
    
    # Get class name
    region_attr = region.get('region_attributes', {})
    if isinstance(region_attr, dict):
        class_name = region_attr.get('name', region_attr.get('class', 'worm'))
    else:
        class_name = 'worm'
    
    class_id = class_mapping.get(class_name, 0)
    
    # Parse shape based on type
    shape_name = shape_attr.get('name', '')
    
    if shape_name == 'polygon' or 'all_points_x' in shape_attr:
        # Polygon annotation
        all_points_x = shape_attr.get('all_points_x', [])
        all_points_y = shape_attr.get('all_points_y', [])
        
        if len(all_points_x) < 3 or len(all_points_y) < 3:
            return None
        
        # Calculate bounding box from polygon
        x_min = min(all_points_x)
        x_max = max(all_points_x)
        y_min = min(all_points_y)
        y_max = max(all_points_y)
        
        if task == "detect":
            # Detection format: x_center, y_center, width, height (normalized)
            x_center = ((x_min + x_max) / 2.0) / img_width
            y_center = ((y_min + y_max) / 2.0) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Clamp to [0, 1]
            x_center = max(0, min(1.0, x_center))
            y_center = max(0, min(1.0, y_center))
            width = max(0, min(1.0, width))
            height = max(0, min(1.0, height))
            
            return {'class_id': class_id, 'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height}
        else:
            # Segmentation format: polygon points (normalized)
            polygon = []
            for x, y in zip(all_points_x, all_points_y):
                # Clamp to image bounds
                x_norm = max(0, min(1.0, x / img_width))
                y_norm = max(0, min(1.0, y / img_height))
                polygon.append((x_norm, y_norm))
            
            return {'class_id': class_id, 'polygon': polygon}
    
    elif shape_name == 'rect' or ('x' in shape_attr and 'y' in shape_attr and 'width' in shape_attr and 'height' in shape_attr):
        # Rectangle annotation
        x = shape_attr.get('x', 0)
        y = shape_attr.get('y', 0)
        width = shape_attr.get('width', 0)
        height = shape_attr.get('height', 0)
        
        if width <= 0 or height <= 0:
            return None
        
        if task == "detect":
            # Detection format: x_center, y_center, width, height (normalized)
            x_center = (x + width / 2.0) / img_width
            y_center = (y + height / 2.0) / img_height
            width_norm = width / img_width
            height_norm = height / img_height
            
            # Clamp to [0, 1]
            x_center = max(0, min(1.0, x_center))
            y_center = max(0, min(1.0, y_center))
            width_norm = max(0, min(1.0, width_norm))
            height_norm = max(0, min(1.0, height_norm))
            
            return {'class_id': class_id, 'x_center': x_center, 'y_center': y_center, 'width': width_norm, 'height': height_norm}
        else:
            # Segmentation format: polygon from rectangle
            polygon = [
                (x / img_width, y / img_height),
                ((x + width) / img_width, y / img_height),
                ((x + width) / img_width, (y + height) / img_height),
                (x / img_width, (y + height) / img_height)
            ]
            
            return {'class_id': class_id, 'polygon': polygon}
    
    elif shape_name == 'circle' or ('cx' in shape_attr and 'cy' in shape_attr and 'r' in shape_attr):
        # Circle annotation
        cx = shape_attr.get('cx', 0)
        cy = shape_attr.get('cy', 0)
        r = shape_attr.get('r', 0)
        
        if r <= 0:
            return None
        
        if task == "detect":
            # Detection format: bounding box from circle
            x_min = cx - r
            x_max = cx + r
            y_min = cy - r
            y_max = cy + r
            
            x_center = cx / img_width
            y_center = cy / img_height
            width = (2 * r) / img_width
            height = (2 * r) / img_height
            
            # Clamp to [0, 1]
            x_center = max(0, min(1.0, x_center))
            y_center = max(0, min(1.0, y_center))
            width = max(0, min(1.0, width))
            height = max(0, min(1.0, height))
            
            return {'class_id': class_id, 'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height}
        else:
            # Segmentation format: polygon approximation of circle (32 points)
            n_points = 32
            polygon = []
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                x = (cx + r * np.cos(angle)) / img_width
                y = (cy + r * np.sin(angle)) / img_height
                x = max(0, min(1.0, x))
                y = max(0, min(1.0, y))
                polygon.append((x, y))
            
            return {'class_id': class_id, 'polygon': polygon}
    
    return None


def create_yolo_dataset_config(output_dir: str, train_split: float = 0.8, val_split: float = 0.1):
    """
    Create YOLO dataset configuration file and split into train/val/test.
    
    Args:
        output_dir: Output directory containing 'images' and 'labels' folders
        train_split: Fraction of data for training (default: 0.8)
        val_split: Fraction of data for validation (default: 0.1)
        # test_split = 1 - train_split - val_split
    
    Returns:
        Path to dataset YAML file
    """
    import random
    
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    # Get all image files
    image_files = []
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        image_files.extend(Path(images_dir).glob(f'*{ext}'))
    
    image_files = [str(f) for f in image_files]
    random.shuffle(image_files)
    
    # Split dataset
    n_total = len(image_files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    # Create split directories
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_images_dir = os.path.join(output_dir, split_name, 'images')
        split_labels_dir = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
        
        import shutil
        for img_path in files:
            img_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            
            # Copy image
            shutil.copy2(img_path, os.path.join(split_images_dir, img_filename))
            
            # Copy label
            label_path = os.path.join(labels_dir, label_filename)
            if os.path.exists(label_path):
                shutil.copy2(label_path, os.path.join(split_labels_dir, label_filename))
    
    # Create dataset YAML file
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w') as f:
        f.write(f"# YOLO Dataset Configuration\n")
        f.write(f"path: {os.path.abspath(output_dir)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        if len(test_files) > 0:
            f.write(f"test: test/images\n")
        f.write(f"\n")
        f.write(f"# Classes\n")
        f.write(f"nc: 1  # number of classes\n")
        f.write(f"names: ['worm']\n")
    
    print(f"\n✅ Dataset configuration created: {yaml_path}")
    print(f"   Train: {len(train_files)} images")
    print(f"   Val: {len(val_files)} images")
    print(f"   Test: {len(test_files)} images")
    
    return yaml_path


def convert_cvat_xml_to_yolo(cvat_xml_path: str, images_dir: str, output_dir: str, class_name: str = "worm", task: str = "seg", augmentation_per_image: int = 0):
    """
    Convert CVAT XML annotations to YOLO format.
    
    CVAT XML format contains bounding boxes (<box> elements) which are converted
    to YOLO format (detection or segmentation based on task parameter).
    
    Args:
        cvat_xml_path: Path to CVAT XML annotation file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO format (will create 'images' and 'labels' subdirectories)
        class_name: Name of the class (default: "worm")
        task: Task mode - "seg" for segmentation (polygon format) or "detect" for detection (bounding box format)
        augmentation_per_image: Number of augmented versions to create per original image (0 = no augmentation)
    
    Returns:
        Dictionary with conversion statistics
    """
    # Create output directories
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    # Parse CVAT XML
    try:
        tree = ET.parse(cvat_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse CVAT XML file: {e}")
    
    # Create class mapping (single class: worm = class 0)
    class_mapping = {class_name: 0}
    
    # Statistics
    stats = {
        'total_images': 0,
        'converted_images': 0,
        'augmented_images': 0,
        'skipped_images': 0,
        'total_annotations': 0,
        'errors': []
    }
    
    # Create augmentation instance if needed
    augmentation = None
    if augmentation_per_image > 0:
        augmentation = DataAugmentation(
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
    
    # Process each image element
    for image_elem in root.findall('image'):
        stats['total_images'] += 1
        
        try:
            # Get image attributes
            image_name = image_elem.get('name')
            if not image_name:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {image_elem.get('id', 'unknown')}: No name attribute")
                continue
            
            # Get image dimensions from XML (more reliable than loading image)
            img_width = int(image_elem.get('width', 0))
            img_height = int(image_elem.get('height', 0))
            
            if img_width <= 0 or img_height <= 0:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {image_name}: Invalid dimensions ({img_width}x{img_height})")
                continue
            
            # Find image file
            image_path = os.path.join(images_dir, image_name)
            original_path = None
            output_image_name = image_name  # Will be updated if conversion happens
            
            if not os.path.exists(image_path):
                # Try different extensions
                base_name = os.path.splitext(image_name)[0]
                found = False
                for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.TIF', '.TIFF', '.PNG', '.JPG', '.JPEG']:
                    alt_path = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        original_path = alt_path
                        image_path = alt_path
                        image_name = base_name + ext
                        found = True
                        break
                
                if not found:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {image_name}: File not found in {images_dir}")
                    continue
            else:
                original_path = image_path
            
            # Check if image is TIF/TIFF and convert to PNG for better compatibility
            image_ext = os.path.splitext(original_path)[1].lower()
            if image_ext in ['.tif', '.tiff']:
                # Convert TIF to PNG
                png_path = os.path.join(images_dir, os.path.splitext(image_name)[0] + '.png')
                
                if not os.path.exists(png_path):
                    # Try to load and convert
                    try:
                        # Try using PIL first (better TIF support)
                        try:
                            from PIL import Image as PILImage
                            pil_img = PILImage.open(original_path)
                            # Convert to RGB if needed (TIF might be grayscale or have alpha)
                            if pil_img.mode != 'RGB':
                                pil_img = pil_img.convert('RGB')
                            pil_img.save(png_path, 'PNG')
                            print(f"  ✅ Converted {image_name} -> {os.path.basename(png_path)} (using PIL)")
                        except ImportError:
                            # Fall back to OpenCV
                            img = cv2.imread(original_path)
                            if img is None:
                                raise ValueError("Could not load TIF with OpenCV")
                            cv2.imwrite(png_path, img)
                            print(f"  ✅ Converted {image_name} -> {os.path.basename(png_path)} (using OpenCV)")
                    except Exception as e:
                        stats['skipped_images'] += 1
                        stats['errors'].append(f"Image {image_name}: TIF to PNG conversion failed - {str(e)}")
                        continue
                
                # Update reference to PNG file
                output_image_name = os.path.splitext(image_name)[0] + '.png'
                image_path = png_path
            
            # Verify image can be loaded
            img = cv2.imread(image_path)
            if img is None:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {output_image_name}: Could not load")
                continue
            
            # Verify dimensions match (warn if they don't)
            h, w = img.shape[:2]
            if w != img_width or h != img_height:
                print(f"⚠️  Warning: Image {output_image_name} dimensions mismatch - XML: {img_width}x{img_height}, Actual: {w}x{h}")
                # Use actual image dimensions
                img_width, img_height = w, h
            
            # Parse bounding box annotations
            annotations = []
            for box_elem in image_elem.findall('box'):
                ann = _parse_cvat_box_to_yolo(box_elem, img_width, img_height, class_mapping, task)
                if ann:
                    annotations.append(ann)
                    stats['total_annotations'] += 1
            
            # Skip images without annotations
            if len(annotations) == 0:
                if DEBUG:
                    print(f"DEBUG: {output_image_name}: skipping, no annotations")
                stats['skipped_images'] += 1
                continue
            
            if DEBUG:
                print(f"DEBUG: Processing {output_image_name}: annotations={len(annotations)}")
            
            # Copy image to output directory (use output_image_name which may be PNG if converted)
            output_image_path = os.path.join(output_images_dir, output_image_name)
            if not os.path.exists(output_image_path):
                import shutil
                shutil.copy2(image_path, output_image_path)
            
            # Create YOLO label file (use output_image_name for consistency)
            label_filename = os.path.splitext(output_image_name)[0] + '.txt'
            label_path = os.path.join(output_labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for ann in annotations:
                    if task == "detect":
                        # YOLO detection format: class_id x_center y_center width height (normalized)
                        line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
                    else:
                        # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates)
                        line = f"{ann['class_id']}"
                        for x, y in ann['polygon']:
                            line += f" {x:.6f} {y:.6f}"
                    f.write(line + '\n')
            
            if DEBUG:
                print(f"DEBUG: {output_image_name}: saved original with {len(annotations)} annotations")
            
            stats['converted_images'] += 1
            
            # Create augmented versions if requested
            if augmentation_per_image > 0 and augmentation is not None and img is not None:
                if DEBUG:
                    print(f"DEBUG: Processing {output_image_name}: original_annotations={len(annotations)}")
                
                # Get image dimensions
                h, w = img.shape[:2]
                
                # Convert annotations to mask
                mask = _yolo_annotations_to_mask(annotations, w, h, task)
                
                # Convert image from BGR to RGB for augmentation
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Generate augmented versions
                base_name = os.path.splitext(output_image_name)[0]
                file_ext = os.path.splitext(output_image_name)[1]
                
                for aug_idx in range(augmentation_per_image):
                    if DEBUG:
                        print(f"DEBUG: Augmenting {output_image_name} aug{aug_idx:03d}: starting with {len(annotations)} annotations")
                    try:
                        # Apply random augmentation
                        aug_img_rgb, aug_mask, _ = augmentation.apply_random_augmentation(img_rgb.copy(), mask.copy())
                        
                        # Validate augmented mask
                        if not augmentation.validate_mask(aug_mask):
                            if DEBUG:
                                print(f"DEBUG: {output_image_name} aug{aug_idx:03d}: mask validation failed, skipping")
                            continue
                        
                        # Convert mask back to annotations
                        aug_annotations = _mask_to_yolo_annotations(aug_mask, annotations, task)
                        
                        if DEBUG:
                            print(f"DEBUG: {output_image_name} aug{aug_idx:03d}: after conversion, annotations={len(aug_annotations)}")
                        
                        if len(aug_annotations) == 0:
                            if DEBUG:
                                print(f"DEBUG: {output_image_name} aug{aug_idx:03d}: no valid annotations after augmentation, skipping")
                            continue
                        
                        # Save augmented image
                        aug_filename = f"{base_name}_aug{aug_idx:03d}{file_ext}"
                        aug_image_path = os.path.join(output_images_dir, aug_filename)
                        aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(aug_image_path, aug_img_bgr)
                        
                        # Save augmented label file
                        aug_label_filename = f"{base_name}_aug{aug_idx:03d}.txt"
                        aug_label_path = os.path.join(output_labels_dir, aug_label_filename)
                        
                        with open(aug_label_path, 'w') as f:
                            for ann in aug_annotations:
                                if task == "detect":
                                    line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}"
                                else:
                                    line = f"{ann['class_id']}"
                                    for x, y in ann['polygon']:
                                        line += f" {x:.6f} {y:.6f}"
                                f.write(line + '\n')
                        
                        if DEBUG:
                            print(f"DEBUG: {output_image_name} aug{aug_idx:03d}: saved with {len(aug_annotations)} annotations")
                        
                        stats['augmented_images'] += 1
                        
                    except Exception as aug_e:
                        # Continue with next augmentation if one fails
                        if DEBUG:
                            print(f"DEBUG: {output_image_name} aug{aug_idx:03d}: exception - {str(aug_e)}")
                        stats['errors'].append(f"Image {output_image_name} augmentation {aug_idx}: {str(aug_e)}")
                        continue
            
        except Exception as e:
            stats['skipped_images'] += 1
            image_id = image_elem.get('id', 'unknown')
            stats['errors'].append(f"Image {image_id}: {str(e)}")
            continue
    
    # Save class names file
    names_path = os.path.join(output_dir, "classes.txt")
    with open(names_path, 'w') as f:
        for class_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            f.write(f"{class_name}\n")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("CVAT XML to YOLO Conversion Statistics")
    print("=" * 80)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successfully converted: {stats['converted_images']}")
    if augmentation_per_image > 0:
        print(f"Augmented images created: {stats['augmented_images']}")
        print(f"Total images in dataset: {stats['converted_images'] + stats['augmented_images']}")
    print(f"Skipped: {stats['skipped_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Output directory: {output_dir}")
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    print("=" * 80)
    if task == "seg":
        print("\n⚠️  NOTE: CVAT XML contains bounding boxes, which are converted to rectangular polygons.")
        print("   This means masks will be rectangular approximations of worm shapes.")
        print("   For better segmentation accuracy, consider using polygon/mask annotations.")
    else:
        print("\n✅ NOTE: CVAT XML bounding boxes are being used directly for detection training.")
    print("=" * 80)
    
    return stats


def _parse_cvat_box_to_yolo(box_elem: ET.Element, img_width: int, img_height: int, class_mapping: Dict, task: str = "seg") -> Optional[Dict]:
    """
    Parse a CVAT <box> element to YOLO format.
    
    CVAT boxes use xtl, ytl, xbr, ybr (top-left and bottom-right corners).
    These are converted to YOLO format based on task parameter.
    
    Args:
        box_elem: XML Element for <box>
        img_width: Image width
        img_height: Image height
        class_mapping: Dictionary mapping class names to class IDs
        task: Task mode - "seg" for segmentation or "detect" for detection
    
    Returns:
        For segmentation: Dictionary with 'class_id' and 'polygon' (normalized coordinates)
        For detection: Dictionary with 'class_id', 'x_center', 'y_center', 'width', 'height' (normalized)
        Returns None if invalid
    """
    # Get label (class name)
    label = box_elem.get('label', 'worm')
    class_id = class_mapping.get(label, 0)
    
    # Get bounding box coordinates
    try:
        xtl = float(box_elem.get('xtl', 0))
        ytl = float(box_elem.get('ytl', 0))
        xbr = float(box_elem.get('xbr', 0))
        ybr = float(box_elem.get('ybr', 0))
    except (ValueError, TypeError):
        return None
    
    # Validate coordinates
    if xbr <= xtl or ybr <= ytl:
        return None
    
    # Clamp coordinates to image bounds
    xtl = max(0, min(img_width, xtl))
    ytl = max(0, min(img_height, ytl))
    xbr = max(0, min(img_width, xbr))
    ybr = max(0, min(img_height, ybr))
    
    if task == "detect":
        # Detection format: x_center, y_center, width, height (normalized)
        x_center = ((xtl + xbr) / 2.0) / img_width
        y_center = ((ytl + ybr) / 2.0) / img_height
        width = (xbr - xtl) / img_width
        height = (ybr - ytl) / img_height
        
        # Ensure all coordinates are in valid range [0, 1]
        x_center = max(0, min(1.0, x_center))
        y_center = max(0, min(1.0, y_center))
        width = max(0, min(1.0, width))
        height = max(0, min(1.0, height))
        
        return {'class_id': class_id, 'x_center': x_center, 'y_center': y_center, 'width': width, 'height': height}
    else:
        # Segmentation format: rectangular polygon (4 points, clockwise from top-left)
        # YOLO format expects normalized coordinates (0-1)
        polygon = [
            (xtl / img_width, ytl / img_height),           # Top-left
            (xbr / img_width, ytl / img_height),           # Top-right
            (xbr / img_width, ybr / img_height),           # Bottom-right
            (xtl / img_width, ybr / img_height)            # Bottom-left
        ]
        
        # Ensure all coordinates are in valid range [0, 1]
        polygon = [(max(0, min(1.0, x)), max(0, min(1.0, y))) for x, y in polygon]
        
        return {'class_id': class_id, 'polygon': polygon}

