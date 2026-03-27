"""
YOLOv8 Segmentation and Detection Function
Compatible interface with RCNN_Segment for easy integration.
Supports both segmentation (masks) and detection (bounding boxes) models.
For detection models, bounding boxes are converted to rectangular masks.
"""

import numpy as np
import cv2
import torch
from .YOLO8_Helper import loadYOLO8


def YOLO8_Segment(model, img_crop, seg_params):
    """
    Segment worms in an image using YOLOv8 model.
    Compatible interface with RCNN_Segment.
    Supports both detection (bounding boxes) and segmentation (masks) models.
    
    Args:
        model: YOLOv8 model (from loadYOLO8)
        img_crop: Input image (numpy array, BGR format)
        seg_params: Dictionary with segmentation parameters:
            - "task": Model task mode - REQUIRED: "seg" (segmentation) or "detect" (detection)
            - "overlap preference": REQUIRED: "size" (choose largest box) or "confidence" (choose highest confidence)
            - "pix conf thresh": Pixel confidence threshold (default: 0.5, ignored for detection models)
            - "overlap thresh": Overlap threshold for NMS (default: 0.5)
            - "min object area": Minimum object area in pixels (default: 80)
            - "object conf thresh": Object confidence threshold (default: 0.25)
            - "iou thresh": IoU threshold for NMS (default: 0.45)
    
    Returns:
        masks: Full masks array [H, W, 1, N] where N is number of detections
        masks_thresh: Thresholded masks [H, W, 1, N]
        weighted: Weighted sum of all masks [H, W]
        weighted_touch: Weighted sum of touching masks [H, W]
        touch_count: List of touch counts for each detection
    """
    # Get parameters with defaults
    task_mode = seg_params.get("task", None)
    if task_mode is None:
        raise ValueError(
            "'task' parameter is required in seg_params. "
            "Must be 'seg' for segmentation models or 'detect' for detection models. "
            "Example: seg_params['task'] = 'detect'"
        )
    if task_mode not in ["seg", "detect"]:
        raise ValueError(f"Invalid task mode: {task_mode}. Must be 'seg' or 'detect'")
    
    # Get overlap preference (required parameter)
    overlap_preference = seg_params.get("overlap preference", None)
    if overlap_preference is None:
        raise ValueError(
            "'overlap preference' parameter is required in seg_params. "
            "Must be 'size' (choose largest box) or 'confidence' (choose highest confidence). "
            "Example: seg_params['overlap preference'] = 'size'"
        )
    if overlap_preference not in ["size", "confidence"]:
        raise ValueError(
            f"Invalid overlap preference: {overlap_preference}. "
            "Must be 'size' or 'confidence'"
        )
    
    pix_conf_thresh = seg_params.get("pix conf thresh", 0.5)
    overlap_thresh = seg_params.get("overlap thresh", 0.5)
    min_object_area = seg_params.get("min object area", 80)
    # Get object_conf_thresh from config (can be None to use YOLO's default)
    object_conf_thresh = seg_params.get("object conf thresh")
    iou_thresh = seg_params.get("iou thresh", 0.45)
    
    # Ensure image is in correct format (BGR, uint8)
    if img_crop.dtype != np.uint8:
        if img_crop.max() <= 1.0:
            img_crop = (img_crop * 255).astype(np.uint8)
        else:
            img_crop = img_crop.astype(np.uint8)
    
    # Convert grayscale to BGR if needed
    if len(img_crop.shape) == 2:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    elif img_crop.shape[2] == 1:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2BGR)
    
    # Run YOLOv8 inference
    # Use image size from seg_params (sliding window size or imgsz), default to 512
    imgsz = seg_params.get("imgsz")
    if imgsz is None:
        imgsz = seg_params.get("sliding window size", 512)
    # Ensure imgsz is not None
    if imgsz is None:
        imgsz = 512
    
    # Build predict arguments - only include conf if it's not None
    predict_args = {
        'iou': iou_thresh,
        'verbose': False,
        'imgsz': imgsz,
    }
    # Only add conf parameter if it's specified in config (not None)
    if object_conf_thresh is not None:
        predict_args['conf'] = object_conf_thresh
    
    results = model.predict(img_crop, **predict_args)
    
    # Extract results from first (and only) image
    result = results[0]
    
    # Get image dimensions
    h, w = img_crop.shape[:2]
    
    # Determine if this is a detection or segmentation model
    has_masks = (result.masks is not None and 
                 hasattr(result.masks, 'data') and 
                 len(result.masks.data) > 0)
    has_boxes = (result.boxes is not None and 
                 len(result.boxes) > 0)
    
    # Check for detections
    if not has_boxes:
        # No detections at all
        masks = np.zeros((h, w, 1, 0), dtype=np.uint8)
        masks_thresh = np.zeros((h, w, 1, 0), dtype=np.uint8)
        weighted = np.zeros((h, w), dtype=np.float32)
        weighted_touch = np.zeros((h, w), dtype=np.float32)
        touch_count = []
        return masks, masks_thresh, weighted, weighted_touch, touch_count
    
    # Get boxes and scores (available for both detection and segmentation models)
    pred_scores = result.boxes.conf.cpu().numpy()  # [N]
    pred_boxes = result.boxes.xyxy.cpu().numpy()  # [N, 4] in xyxy format
    
    # Process based on task mode
    if task_mode == "detect":
        # Detection model: convert bounding boxes to rectangular masks
        # Note: has_boxes is already True at this point (checked earlier)
        n_detections = len(pred_boxes)
        pred_masks = []
        
        for i in range(n_detections):
            x1, y1, x2, y2 = pred_boxes[i]
            # Ensure coordinates are within image bounds and convert to integers
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            
            # Create rectangular mask (filled box)
            mask = np.zeros((h, w), dtype=np.float32)
            if x2 > x1 and y2 > y1:  # Valid box
                mask[y1:y2, x1:x2] = 1.0
            pred_masks.append(mask)
        
        pred_masks = np.array(pred_masks)  # [N, H, W]
        
    else:  # task_mode == "seg"
        # Segmentation model: use masks from model
        if not has_masks:
            # Segmentation model but no masks found - fall back to empty
            print("⚠️  Warning: Segmentation model but no masks found. Returning empty results.")
            masks = np.zeros((h, w, 1, 0), dtype=np.uint8)
            masks_thresh = np.zeros((h, w, 1, 0), dtype=np.uint8)
            weighted = np.zeros((h, w), dtype=np.float32)
            weighted_touch = np.zeros((h, w), dtype=np.float32)
            touch_count = []
            return masks, masks_thresh, weighted, weighted_touch, touch_count
        
        pred_masks = result.masks.data.cpu().numpy()  # [N, H, W]
        
        # Resize masks to original image size if needed
        if pred_masks.shape[1] != h or pred_masks.shape[2] != w:
            # Resize masks to original image size
            resized_masks = []
            for mask in pred_masks:
                mask_resized = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                resized_masks.append(mask_resized)
            pred_masks = np.array(resized_masks)
    
    n_detections = len(pred_masks)
    
    # Apply pixel confidence threshold
    # For detection models, masks are already binary (0 or 1), so thresholding has no effect
    # but we keep it for consistency with the interface
    if task_mode == "detect":
        # Detection masks are already binary (filled rectangles), so just convert to uint8
        masks_thresh = (pred_masks >= 0.5).astype(np.uint8)
    else:
        # Segmentation masks need thresholding
        masks_thresh = (pred_masks >= pix_conf_thresh).astype(np.uint8)
    
    # Filter by minimum area
    valid_indices = []
    for i in range(n_detections):
        area = np.sum(masks_thresh[i])
        if area >= min_object_area:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        masks = np.zeros((h, w, 1, 0), dtype=np.uint8)
        masks_thresh = np.zeros((h, w, 1, 0), dtype=np.uint8)
        weighted = np.zeros((h, w), dtype=np.float32)
        weighted_touch = np.zeros((h, w), dtype=np.float32)
        touch_count = []
        return masks, masks_thresh, weighted, weighted_touch, touch_count
    
    # Keep only valid detections
    masks_thresh = masks_thresh[valid_indices]
    pred_masks = pred_masks[valid_indices]
    pred_scores = pred_scores[valid_indices]
    
    # Apply additional Non-Maximum Suppression based on mask overlap if needed
    if len(masks_thresh) > 1 and overlap_thresh < 1.0:
        keep_indices = _apply_mask_nms(masks_thresh, pred_scores, overlap_thresh, overlap_preference)
        masks_thresh = masks_thresh[keep_indices]
        pred_masks = pred_masks[keep_indices]
        pred_scores = pred_scores[keep_indices]
    
    n_detections = len(masks_thresh)
    
    # Reshape masks to [H, W, 1, N] format
    masks = pred_masks.astype(np.float32)
    masks = np.transpose(masks, (1, 2, 0))  # [H, W, N]
    masks = np.expand_dims(masks, axis=2)  # [H, W, 1, N]
    masks = (masks * 255).astype(np.uint8)
    
    masks_thresh = np.transpose(masks_thresh, (1, 2, 0))  # [H, W, N]
    masks_thresh = np.expand_dims(masks_thresh, axis=2)  # [H, W, 1, N]
    masks_thresh = (masks_thresh * 255).astype(np.uint8)
    
    # Compute weighted sum (weighted by confidence scores)
    weighted = np.zeros((h, w), dtype=np.float32)
    for i in range(n_detections):
        weighted += pred_masks[i] * pred_scores[i]
    
    # Compute touch count for each detection
    # A detection is "touching" if it overlaps significantly with other detections
    touch_count = []
    for i in range(n_detections):
        mask_i = masks_thresh[:, :, 0, i].astype(bool)
        overlap_count = 0
        for j in range(n_detections):
            if i != j:
                mask_j = masks_thresh[:, :, 0, j].astype(bool)
                overlap = np.sum(mask_i & mask_j)
                if overlap > 0:
                    overlap_count += 1
        touch_count.append(overlap_count)
    
    # Compute weighted_touch (weighted sum of masks that are touching)
    weighted_touch = np.zeros((h, w), dtype=np.float32)
    for i in range(n_detections):
        if touch_count[i] > 0:
            weighted_touch += pred_masks[i] * pred_scores[i]
    
    return masks, masks_thresh, weighted, weighted_touch, touch_count


def _apply_mask_nms(masks, scores, overlap_thresh, overlap_preference):
    """
    Apply Non-Maximum Suppression on masks based on overlap.
    Overlap is calculated as: intersection_area / min(area_box1, area_box2)
    (i.e., the fraction of the smaller box that is covered by the intersection)
    
    When objects overlap, chooses based on overlap_preference:
    - "size": chooses the LARGEST box (by area)
    - "confidence": chooses the HIGHEST confidence score
    
    Args:
        masks: Array of masks [N, H, W]
        scores: Array of scores [N]
        overlap_thresh: Overlap threshold for suppression (0.0 to 1.0)
        overlap_preference: "size" or "confidence" - determines which box to keep on overlap
    
    Returns:
        keep_indices: Indices of masks to keep
    """
    if len(masks) == 0:
        return []
    
    n = len(masks)
    
    # Calculate overlap matrix (intersection over smaller box area)
    # Overlap is calculated as the fraction of the smaller box that is covered by intersection
    overlaps = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            mask_i = masks[i].astype(bool)
            mask_j = masks[j].astype(bool)
            
            intersection = np.sum(mask_i & mask_j)
            area_i = np.sum(mask_i)
            area_j = np.sum(mask_j)
            
            # Overlap is relative to the smaller box
            if area_i > 0 and area_j > 0:
                min_area = min(area_i, area_j)
                if min_area > 0:
                    overlap = intersection / min_area
                    overlaps[i, j] = overlap
                    overlaps[j, i] = overlap
    
    # Sort based on preference
    if overlap_preference == "size":
        # Calculate areas for each mask (to determine largest)
        areas = np.array([np.sum(mask.astype(bool)) for mask in masks])
        sorted_indices = np.argsort(areas)[::-1]  # Descending: largest first
    elif overlap_preference == "confidence":
        sorted_indices = np.argsort(scores)[::-1]  # Descending: highest confidence first
    else:
        raise ValueError(f"Invalid overlap_preference: {overlap_preference}. Must be 'size' or 'confidence'")
    
    keep = []
    
    while len(sorted_indices) > 0:
        # Take the preferred detection (largest or highest confidence)
        current = sorted_indices[0]
        keep.append(current)
        
        # Remove all detections that overlap too much with current
        remaining = []
        for idx in sorted_indices[1:]:
            if overlaps[current, idx] < overlap_thresh:
                remaining.append(idx)
        
        sorted_indices = np.array(remaining)
    
    return keep


# For backward compatibility, also provide loadYOLO8 here
from .YOLO8_Helper import loadYOLO8

