from email import message
import os
import copy

from cv2 import EVENT_LBUTTONDBLCLK
from Plate import *
from MultiWellAnalysis import *
from tkinter import messagebox
import tkinter as tk
import numpy as np

# Force UI windows to top
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)


# Generate the analysis object (loads paths and params)
global click_pt, view_frac, zoom_lvl, zoom_lvls, mask_debug, mouse_context
k = 0 # key needs global scope?
click_pt = []
view_frac = 1.0
zoom_lvl = 0 # 0,1,2
zoom_lvls = [1.0,0.6,0.3]
mask_debug = False
mouse_context = {}  # Context for mouse callback (updated in main loop)
MWA = MultiWellAnalysis()


def load_masks(base_dir, filename, target_shape):
    mask_dir = os.path.join(base_dir, "Masks")
    candidate_names = [
        filename,
        filename.replace(".png", "_mask.png"),
        filename.replace(".tif", "_mask.tif"),
        filename.replace(".tiff", "_mask.tiff"),
    ]
    mask_path = None
    for candidate in candidate_names:
        candidate_path = os.path.join(mask_dir, candidate)
        if os.path.exists(candidate_path):
            mask_path = candidate_path
            break

    if mask_path is None:
        return None, None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask, mask


def extract_mask_components(binary_mask, label_mask=None):
    if binary_mask is None:
        return []
    components = []

    if label_mask is not None:
        unique_values = np.unique(label_mask)
        unique_values = unique_values[unique_values > 0]
        # First split by unique label values
        if len(unique_values) > 0 and len(unique_values) <= 512:
            for value in unique_values:
                label_component = (label_mask == value).astype(np.uint8)
                if not label_component.any():
                    continue
                # Then further split each label mask by connected components
                num_labels, labels = cv2.connectedComponents(label_component)
                for label_idx in range(1, num_labels):
                    component_mask = (labels == label_idx).astype(np.uint8)
                    if component_mask.any():
                        components.append(component_mask)
            if components:
                return components

    # Fallback: if no label_mask or label-based splitting didn't work, use connected components on binary mask
    num_labels, labels = cv2.connectedComponents(binary_mask.astype(np.uint8))
    for label_idx in range(1, num_labels):
        component_mask = (labels == label_idx).astype(np.uint8)
        if component_mask.any():
            components.append(component_mask)
    return components


def rasterize_worm_contours(worms, target_shape):
    height, width = target_shape
    worm_masks = []
    for worm in worms:
        contours_raw = worm.get("contours", [])
        if not contours_raw:
            continue
        worm_mask = np.zeros((height, width), dtype=np.uint8)
        contour_array = np.asarray(contours_raw, dtype=np.int32)
        contour_list = []

        if contour_array.size == 0:
            continue

        if contour_array.ndim == 3 and contour_array.shape[-1] == 2:
            if contour_array.shape[1] == 1:
                contour_list = [contour_array.reshape(-1, 1, 2)]
            else:
                contour_list = [contour_array[i].reshape(-1, 1, 2) for i in range(contour_array.shape[0])]
        elif contour_array.ndim == 2 and contour_array.shape[-1] == 2:
            contour_list = [contour_array.reshape(-1, 1, 2)]
        else:
            contour_list = [contour_array.reshape(-1, 1, 2)]

        sanitized_contours = []
        for contour_np in contour_list:
            if contour_np.shape[0] < 3:
                continue
            contour_np[:, 0, 0] = np.clip(contour_np[:, 0, 0], 0, width - 1)
            contour_np[:, 0, 1] = np.clip(contour_np[:, 0, 1], 0, height - 1)
            sanitized_contours.append(contour_np)

        if sanitized_contours:
            cv2.drawContours(worm_mask, sanitized_contours, -1, color=1, thickness=-1)

        if worm_mask.any():
            worm_masks.append(worm_mask)
    return worm_masks


def mask_to_bbox(binary_mask):
    if binary_mask is None or not binary_mask.any():
        return None
    ys, xs = np.where(binary_mask > 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def greedy_match_iou(pred_masks, gt_masks, threshold=0.05):
    if not pred_masks or not gt_masks:
        tp = 0
        fp = len(pred_masks)
        fn = len(gt_masks)
        return [], tp, fp, fn, 0.0

    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    pred_bools = [mask.astype(bool) for mask in pred_masks]
    gt_bools = [mask.astype(bool) for mask in gt_masks]

    for p_idx, pred_bool in enumerate(pred_bools):
        pred_area = pred_bool.sum()
        if pred_area == 0:
            continue
        for g_idx, gt_bool in enumerate(gt_bools):
            intersection = np.logical_and(pred_bool, gt_bool).sum()
            if intersection == 0:
                continue
            union = np.logical_or(pred_bool, gt_bool).sum()
            if union == 0:
                continue
            iou_matrix[p_idx, g_idx] = intersection / union

    global_max_iou = float(iou_matrix.max()) if iou_matrix.size else 0.0

    matches = []
    used_preds = set()
    used_gts = set()

    while True:
        max_index = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        best_iou = iou_matrix[max_index]
        if best_iou < threshold or best_iou <= 0:
            break
        p_idx, g_idx = max_index
        matches.append((p_idx, g_idx, float(best_iou)))
        used_preds.add(p_idx)
        used_gts.add(g_idx)
        iou_matrix[p_idx, :] = -1
        iou_matrix[:, g_idx] = -1

    tp = len(matches)
    fp = len(pred_masks) - len(used_preds)
    fn = len(gt_masks) - len(used_gts)
    return matches, tp, fp, fn, global_max_iou


def compute_detection_metrics(frame_data, target_shape, mask_binary, mask_label=None):
    if mask_binary is None:
        return None
    worms = frame_data["Well 1"]["worms"]
    
    # Filter out worms rejected by stage 2 (secondary classifier)
    # Rejected worms should be counted as negatives, not as positive predictions
    confirmed_worms = []
    rejected_count = 0
    for worm in worms:
        # Only include worms that are confirmed or pending (not rejected)
        # If classification_status is missing (old data), include it (backward compatibility)
        status = worm.get("classification_status", "pending")
        if status == "rejected":
            rejected_count += 1
        else:
            confirmed_worms.append(worm)
    
    pred_masks = rasterize_worm_contours(confirmed_worms, target_shape)
    gt_masks = extract_mask_components(mask_binary, mask_label)
    pred_union = np.zeros(target_shape, dtype=np.uint8)
    for mask in pred_masks:
        pred_union = np.logical_or(pred_union, mask.astype(bool))
    pred_union = pred_union.astype(np.uint8)

    gt_union = np.zeros(target_shape, dtype=np.uint8)
    for mask in gt_masks:
        gt_union = np.logical_or(gt_union, mask.astype(bool))
    gt_union = gt_union.astype(np.uint8)

    matches, tp, fp, fn, max_iou = greedy_match_iou(pred_masks, gt_masks, threshold=0.05)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if tp == 0 and fn == 0 else 0.0)
    specificity = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 and fp == 0 else 0.0)

    pred_boxes = [mask_to_bbox(mask) for mask in pred_masks]
    gt_boxes = [mask_to_bbox(mask) for mask in gt_masks]

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "matches": matches,
        "pred_boxes": pred_boxes,
        "gt_boxes": gt_boxes,
        "pred_union": pred_union,
        "gt_union": gt_union,
        "pred_count": len(pred_masks),
        "gt_count": len(gt_masks),
        "max_iou": max_iou,
        "rejected_count": rejected_count,  # Number of worms rejected by stage 2 (not counted as predictions)
    }


def show_mask_overlap_debug(frame_key, metrics):
    if metrics is None:
        print(f"[MWA Viewer] No metrics available to display for frame '{frame_key}'.")
        return

    print(
        f"[MWA Viewer] Frame '{frame_key}': pred worms={metrics.get('pred_count', 'n/a')}, "
        f"mask components={metrics.get('gt_count', 'n/a')}, max IoU={metrics.get('max_iou', 0.0):.3f}"
    )

    pred_union = metrics.get("pred_union")
    gt_union = metrics.get("gt_union")
    if pred_union is None or gt_union is None:
        print(f"[MWA Viewer] Missing union masks for frame '{frame_key}'.")
        return

    if pred_union.shape != gt_union.shape:
        print(f"[MWA Viewer] Shape mismatch for debug overlay on frame '{frame_key}'.")
        return

    height, width = pred_union.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)

    pred_bool = pred_union.astype(bool)
    gt_bool = gt_union.astype(bool)
    overlap = np.logical_and(pred_bool, gt_bool)

    overlay[pred_bool] = (0, 0, 255)     # Red for predictions
    overlay[gt_bool] = (0, 255, 0)       # Green for masks
    overlay[overlap] = (0, 255, 255)     # Yellow where they overlap

    window_name = f"Mask Debug: {frame_key}"
    cv2.imshow(window_name, overlay)
    print(f"[MWA Viewer] Showing mask debug overlay for '{frame_key}'. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


class ManualClassificationDialog:
    """Modal dialog for manually classifying image patches."""
    
    CATEGORIES = {
        "TP": {"name": "True Positive", "color": (0, 255, 0), "label": "TP"},
        "TN": {"name": "True Negative", "color": (144, 238, 144), "label": "TN"},  # Lighter green
        "FP": {"name": "False Positive", "color": (0, 0, 255), "label": "FP"},
        "FN": {"name": "False Negative", "color": (0, 165, 255), "label": "FN"},  # Orange
    }
    
    PATCH_SIZE = 48
    DISPLAY_SIZE = 300
    WINDOW_HEIGHT = 450  # Increased to accommodate confirmation bar below image
    WINDOW_WIDTH = 600
    
    def __init__(self, raw_image, center_x, center_y, source_folder, original_filename):
        """
        Initialize the classification dialog.
        
        Args:
            raw_image: Full-resolution raw image (no overlays)
            center_x: X coordinate in full-resolution image space
            center_y: Y coordinate in full-resolution image space
            source_folder: Base directory where images are stored
            original_filename: Original filename (e.g., "frame_001.png")
        """
        self.raw_image = raw_image
        self.center_x = int(center_x)
        self.center_y = int(center_y)
        self.source_folder = source_folder
        self.original_filename = original_filename
        
        # Extract patch (handling edge cases)
        self.patch = self._extract_patch()
        if self.patch is None:
            raise ValueError("Cannot extract patch from image")
        
        self.selected_category = None
        self.confirmed = False
        self.cancelled = False
        
    def _extract_patch(self):
        """Extract a PATCH_SIZE x PATCH_SIZE patch from the raw image, centered on the clicked location."""
        half_size = self.PATCH_SIZE // 2
        img_h, img_w = self.raw_image.shape[:2]
        
        # Calculate ideal bounds centered on click
        ideal_x0 = self.center_x - half_size
        ideal_y0 = self.center_y - half_size
        ideal_x1 = self.center_x + half_size
        ideal_y1 = self.center_y + half_size
        
        # Start with ideal bounds and adjust only if necessary to stay within image bounds
        x0 = ideal_x0
        y0 = ideal_y0
        x1 = ideal_x1
        y1 = ideal_y1
        
        # If patch would go outside image bounds, shift it to stay within bounds
        # Try to keep the clicked point as centered as possible
        if x0 < 0:
            # Too far left - shift right
            shift = -x0
            x0 = 0
            x1 = min(img_w, x1 + shift)
        if x1 > img_w:
            # Too far right - shift left
            shift = x1 - img_w
            x1 = img_w
            x0 = max(0, x0 - shift)
        
        if y0 < 0:
            # Too far up - shift down
            shift = -y0
            y0 = 0
            y1 = min(img_h, y1 + shift)
        if y1 > img_h:
            # Too far down - shift up
            shift = y1 - img_h
            y1 = img_h
            y0 = max(0, y0 - shift)
        
        # Ensure we have at least PATCH_SIZE pixels in both dimensions
        # This handles edge cases where the image is smaller than PATCH_SIZE
        actual_w = x1 - x0
        actual_h = y1 - y0
        
        if actual_w < self.PATCH_SIZE:
            # Need to adjust width - try to keep clicked point centered
            if x0 == 0:
                # At left edge, extend rightward
                x1 = min(img_w, self.PATCH_SIZE)
                x0 = 0
            elif x1 == img_w:
                # At right edge, extend leftward
                x0 = max(0, img_w - self.PATCH_SIZE)
                x1 = img_w
            else:
                # In middle, center on clicked point if possible
                x0 = max(0, self.center_x - half_size)
                x1 = min(img_w, x0 + self.PATCH_SIZE)
                if x1 - x0 < self.PATCH_SIZE:
                    x0 = max(0, img_w - self.PATCH_SIZE)
                    x1 = img_w
        
        if actual_h < self.PATCH_SIZE:
            # Need to adjust height - try to keep clicked point centered
            if y0 == 0:
                # At top edge, extend downward
                y1 = min(img_h, self.PATCH_SIZE)
                y0 = 0
            elif y1 == img_h:
                # At bottom edge, extend upward
                y0 = max(0, img_h - self.PATCH_SIZE)
                y1 = img_h
            else:
                # In middle, center on clicked point if possible
                y0 = max(0, self.center_y - half_size)
                y1 = min(img_h, y0 + self.PATCH_SIZE)
                if y1 - y0 < self.PATCH_SIZE:
                    y0 = max(0, img_h - self.PATCH_SIZE)
                    y1 = img_h
        
        # Extract patch
        patch = self.raw_image[y0:y1, x0:x1].copy()
        
        # Ensure patch is exactly PATCH_SIZE x PATCH_SIZE (resize if necessary)
        if patch.shape[0] != self.PATCH_SIZE or patch.shape[1] != self.PATCH_SIZE:
            patch = cv2.resize(patch, (self.PATCH_SIZE, self.PATCH_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Update center to actual center of extracted patch (before resize)
        self.actual_center_x = (x0 + x1) // 2
        self.actual_center_y = (y0 + y1) // 2
        
        return patch
    
    def _create_display_image(self):
        """Create the display image with buttons and confirmation bar."""
        display_img = np.ones((self.WINDOW_HEIGHT, self.WINDOW_WIDTH, 3), dtype=np.uint8) * 255
        
        # Upsample patch to display size
        patch_display = cv2.resize(self.patch, (self.DISPLAY_SIZE, self.DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        
        # Place patch on left side
        patch_y = 50  # Leave room for "What is this?" text
        display_img[patch_y:patch_y+self.DISPLAY_SIZE, 0:self.DISPLAY_SIZE] = patch_display
        
        # Add "What is this?" text at top
        text = "What is this?"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.0, 2)[0]
        text_x = (self.DISPLAY_SIZE - text_size[0]) // 2
        cv2.putText(display_img, text, (text_x, 35), font, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Draw buttons on the right
        button_start_x = self.DISPLAY_SIZE + 20
        button_width = self.WINDOW_WIDTH - button_start_x - 20
        button_height = 50
        button_spacing = 10
        button_y_start = 50
        
        self.button_rects = {}
        for i, (key, info) in enumerate(self.CATEGORIES.items()):
            button_y = button_y_start + i * (button_height + button_spacing)
            button_rect = (button_start_x, button_y, button_start_x + button_width, button_y + button_height)
            self.button_rects[key] = button_rect
            
            # Draw button background
            color = info["color"]
            cv2.rectangle(display_img, (button_start_x, button_y), 
                         (button_start_x + button_width, button_y + button_height), 
                         color, -1)
            
            # Draw border if selected - thick black border for selected button
            border_color = (0, 0, 0)  # Always black
            border_thickness = 5 if self.selected_category == key else 1
            cv2.rectangle(display_img, (button_start_x, button_y), 
                         (button_start_x + button_width, button_y + button_height), 
                         border_color, border_thickness)
            
            # Add text
            text = info["name"]
            text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
            text_x = button_start_x + (button_width - text_size[0]) // 2
            text_y = button_y + (button_height + text_size[1]) // 2
            cv2.putText(display_img, text, (text_x, text_y), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Cancel button
        cancel_y = button_y_start + len(self.CATEGORIES) * (button_height + button_spacing) + 20
        cancel_rect = (button_start_x, cancel_y, button_start_x + button_width, cancel_y + button_height)
        self.button_rects["Cancel"] = cancel_rect
        cv2.rectangle(display_img, (button_start_x, cancel_y), 
                     (button_start_x + button_width, cancel_y + button_height), 
                     (128, 128, 128), -1)
        cv2.rectangle(display_img, (button_start_x, cancel_y), 
                     (button_start_x + button_width, cancel_y + button_height), 
                     (0, 0, 0), 1)
        text = "Cancel"
        text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
        text_x = button_start_x + (button_width - text_size[0]) // 2
        text_y = cancel_y + (button_height + text_size[1]) // 2
        cv2.putText(display_img, text, (text_x, text_y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Confirmation bar at bottom (only visible if category selected)
        if self.selected_category is not None:
            bar_height = 40
            # Position bar below the image (patch_y + DISPLAY_SIZE) with some spacing
            patch_bottom = patch_y + self.DISPLAY_SIZE
            bar_y = patch_bottom + 20  # 20 pixels spacing below the image
            # Use the same color as the selected button
            button_color = self.CATEGORIES[self.selected_category]["color"]
            cv2.rectangle(display_img, (10, bar_y), (self.WINDOW_WIDTH - 10, bar_y + bar_height), 
                         button_color, -1)
            cv2.rectangle(display_img, (10, bar_y), (self.WINDOW_WIDTH - 10, bar_y + bar_height), 
                         (0, 0, 0), 2)
            
            category_name = self.CATEGORIES[self.selected_category]["name"]
            confirm_text = f"Click here to confirm {category_name}"
            text_size = cv2.getTextSize(confirm_text, font, 0.7, 2)[0]
            text_x = (self.WINDOW_WIDTH - text_size[0]) // 2
            text_y = bar_y + (bar_height + text_size[1]) // 2
            # Use white text for better visibility on colored background
            cv2.putText(display_img, confirm_text, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            self.confirm_rect = (10, bar_y, self.WINDOW_WIDTH - 10, bar_y + bar_height)
        else:
            self.confirm_rect = None
        
        return display_img
    
    def _handle_click(self, x, y):
        """Handle mouse click in the dialog window."""
        # Check category buttons
        for key, rect in self.button_rects.items():
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                if key == "Cancel":
                    self.cancelled = True
                    return True
                else:
                    self.selected_category = key
                    return False
        
        # Check confirmation bar
        if self.confirm_rect is not None and self.selected_category is not None:
            x0, y0, x1, y1 = self.confirm_rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.confirmed = True
                return True
        
        return False
    
    def show(self):
        """Show the modal classification dialog and return the result."""
        window_name = "Manual Classification"
        cv2.namedWindow(window_name)
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._handle_click(x, y):
                    return
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            # Check if window was closed (returns -1 if window doesn't exist)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.cancelled = True
                break
            
            display_img = self._create_display_image()
            cv2.imshow(window_name, display_img)
            
            key = cv2.waitKey(30) & 0xFF
            
            # ESC key cancels
            if key == 27:
                self.cancelled = True
                break
            
            # Check if confirmed or cancelled via mouse
            if self.confirmed or self.cancelled:
                break
        
        try:
            cv2.destroyWindow(window_name)
        except:
            pass  # Window may already be closed
        
        if self.confirmed:
            self._save_patch()
            return True
        
        return False
    
    def _save_patch(self):
        """Save the classified patch to the appropriate subfolder."""
        if self.selected_category is None:
            return
        
        category_name = self.CATEGORIES[self.selected_category]["name"]
        category_label = self.CATEGORIES[self.selected_category]["label"]
        
        # Create subfolder structure: ManualClassification/<Category>/
        manual_dir = os.path.join(self.source_folder, "ManualClassification", category_name)
        os.makedirs(manual_dir, exist_ok=True)
        
        # Generate filename: <original_filename>_<label>_x<x>_y<y>_w<size>.png
        # Remove extension from original filename
        base_name = os.path.splitext(os.path.basename(self.original_filename))[0]
        filename = f"{base_name}_{category_label}_x{self.actual_center_x}_y{self.actual_center_y}_w{self.PATCH_SIZE}.png"
        
        output_path = os.path.join(manual_dir, filename)
        cv2.imwrite(output_path, self.patch)
        print(f"[MWA Viewer] Saved classification to: {output_path}")


# Mouse callback
def onMouse(event, x, y, flags, param):
    global click_pt, view_frac, zoom_lvl, zoom_lvls, mouse_context

    # Update the click position every time the mouse moves
    if event == cv2.EVENT_MOUSEMOVE:
        click_pt = [int(x/view_frac), int(y/view_frac)]

    # Handle double-click for manual classification
    if event == EVENT_LBUTTONDBLCLK:
        # Get context from global mouse_context
        raw_image = mouse_context.get("raw_image")
        source_folder = mouse_context.get("source_folder")
        current_filename = mouse_context.get("current_filename")
        zoom_info = mouse_context.get("zoom_info", {})
        
        if raw_image is not None and source_folder is not None and current_filename is not None:
            # Calculate full-resolution coordinates based on zoom state
            # x, y are in window coordinate space (0 to WINDOW_WIDTH, 0 to WINDOW_HEIGHT)
            
            # If zoomed, we need to account for the zoom transform
            if zoom_info.get("zoomed", False) and zoom_info.get("crop_rect") is not None:
                # The displayed image is a cropped region that was resized to fit the window
                crop_x0, crop_y0, crop_x1, crop_y1 = zoom_info["crop_rect"]
                crop_w = crop_x1 - crop_x0
                crop_h = crop_y1 - crop_y0
                window_w = zoom_info.get("window_width", 1000)
                window_h = zoom_info.get("window_height", 1000)
                
                # Convert window coordinates to crop coordinates
                crop_x = (x / window_w) * crop_w
                crop_y = (y / window_h) * crop_h
                
                # Convert crop coordinates to full-resolution coordinates
                img_coord_x = int(crop_x0 + crop_x)
                img_coord_y = int(crop_y0 + crop_y)
            else:
                # Not zoomed, calculate scale factors from actual window dimensions
                window_w = zoom_info.get("window_width", 1000)
                window_h = zoom_info.get("window_height", 1000)
                # Convert window coordinates to full-resolution image coordinates
                # The image was resized to (window_w, window_h), so scale back
                img_coord_x = int(x * (raw_image.shape[1] / window_w))
                img_coord_y = int(y * (raw_image.shape[0] / window_h))
            
            try:
                dialog = ManualClassificationDialog(
                    raw_image=raw_image,
                    center_x=img_coord_x,
                    center_y=img_coord_y,
                    source_folder=source_folder,
                    original_filename=current_filename
                )
                dialog.show()
            except Exception as e:
                print(f"[MWA Viewer] Error in manual classification: {e}")

    if event == cv2.EVENT_MOUSEWHEEL:
        # The event data for mouse scroll is encoded in flags
        # A positive value indicates scrolling up, and a negative value indicates scrolling down
        old_zoom_lvl = copy.deepcopy(zoom_lvl)
        if flags > 0:  # Scrolling up
            zoom_lvl += 1
        else:  # Scrolling down
            zoom_lvl -= 1

        # Restrict zoom_lvl to within the available zoom levels
        zoom_lvl = max(0, min(zoom_lvl, len(zoom_lvls) - 1))

        # If we are zoomed, the click point is relative to the last one
        if old_zoom_lvl != zoom_lvl:
            click_pt = [int(x/view_frac), int(y/view_frac)]



def MultiWellAnalysisViewer():
    global view_frac, click_pt, zoom_lvl, zoom_lvls, mask_debug

    # Load corresponding JSON file
    fjson = os.path.join(MWA.plate_params["paths"][MWA.path_key],"_Analysis","analysis_dump.json")
    if not os.path.exists(fjson):
        msg = "Could not find file:\n\n" + fjson + "\n\nPlease make sure to fully analyze this dataset and try again."
        messagebox.showerror("Could not find JSON file",msg)
        exit(0)
    D = loadJSON(fjson)
    MD= loadJSON(fjson.replace("_dump","_metadata"))

    # Interactive preview
    keys = list(D)
    base_path = MWA.plate_params["paths"][MWA.path_key]
    mask_dir = os.path.join(base_path, "Masks")
    
    # Check if Masks folder exists
    masks_available = os.path.exists(mask_dir) and os.path.isdir(mask_dir)

    # Instructions
    if not MultiWellAnalysis.skip_menus:
        msg = "Use the [ and ] keys to go back / advance one frame.\n\nUse the { and } keys to go back / advance 10 frames (hold shift)"
        msg += "\n\npress the o key to show or hide overlays (worm outlines)."
        msg += "\n\nuse the SCROLL WHEEL to zoom in or out. While zoomed in, move the mouse to pan."
        msg += "\n\nDOUBLE-CLICK anywhere on the image to manually classify a 48x48 patch (TP/TN/FP/FN)."
        if masks_available:
            msg += "\n\nPlace per-frame mask images in the 'Masks' subfolder alongside the raw frames."
            msg += "\n\npress the m key to toggle mask bounding box diagnostics (requires Masks folder)."
            msg += "\n\npress the d key to view a mask overlap debug image (pred=red, mask=green, overlap=yellow)."
        msg += "\n\nUse ESC to exit"
        messagebox.showinfo("Instructions",msg)
    
    frame_metrics = {}
    total_tp = total_fp = total_fn = 0
    frames_with_masks = 0
    frames_missing_masks = []
    frames_zero_matches = []
    total_pred_worms = 0
    total_gt_worms = 0
    total_rejected_worms = 0

    if masks_available:
        print(f"[MWA Viewer] Expecting per-frame masks in: {mask_dir}")
    else:
        print(f"[MWA Viewer] Masks folder not found at: {mask_dir}. Skipping mask processing.")
        mask_debug = False  # Disable mask debug mode

    if masks_available:
        for key in keys:
            fimg = os.path.join(base_path, key)
            img = cv2.imread(fimg)
            if img is None:
                print(f"[MWA Viewer] Warning: could not load source image '{key}'. Skipping frame for dataset stats.")
                frame_metrics[key] = None
                continue
            height, width = img.shape[:2]
            mask_binary, mask_label = load_masks(base_path, key, (height, width))
            metrics = compute_detection_metrics(D[key], (height, width), mask_binary, mask_label)
            frame_metrics[key] = metrics
            if metrics is None:
                frames_missing_masks.append(key)
                continue
            frames_with_masks += 1
            total_pred_worms += metrics["pred_count"]
            total_gt_worms += metrics["gt_count"]
            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]
            total_rejected_worms += metrics.get("rejected_count", 0)
            if metrics["tp"] == 0:
                frames_zero_matches.append(
                    (key, metrics.get("max_iou", 0.0), metrics.get("pred_count"), metrics.get("gt_count"))
                )
    else:
        # If masks not available, initialize frame_metrics with None for all frames
        for key in keys:
            frame_metrics[key] = None

    total_frames = len(keys)
    frames_without_masks = total_frames - frames_with_masks
    print(f"[MWA Viewer] Frames with masks: {frames_with_masks}/{total_frames} ({frames_without_masks} without masks)")

    if frames_with_masks > 0:
        dataset_sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else (1.0 if total_tp == 0 and total_fn == 0 else 0.0)
        dataset_specificity = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else (1.0 if total_tp == 0 and total_fp == 0 else 0.0)
        print(f"[MWA Viewer] Dataset sensitivity (recall @IoU>=0.05): {dataset_sensitivity:.3f}")
        print(f"[MWA Viewer] Dataset specificity (precision @IoU>=0.05): {dataset_specificity:.3f}")
        print(f"[MWA Viewer] Total predicted worms (dataset, after stage 2 filtering): {total_pred_worms}")
        print(f"[MWA Viewer] Total rejected worms (stage 2, not counted as predictions): {total_rejected_worms}")
        print(f"[MWA Viewer] Total mask worms (dataset): {total_gt_worms}")
        print(f"[MWA Viewer] Aggregate TP={total_tp}, FP={total_fp}, FN={total_fn}")
    else:
        print("[MWA Viewer] No masks found; skipping dataset-level metrics.")
        print(f"[MWA Viewer] Total predicted worms (dataset): {total_pred_worms}")
        print(f"[MWA Viewer] Total mask worms (dataset): {total_gt_worms}")
        print(f"[MWA Viewer] Aggregate TP={total_tp}, FP={total_fp}, FN={total_fn}")

    if frames_without_masks and frames_missing_masks:
        preview = ", ".join(frames_missing_masks[:5])
        if preview:
            print(f"[MWA Viewer] Example frames missing masks: {preview}")

    if frames_zero_matches:
        sample = ", ".join(
            f"{name} (maxIoU={max_iou:.2f}, pred={pred}, gt={gt})"
            for name, max_iou, pred, gt in frames_zero_matches[:5]
        )
        print(
            f"[MWA Viewer] Frames with zero matches despite masks: {len(frames_zero_matches)} "
            f"(showing up to 5): {sample}"
        )

    print("[MWA Viewer] Specificity reported here reflects detection precision (true negatives are undefined).")

    Nf = len(keys)-1
    i = 0
    overlays = True
    cv2.namedWindow("MWA Viewer")
    cv2.setMouseCallback("MWA Viewer", onMouse)

    while(True):

        # Load image
        fimg = os.path.join(base_path, keys[i])
        img_load = cv2.imread(fimg)
        if img_load is None:
            print(f"[MWA Viewer] Warning: could not load image '{keys[i]}'. Skipping to next frame.")
            # Skip to next frame (or previous if at the end)
            if i < Nf:
                i = i + 1
            else:
                i = max(0, i - 1)
            continue
        
        if img_load.ndim < 3: img_load = cv2.cvtColor(img_load,cv2.COLOR_GRAY2BGR)
        
        # Update mouse context for double-click classification
        global mouse_context
        mouse_context = {
            "raw_image": img_load,  # Full-resolution raw image (no overlays)
            "source_folder": base_path,
            "current_filename": keys[i]
        }
        
        # Display until keypress or mouse click
        WINDOW_HEIGHT = 1000  # Change to your preferred window height
        height, width, _ = img_load.shape
        WINDOW_WIDTH = int(WINDOW_HEIGHT * (width / height))

        Dframe = D[keys[i]]
        detection_metrics = frame_metrics.get(keys[i])

        while True:

            # Fresh copy the image is required for overlays to redraw properly 
            img = copy.deepcopy(img_load)

            # Load metadata for this frame and draw wells and worms
            zoom_rad = int(Dframe["Well 1"]["well circle"][2]*zoom_lvls[zoom_lvl])

            # Annotate
            img = Plate.drawFromDict(img,Dframe,local = False)
            rad = 4 if zoom_lvl > 0 else 4+int(img.shape[1]/1000)

            if overlays:
                img = MultiWellAnalysis.drawFromDict(img, Dframe, local=False, rad=rad)

            if masks_available and mask_debug and detection_metrics is not None:
                matched_preds = {idx for idx, _, _ in detection_metrics["matches"]}
                matched_gts = {idx for _, idx, _ in detection_metrics["matches"]}
                for gt_idx, bbox in enumerate(detection_metrics["gt_boxes"]):
                    if bbox is None:
                        continue
                    color = (0, 255, 0) if gt_idx in matched_gts else (0, 165, 255)
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                for pred_idx, bbox in enumerate(detection_metrics["pred_boxes"]):
                    if bbox is None or pred_idx in matched_preds:
                        continue
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

            view_frac = 900/img.shape[0]
        
            # Zoom if requested
            zoom_crop_rect = None
            if zoom_lvl>0:
                x0 = max((0,click_pt[0]-zoom_rad))
                x1 = min((img.shape[1]-1,click_pt[0]+zoom_rad))
                y0 = max((0,click_pt[1]-zoom_rad))
                y1 = min((img.shape[0]-1,click_pt[1]+zoom_rad))
                if x1 > x0 and y1 > y0:  # Make sure zoom region is valid before adjusting size
                    img_show = img[y0:y1,x0:x1].copy()
                    img_show = cv2.convertScaleAbs(img_show, alpha=MD["display"]["alpha"], beta=MD["display"]["beta"]) # Adjust color
                    zoom_crop_rect = (x0, y0, x1, y1)

                    # if overlays:    
                    #     img_show = MultiWellAnalysis.drawFromDict(img_show, Dframe, local=False, rad=rad)
                else:
                    img_show = img.copy()

            else:
                img_show = img.copy()
            
            # Update zoom info in mouse context for coordinate transformation
            mouse_context["zoom_info"] = {
                "zoomed": zoom_lvl > 0 and zoom_crop_rect is not None,
                "crop_rect": zoom_crop_rect,
                "window_width": WINDOW_WIDTH,
                "window_height": WINDOW_HEIGHT
            }

            # Resize image to fixed size before showing it
            img_show = cv2.resize(img_show, (WINDOW_WIDTH, WINDOW_HEIGHT))
            #img_show = cv2.putText(img_show,"t=%d sec / %0.2f hrs - double click to zoom" % (Dframe['t'], Dframe['t']/3600),(10,45),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0,255,0),2)
            status_text = f"{keys[i]} | o overlays | [ ] frame | SCROLL zoom"
            if masks_available:
                status_text += " | m mask boxes | d overlap img"
            img_show = cv2.putText(img_show, status_text, (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
            # Add color legend for secondary classification
            legend_y = 80
            img_show = cv2.putText(img_show,"Green: Confirmed worms",(10,legend_y),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,255,0),2)
            img_show = cv2.putText(img_show,"Red: Rejected worms (false positives)",(10,legend_y+25),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,0,255),2)
            img_show = cv2.putText(img_show,"Blue dots: Worm centroids",(10,legend_y+50),cv2.FONT_HERSHEY_SIMPLEX,0.6, (255,0,0),2)

            metrics_y = legend_y + 80
            if masks_available:
                if detection_metrics is not None:
                    rejected_info = ""
                    if detection_metrics.get("rejected_count", 0) > 0:
                        rejected_info = f" | Rejected: {detection_metrics['rejected_count']}"
                    metrics_text = (
                        f"Sens (recall): {detection_metrics['sensitivity']:.2f} "
                        f"| Spec (precision): {detection_metrics['specificity']:.2f} "
                        f"| TP {detection_metrics['tp']} FP {detection_metrics['fp']} FN {detection_metrics['fn']}"
                        f"{rejected_info}"
                    )
                else:
                    metrics_text = "Mask not found for this frame; add to Masks/ subfolder to enable metrics."
                img_show = cv2.putText(img_show,metrics_text,(10,metrics_y),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0,255,255),2)

            

            cv2.imshow("MWA Viewer",img_show)
            k = cv2.waitKey(30)

            if k > 0: break

        if k == ord("["):    i = max((0,i-1))
        elif k == ord("]"):  i = min((i+1,Nf))
        elif k == ord("{"):  i = max((0,i-10))
        elif k == ord("}"):  i = min((i+10,Nf))
        elif k == ord("o"):  overlays = not overlays
        elif k == ord("m") and masks_available:  mask_debug = not mask_debug
        elif k == ord("d") and masks_available:
            metrics = frame_metrics.get(keys[i])
            if metrics is None:
                print(f"[MWA Viewer] Cannot display overlap for '{keys[i]}' (missing mask or source image).")
            else:
                show_mask_overlap_debug(keys[i], metrics)
        elif k == 27: exit(0)


if __name__ == "__main__":
    MultiWellAnalysisViewer()
