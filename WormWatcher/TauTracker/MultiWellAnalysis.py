# Multiwell analysis
#
# Holds individual worm trackers for analyzing many wells of a worm plate

from torch import save
from Plate import *
from WormTracker import *
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import scipy.signal
import copy
import traceback
import datetime
import cv2

from tkinter import filedialog, messagebox
import tkinter as tk

# Import YOLO8 functions
from YOLO8.YOLO8_Helper import loadYOLO8
from YOLO8.YOLO8_Segment import YOLO8_Segment

# Force UI windows to top
root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)


# TODO: Implement an Anlysis object for every type of analysis that might need to be performed in
# a WellAnalysis, such as Lawn Distance measurement (average for all worms in well) or a neighbor
# distance analysis (average for the well)
class Analysis:
    pass

# WellAnalysis is the basic container for an analysis of Worm in one well in one frame.
class WellAnalysis:

    def __init__(self, well:Well, seg_params:dict, sliding_window_size:int = None):

        # Store the well coordinates / geometry
        self.well = well
        self.seg_params = seg_params
        # Get sliding window size from seg_params if not provided, default to 512
        self.sliding_window_size = sliding_window_size if sliding_window_size is not None else seg_params.get("sliding window size", 512)
        self.WT = WormTracker()
        
        # Initialize secondary classifier attributes to None - will be set later if needed
        self.secondary_classifier = None
        self.secondary_classifier_window_size = None
        self.secondary_classification_mode = None 

    def _extract_overlapping_tiles(self, img, tile_size, overlap_ratio=0.25):
        """
        Extract overlapping tiles from an image for sliding window processing.
        
        Args:
            img: Input image
            tile_size: Size of each tile (square)
            overlap_ratio: Fraction of overlap between adjacent tiles
            
        Returns:
            List of (tile_img, x_offset, y_offset) tuples
        """
        h, w = img.shape[:2]
        tiles = []
        
        # Calculate step size (tile_size - overlap)
        step_size = int(tile_size * (1 - overlap_ratio))
        
        # Generate tiles with overlap
        for y in range(0, h - tile_size + 1, step_size):
            for x in range(0, w - tile_size + 1, step_size):
                tile = img[y:y+tile_size, x:x+tile_size]
                tiles.append((tile, x, y))
        
        # Handle edge cases - add tiles for remaining areas
        # Right edge
        if w % step_size != 0:
            x = w - tile_size
            for y in range(0, h - tile_size + 1, step_size):
                tile = img[y:y+tile_size, x:x+tile_size]
                tiles.append((tile, x, y))
        
        # Bottom edge
        if h % step_size != 0:
            y = h - tile_size
            for x in range(0, w - tile_size + 1, step_size):
                tile = img[y:y+tile_size, x:x+tile_size]
                tiles.append((tile, x, y))
        
        # Bottom-right corner
        if w % step_size != 0 and h % step_size != 0:
            x = w - tile_size
            y = h - tile_size
            tile = img[y:y+tile_size, x:x+tile_size]
            tiles.append((tile, x, y))
            
        return tiles

    def _stitch_masks(self, tile_results, img_shape, tile_size, overlap_ratio=0.25):
        """
        Efficiently stitch together masks from overlapping tiles, handling duplicates.
        
        Args:
            tile_results: List of (masks_thresh, x_offset, y_offset) tuples
            img_shape: Shape of the original image (H, W)
            tile_size: Size of each tile
            overlap_ratio: Overlap ratio used in tile extraction
            
        Returns:
            Combined masks_thresh array
        """
        h, w = img_shape[:2]
        
        # Lightweight detection storage: avoid creating full global masks
        detections = []
        
        # Extract all detections with local coordinates
        for masks_thresh, x_offset, y_offset in tile_results:
            if masks_thresh is None or masks_thresh.shape[3] == 0:
                continue
                
            # For each worm in this tile
            for i in range(masks_thresh.shape[3]):
                local_mask = masks_thresh[:, :, 0, i]
                if np.sum(local_mask) == 0:
                    continue
                
                # Calculate centroid and bounding box from local mask (much faster)
                ys, xs = np.where(local_mask > 0)
                if len(xs) == 0:
                    continue
                    
                # Local centroid and bounding box
                local_cx = np.mean(xs)
                local_cy = np.mean(ys)
                local_area = len(xs)
                
                # Global coordinates
                global_cx = local_cx + x_offset
                global_cy = local_cy + y_offset
                
                # Tight bounding box
                bbox = (np.min(xs) + x_offset, np.min(ys) + y_offset, 
                       np.max(xs) + x_offset, np.max(ys) + y_offset)
                
                detections.append({
                    'local_mask': local_mask,
                    'x_offset': x_offset,
                    'y_offset': y_offset,
                    'global_cx': global_cx,
                    'global_cy': global_cy,
                    'area': local_area,
                    'bbox': bbox
                })
        
        if not detections:
            return np.zeros((h, w, 1, 0), dtype=np.uint8)
        
        print(f"  Found {len(detections)} detections, removing duplicates...")
        
        # Get overlap threshold from parameters (same as used in per-tile NMS)
        overlap_thresh = self.seg_params.get("overlap thresh", 0.25)
        
        # Fast NMS using spatial indexing and bounding box overlap (intersection over min area)
        min_distance = tile_size * overlap_ratio * 0.5
        
        def bbox_overlap_imin(bbox1, bbox2):
            """Fast bounding box overlap calculation: intersection_area / min(area1, area2)
            Matches the overlap metric used in YOLO8_Segment.py _apply_mask_nms"""
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            
            # Calculate intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
                return 0.0
            
            inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            
            min_area = min(box1_area, box2_area)
            if min_area > 0:
                return inter_area / min_area
            return 0.0
        
        # Sort detections by area (descending) for better NMS
        detections.sort(key=lambda x: x['area'], reverse=True)
        
        # Efficient NMS using spatial hashing
        grid_size = int(min_distance)
        spatial_grid = {}
        
        def get_grid_key(cx, cy):
            return (int(cx // grid_size), int(cy // grid_size))
        
        def get_nearby_grid_keys(cx, cy):
            base_key = get_grid_key(cx, cy)
            keys = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    keys.append((base_key[0] + dx, base_key[1] + dy))
            return keys
        
        filtered_detections = []
        
        for detection in detections:
            cx, cy = detection['global_cx'], detection['global_cy']
            
            # Check nearby grid cells for duplicates
            is_duplicate = False
            nearby_keys = get_nearby_grid_keys(cx, cy)
            
            for grid_key in nearby_keys:
                if grid_key in spatial_grid:
                    for other_idx in spatial_grid[grid_key]:
                        other = filtered_detections[other_idx]
                        
                        # Quick distance check
                        distance = np.sqrt((cx - other['global_cx'])**2 + 
                                         (cy - other['global_cy'])**2)
                        
                        if distance < min_distance:
                            # Use bounding box overlap (intersection over min area) as filter
                            # Matches the overlap metric used in per-tile NMS
                            bbox_overlap = bbox_overlap_imin(detection['bbox'], other['bbox'])
                            if bbox_overlap >= overlap_thresh:  # Use parameter threshold for considering as duplicate
                                is_duplicate = True
                                break
                
                if is_duplicate:
                    break
            
            if not is_duplicate:
                # Add to filtered list and spatial grid
                det_idx = len(filtered_detections)
                filtered_detections.append(detection)
                
                grid_key = get_grid_key(cx, cy)
                if grid_key not in spatial_grid:
                    spatial_grid[grid_key] = []
                spatial_grid[grid_key].append(det_idx)
        
        print(f"  Reduced to {len(filtered_detections)} unique detections")
        
        # Only now create the final global masks
        if not filtered_detections:
            return np.zeros((h, w, 1, 0), dtype=np.uint8)
        
        n_masks = len(filtered_detections)
        combined_masks = np.zeros((h, w, 1, n_masks), dtype=np.uint8)
        
        for i, detection in enumerate(filtered_detections):
            local_mask = detection['local_mask']
            x_offset = detection['x_offset']
            y_offset = detection['y_offset']
            
            # Create global mask only for final results
            y_end = min(y_offset + local_mask.shape[0], h)
            x_end = min(x_offset + local_mask.shape[1], w)
            local_h = y_end - y_offset
            local_w = x_end - x_offset
            
            combined_masks[y_offset:y_end, x_offset:x_end, 0, i] = local_mask[:local_h, :local_w]
        
        return combined_masks

    def _process_with_sliding_window(self, img_crop, model):
        """
        Process large image using sliding window approach.
        
        Args:
            img_crop: Cropped well image
            model: YOLO8 model
            
        Returns:
            Combined masks_thresh, weighted, weighted_touch, touch_count
        """
        print(f"Using sliding window processing with {self.sliding_window_size}x{self.sliding_window_size} tiles")
        
        # Extract overlapping tiles
        tiles = self._extract_overlapping_tiles(img_crop, self.sliding_window_size)
        total_tiles = len(tiles)
        print(f"Processing {total_tiles} tiles...")
        
        # Process each tile with progress tracking
        tile_results = []
        all_touch_counts = []
        
        start_time = time.time()
        
        for tile_idx, (tile_img, x_offset, y_offset) in enumerate(tiles):
            tile_start_time = time.time()
            
            try:
                # Process this tile with YOLO8
                masks, masks_thresh, weighted, weighted_touch, touch_count = YOLO8_Segment(
                    model, tile_img, self.seg_params
                )
                
                tile_results.append((masks_thresh, x_offset, y_offset))
                if touch_count is not None:
                    all_touch_counts.extend(touch_count)
                    
            except Exception as e:
                print(f"Error processing tile at ({x_offset}, {y_offset}): {e}")
                continue
            
            # Calculate and display progress every 10 tiles
            tiles_completed = tile_idx + 1
            elapsed_time = time.time() - start_time
            
            if tiles_completed > 0 and (tiles_completed % 20 == 0 or tiles_completed == total_tiles):
                avg_time_per_tile = elapsed_time / tiles_completed
                remaining_tiles = total_tiles - tiles_completed
                estimated_remaining_time = avg_time_per_tile * remaining_tiles
                
                # Format time nicely
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.1f}s"
                    elif seconds < 3600:
                        mins = int(seconds // 60)
                        secs = int(seconds % 60)
                        return f"{mins}m {secs}s"
                    else:
                        hours = int(seconds // 3600)
                        mins = int((seconds % 3600) // 60)
                        return f"{hours}h {mins}m"
                
                print(f"  Tile {tiles_completed}/{total_tiles} completed "
                      f"(~{format_time(estimated_remaining_time)} remaining)")
        
        print(f"Completed all {total_tiles} tiles in {format_time(time.time() - start_time)}")
        
        # Stitch results together
        print("Stitching tile results...")
        combined_masks_thresh = self._stitch_masks(tile_results, img_crop.shape, self.sliding_window_size)
        
        # Create combined touch_count (simplified - just count from stitched masks)
        combined_touch_count = [0] * combined_masks_thresh.shape[3]  # Assume no touching for now
        
        # Create placeholder outputs (not used in the main analysis)
        combined_masks = combined_masks_thresh  # Simplified
        combined_weighted = np.sum(combined_masks_thresh, axis=3)
        combined_weighted_touch = np.zeros_like(combined_weighted)
        
        return combined_masks, combined_masks_thresh, combined_weighted, combined_weighted_touch, combined_touch_count

    # Segment worms in an image
    def segmentWormsInWell(self, img, model, show_debug = False):

        # Crop the image of the well from the big image
        img_crop = self.well.cropWell(img)
        img_crop = applyTargetMean(img_crop,self.seg_params["target mean"])

        #img_crop = normalize2uint8(img_crop)
        img_crop = cv2.GaussianBlur(img_crop,(5,5),3)

        # Check if we need to use sliding window approach
        h, w = img_crop.shape[:2]
        use_sliding_window = (h > self.sliding_window_size or w > self.sliding_window_size)
        
        if use_sliding_window:
            # Use sliding window processing for large images
            masks, masks_thresh, weighted, weighted_touch, touch_count = self._process_with_sliding_window(img_crop, model)
        else:
            # Use standard processing for smaller images
            masks, masks_thresh, weighted, weighted_touch, touch_count = YOLO8_Segment(model,img_crop,self.seg_params)

        # Compute clustering ratio
        touch_count_array = np.array(touch_count) if touch_count is not None else np.array([])
        self.N_touching = np.sum(touch_count_array > 0)
        self.N_alone = np.sum(touch_count_array == 0)
        self.N_total = len(touch_count_array)
        self.frac_touching = self.N_touching / self.N_total if self.N_total > 5 else None

        # If no worms, cannot proceed with analysis
        if self.N_total < 2 or np.sum(masks_thresh) < 3: 
            self.N_total = 0
            # Initialize default values for empty wells
            self.min_size = 0
            self.max_size = 0
            self.median_size = 0
            self.mean_distance_to_N_neighbors = np.nan
            self.N_touching = 0
            self.N_alone = 0
            self.frac_touching = None
            return

        # Compute sizes of worm-y objects
        contours, _ = cv2.findContours(np.sum(masks_thresh, axis=3).astype(np.uint8), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contours were found
        if len(contours) == 0:
            self.N_total = 0
            self.min_size = 0
            self.max_size = 0
            self.median_size = 0
            self.mean_distance_to_N_neighbors = np.nan
            self.N_touching = 0
            self.N_alone = 0
            self.frac_touching = None
            return
            
        object_sizes = [cv2.contourArea(c) for c in contours]
        
        # Check if any objects have valid areas
        if len(object_sizes) == 0 or all(size == 0 for size in object_sizes):
            self.N_total = 0
            self.min_size = 0
            self.max_size = 0
            self.median_size = 0
            self.mean_distance_to_N_neighbors = np.nan
            self.N_touching = 0
            self.N_alone = 0
            self.frac_touching = None
            return
            
        self.min_size = np.min(object_sizes)
        self.max_size = np.max(object_sizes)
        self.median_size = np.median(object_sizes)
        self.bigger_objects = [s for s in object_sizes if s > 1.8 * self.median_size]

        # Post-segment
        self.WT.postSegmentWorms(masks_thresh)

        # Filter out worms outside well boundary (only if actual well was detected)
        if self.well.radius < min(img_crop.shape[:2]) * 0.45:  # Well detected (not whole image)
            initial_count = len(self.WT.worms)
            self.WT.worms = [worm for worm in self.WT.worms 
                           if np.sqrt((worm.centroid[0] - self.well.radius)**2 + (worm.centroid[1] - self.well.radius)**2) <= self.well.radius]
            filtered_count = len(self.WT.worms)
            if initial_count > filtered_count:
                print(f"🔍 Filtered {initial_count - filtered_count} worms outside well boundary ({initial_count} → {filtered_count})")

        # Store original image for secondary classification
        self.original_image = img.copy()

        # -----------------  Compute distance to lawn center --------------- #
        if len(self.WT.worms) > 0:
            for worm in self.WT.worms:
                worm.dist_to_lawn_ctr = ptDist(worm.centroid,self.well.lawn_circle[:2])
        else:
            # No worms to calculate lawn distance for
            pass

        # -----------------  Compute distance to N nearest neighbors --------------- #

        # Extract all worm centroids
        centroids = [worm.centroid for worm in self.WT.worms]
        
        # Check if we have enough worms for neighbor analysis
        if len(centroids) < 2:
            # Not enough worms for neighbor analysis
            for worm in self.WT.worms:
                worm.mean_distance_to_N_neighbors = np.nan
            self.mean_distance_to_N_neighbors = np.nan
        else:
            # Define a function to compute the Euclidean distance between two points
            def euclidean_distance(pt1, pt2):
                return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

            N = 2  # Change this value to your desired N
            total_distance = 0
            total_worms = 0

            # Compute mean distance between each worm centroid and its N nearest neighbors
            for i, worm in enumerate(self.WT.worms):
                distances = [euclidean_distance(worm.centroid, other_centroid) for j, other_centroid in enumerate(centroids) if i != j]
                distances.sort()
                
                # Check if there's only one worm
                if len(distances) == 0:
                    worm.mean_distance_to_N_neighbors = np.nan
                    continue
                
                # Use all available neighbors if fewer than N worms are present
                nearest_N_distances = distances[:min(N, len(distances))]
                mean_distance = np.median(nearest_N_distances) # see if median has better accuracy...
                worm.mean_distance_to_N_neighbors = mean_distance

                # Update the running sum and the count
                total_distance += mean_distance
                total_worms += 1

            # Calculate the overall average distance to N nearest neighbors
            if total_worms > 0:
                overall_average_distance = total_distance / total_worms
            else:
                overall_average_distance = np.nan  # or any other default value if no worms are present
            self.mean_distance_to_N_neighbors = overall_average_distance



        # Show debug display
        if show_debug:

            # Stack a labeled and nonlabeled image side by side
            msg = "Found %d worms" % (len(self.WT.worms))
            if use_sliding_window:
                msg += " (sliding window)"
            print(msg)
            img_color = img_crop*1 # deep copy
            img_color = np.concatenate((img_color,img_color),axis=1)

            # Draw worms on labeled image
            cv2.putText(img_color,msg,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
            if len(self.WT.worms) > 0:
                self.WT.drawWorms(img_color)

                # Draw lines to the lawn center
                for worm in self.WT.worms:
                    cv2.line(img_color,worm.centroid,tuple(self.well.lawn_circle[:2]),(0,255,0),1)

            # display
            cv2.imshow("DEBUG",img_color)
            cv2.waitKey(0)
        pass

    # Store the WellAnalysis object (one well's data in one frame) as dict e.g. for saving to disk
    def toDict(self):

        # If Zero worms were found we need to fill in default values for the various variables (needed in other places too)
        if self.N_total == 0:
            self.min_size = 0
            self.max_size = 0
            self.median_size = 0
            self.mean_distance_to_N_neighbors = np.nan

        D = {}
        D["N total"] = self.N_total
        D["N touching"] = self.N_touching
        D["N alone"] = self.N_alone
        D["min worm size"] = self.min_size 
        D["max worm size"] = self.max_size 
        D["median worm size"] = self.median_size 
        # Handle np.nan values for JSON serialization
        if np.isnan(self.mean_distance_to_N_neighbors):
            D["mean dist to N nearest neighbors"] = None
        else:
            D["mean dist to N nearest neighbors"] = self.mean_distance_to_N_neighbors
        D["lawn circle"] = self.well.global_lawn_circle.tolist() 
        D["well circle"] = self.well.global_circle.tolist() 
        D["local lawn circle"] = self.well.lawn_circle.tolist() 
        D["local well circle"] = self.well.local_circle.tolist() 
        D["worms"] = self.WT.toDict()
        return D

    # Populate a Well object from a dict, e.g. which was saved to disk as JSON
    def fromDict(self):
        self.well
        pass

    def _apply_secondary_classification(self):
        """Apply secondary classifier to all detected worms in this well"""
        if self.secondary_classifier is None:
            raise ValueError("secondary_classifier must be set before calling _apply_secondary_classification")
        
        if self.secondary_classification_mode is None:
            raise ValueError("secondary_classification_mode must be set before calling _apply_secondary_classification")
        
        print(f"🔍 Applying secondary classification to {len(self.WT.worms)} worms...")
        
        # Debug counters
        predictions_0 = 0
        predictions_1 = 0
        
        # Get classification mode outside the loop to ensure it's always defined
        classification_mode = self.secondary_classification_mode
        
        for worm in self.WT.worms:
            # Extract crop around worm centroid
            crop = self._extract_worm_crop(worm)
            
            # Run secondary classification
            result = self.secondary_classifier.predict([crop])
            
            # Update worm status
            worm.secondary_confidence = result['probabilities'][0]
            worm.secondary_classification = "worm" if result['predictions'][0] == 1 else "false_positive"
            
            # Debug: count predictions
            if result['predictions'][0] == 0:
                predictions_0 += 1
            else:
                predictions_1 += 1
            
            # Apply filtering logic
            if classification_mode == "filter":
                if result['predictions'][0] == 0:  # False positive
                    worm.classification_status = "rejected"
                else:
                    worm.classification_status = "confirmed"
            else:  # mark_only mode
                worm.classification_status = "pending"  # Keep all, just mark
        
        # Calculate statistics based on confirmed worms
        confirmed_worms = [w for w in self.WT.worms if w.classification_status == "confirmed"]
        rejected_worms = [w for w in self.WT.worms if w.classification_status == "rejected"]
        pending_worms = [w for w in self.WT.worms if w.classification_status == "pending"]
        
        self.N_total_confirmed = len(confirmed_worms)
        self.N_total_rejected = len(rejected_worms)
        self.N_total_original = len(self.WT.worms)
        
        print(f"   Predictions: {predictions_1} worms, {predictions_0} false_positives")
        print(f"   Status: {self.N_total_confirmed} confirmed, {self.N_total_rejected} rejected, {len(pending_worms)} pending")
        print(f"   Mode: {classification_mode}")

    def _extract_worm_crop(self, worm):
        """Extract crop around worm for secondary classification"""
        # Get original well crop
        well_crop = self.well.cropWell(self.original_image)
        
        # Extract window around worm centroid
        if self.secondary_classifier_window_size is None:
            raise ValueError("secondary_classifier_window_size must be set before calling _extract_worm_crop")
        window_size = self.secondary_classifier_window_size
        half_window = window_size // 2
        
        cx, cy = int(worm.centroid[0]), int(worm.centroid[1])
        
        # Ensure crop fits within well bounds
        x1 = max(0, cx - half_window)
        y1 = max(0, cy - half_window)
        x2 = min(well_crop.shape[1], cx + half_window)
        y2 = min(well_crop.shape[0], cy + half_window)
        
        # Extract and pad if necessary
        crop = well_crop[y1:y2, x1:x2]
        
        # Convert to grayscale if it's a color image
        if len(crop.shape) == 3:  # Color image
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Pad to required size if needed
        if crop.shape != (window_size, window_size):
            # Pad with zeros or edge values
            padded_crop = np.zeros((window_size, window_size), dtype=crop.dtype)
            h, w = crop.shape
            y_offset = (window_size - h) // 2
            x_offset = (window_size - w) // 2
            padded_crop[y_offset:y_offset+h, x_offset:x_offset+w] = crop
            crop = padded_crop
        
        return crop



class MultiWellAnalysis:

    skip_menus = False

    def __init__(self, fjson = "AnalyzeClustering.json", skip_menus = False): 

        MultiWellAnalysis.skip_menus = skip_menus

        # Load analysis parameters
        params = loadJSON(fjson)

        # Confirm the dark field and fluorescence paths
        if not MultiWellAnalysis.skip_menus:
            forceFileDialogsToFront()
            params["plate"]["paths"]["dark field"] = filedialog.askdirectory(title = "Select DARK FIELD path",initialdir=params["plate"]["paths"]["dark field"])
            if params["plate"]["analyze fluorescence frames"]:
                params["plate"]["paths"]["fluo"] = filedialog.askdirectory(title="Select FLUORESCENCE path",initialdir=params["plate"]["paths"]["fluo"])
            else:
                params["plate"]["paths"]["fluo"] = copy.deepcopy(params["plate"]["paths"]["dark field"] )
            saveJSON(fjson,params)

        # Set variables
        self.plate_params = params["plate"]
        self.seg_params = params["worm"]
        self.plot_params = params["plots"]
        self.all_params = params
        self.path_key = self.plate_params["main path key"]
        
        # Secondary classifier configuration
        self.secondary_classifier_params = params.get("secondary_classifier", {})
        self.secondary_classifier = None







    def analyzeImageSeries(self, path_key = "fluo"):

        # Skip analysis if already done (check with user first)
        if self.checkIfAlreadyDone():
            if not MultiWellAnalysis.skip_menus:
                if not messagebox.askyesno("Repeat analysis?",
                    "Data in:\n\n%s\n\nhas already been analyzed. Do you want to repeat the analysis from scratch?\n\nIf you want to quickly regenerate plots instead, click \"NO\"." %
                    self.plate_params["paths"]["fluo"]):
                    return
            else:
                # In automated mode, always proceed with analysis (overwrite existing)
                print("⚡ Automated mode: Overwriting existing analysis results")
                pass

        # Load the worm model (shared use among all wells)
        print('Loading YOLOv8 segmentation model...')
        self.seg_params["use cuda"] = torch.cuda.is_available()
        self.model = loadYOLO8(self.seg_params["model path"], 2, self.seg_params["use cuda"])        

        # Find the main experiment images, e.g. fluo
        filenames = os.listdir(self.plate_params["paths"][path_key])
        filenames = [i for i in filenames if "png" in i]

        # Create a list of plate objects, one per image. In case of series of a single plate, (set segment wells in first frame only to true)
        # then these will all be references to self.Plates[0]
        self.Plates: list[Plate] = []

        # Track overall analysis start time
        analysis_start_time = time.time()
        
        # Track errors for comprehensive reporting
        processing_errors = []

        for i,fname in enumerate(filenames): 
            try:
                # First, segment the plate and check with user that it is correct
                print('Segmenting wells...')

                # Cases where all plate objects are the same and refer to the first one... just ref it
                if self.plate_params["segment wells in first frame only"] and i > 0:
                    self.Plates.append(self.Plates[0])

                # First image OR doing a separate plate object for every image
                else:
                    
                    # Create the plate object and show preview
                    self.Plates.append(Plate(self.plate_params,use_file_idx=i))
                    _,self.wells_image = self.Plates[i].drawWells(wait_ms=33)
                    cv2.destroyAllWindows()            

                # Objects setup for i==0
                if i == 0:
                    # Preallocate overall experiment on i==0
                    total_jdata = {}
                    t = np.ones((len(filenames),))*np.nan
                    all_timepoint_data = {}
                    for param_type in self.all_params["data output"]["pages options"]:
                        if "Lawn dist indiv" in param_type:
                            # Lawn dist indiv parameter needs to be a list.
                            all_timepoint_data[param_type] = []
                        elif param_type in ["worm X positions", "worm Y positions"]:
                            # Worm X/Y positions stored as comma-separated numeric values per well per timepoint
                            all_timepoint_data[param_type] = np.full((len(filenames),len(self.Plates[i].wells)), "", dtype=object)
                        else:
                            # All other parameters are arrays where row is timepoint and column is well number
                            all_timepoint_data[param_type] = np.ones((len(filenames),len(self.Plates[i].wells)))*np.nan
                    

                    
                # Second, generate a WellAnalysis / worm tracker for each well to hold the worm info in each frame
                # note that we are only creating 6 for the 6 wells. 
                print('Initializing per-well analyses...')
                self.well_analyses:list[WellAnalysis] = []
                for well in self.Plates[i].wells:
                    self.well_analyses.append(WellAnalysis(well,self.seg_params))


                # Store timepoint
                t0 = time.time()
                t[i] = self.fname2time(fname)

                # Load image
                #print('frame=',i)
                img = cv2.imread(os.path.join(self.plate_params["paths"]["fluo"],filenames[i]),cv2.IMREAD_COLOR)

                # Analyze wells individually on this frame, assuming up to 200 worms per well
                total_jdata[fname] = {"t": self.fname2time(fname)}
                for j,well_analysis in enumerate(self.well_analyses):
                    well_analysis.segmentWormsInWell(img,self.model, show_debug = self.seg_params["show worm segmentation images"])

                    # store this well's data in arrays for analysis # TODO this belongs in WellAnalysis to output a single value for the well
                    # for k,worm in enumerate(well_analysis.WT.worms):
                    #     lawn_dists[j,k] = worm.dist_to_lawn_ctr

                    # Store this well's worm and other segmentations in the large metadata dict
                    total_jdata[fname]["Well " + str(j+1)] = {}
                    total_jdata[fname]["Well " + str(j+1)]["N worms"] = well_analysis.N_total
                    D = well_analysis.toDict()
                    total_jdata[fname]["Well " + str(j+1)]["worms"] = D["worms"]
                    total_jdata[fname]["Well " + str(j+1)]["lawn circle"] = D["lawn circle"]
                    total_jdata[fname]["Well " + str(j+1)]["well circle"] = D["well circle"]
                    total_jdata[fname]["Well " + str(j+1)]["local lawn circle"] = D["local lawn circle"]
                    total_jdata[fname]["Well " + str(j+1)]["local well circle"] = D["local well circle"]
                    total_jdata[fname]["Well " + str(j+1)]["mean dist to N nearest neighbors"] = D["mean dist to N nearest neighbors"]
                    #total_jdata[fname]["Well " + str(j+1)]["fraction clustering"] = D[""]

                    pass

                    # Store data for this timepoint which can be easily output
                    # TODO: add here for any new parameters to add
                    # all_timepoint_data["Lawn dist"][i,j] = lawn_dists_avg #TODO
                    all_timepoint_data["N worms"][i,j] = len(well_analysis.WT.worms)
                    
                    # Handle NaN values for neighbor distance
                    if np.isnan(well_analysis.mean_distance_to_N_neighbors):
                        all_timepoint_data["Neighbor dist"][i,j] = np.nan
                    else:
                        all_timepoint_data["Neighbor dist"][i,j] = well_analysis.mean_distance_to_N_neighbors
                        
                    all_timepoint_data["fraction touching"][i,j] = well_analysis.frac_touching
                    all_timepoint_data["fraction touching (from alone)"][i,j] = (self.plate_params["Ideal worms per well"] - well_analysis.N_alone) / self.plate_params["Ideal worms per well"]
                    all_timepoint_data["N touching"][i,j] = well_analysis.N_touching
                    all_timepoint_data["N alone"][i,j] = well_analysis.N_alone
                    

                    
                    # Store worm X and Y positions separately if these analysis types are enabled
                    if "worm X positions" in all_timepoint_data:
                        x_positions = []
                        for worm in well_analysis.WT.worms:
                            x, y = worm.centroid
                            x_positions.append(f"{x:.1f}")
                        all_timepoint_data["worm X positions"][i,j] = ",".join(x_positions)
                    
                    if "worm Y positions" in all_timepoint_data:
                        y_positions = []
                        for worm in well_analysis.WT.worms:
                            x, y = worm.centroid
                            y_positions.append(f"{y:.1f}")
                        all_timepoint_data["worm Y positions"][i,j] = ",".join(y_positions)
                
                # Store indvidual data
                # all_timepoint_data["Lawn dist indiv"].append(copy.deepcopy(lawn_dists)) #TODO refactor to within the loop

                # ============= DETAILED PROGRESS UPDATE =============
                current_time = time.time()
                dt = current_time - t0  # Time for current iteration
                elapsed_total = current_time - analysis_start_time  # Actual total elapsed time
                
                # Calculate remaining time based on actual progress
                frames_completed = i + 1
                avg_time_per_frame = elapsed_total / frames_completed
                frames_remaining = len(filenames) - frames_completed
                t_remain = avg_time_per_frame * frames_remaining
                
                # Calculate detailed timing
                hours_elapsed = int(elapsed_total // 3600)
                minutes_elapsed = int((elapsed_total % 3600) // 60)
                seconds_elapsed = int(elapsed_total % 60)
                
                hours_remain = int(t_remain // 3600)
                minutes_remain = int((t_remain % 3600) // 60)
                seconds_remain = int(t_remain % 60)
                
                # Format timing strings
                if hours_elapsed > 0:
                    elapsed_str = f"{hours_elapsed}h {minutes_elapsed}m {seconds_elapsed}s"
                elif minutes_elapsed > 0:
                    elapsed_str = f"{minutes_elapsed}m {seconds_elapsed}s"
                else:
                    elapsed_str = f"{seconds_elapsed}s"
                    
                if hours_remain > 0:
                    remain_str = f"{hours_remain}h {minutes_remain}m {seconds_remain}s"
                elif minutes_remain > 0:
                    remain_str = f"{minutes_remain}m {seconds_remain}s"
                else:
                    remain_str = f"{seconds_remain}s"
                
                # Calculate percentage and rate
                progress_pct = (frames_completed / len(filenames)) * 100
                rate_per_min = 60.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0
                
                print("=" * 80)
                print(f"🎯 IMAGE ANALYSIS PROGRESS - FRAME {i+1}/{len(filenames)}")
                print("=" * 80)
                print(f"📊 PROGRESS:     {progress_pct:5.1f}% complete")
                print(f"⏱️  TIME ELAPSED: {elapsed_str}")
                print(f"⏳ TIME REMAINING: {remain_str}")
                print(f"🚀 PROCESSING RATE: {rate_per_min:.1f} frames/min ({avg_time_per_frame:.1f}s avg per frame)")
                print("=" * 80)
                pass

                # Image of well seg (only if enabled in config)
                fout = self.getOutNames()
                if self.plate_params.get("save well segmentation images", False):
                    cv2.imwrite(fout["well img"].replace(".png","_"+fname), self.wells_image)
                    
            except Exception as e:
                raise
                # Capture detailed error information
                error_info = {
                    'filename': fname,
                    'frame_index': i,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'full_traceback': traceback.format_exc(),
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                processing_errors.append(error_info)
                
                print("!" * 80)
                print(f"❌ ERROR PROCESSING FRAME {i+1}/{len(filenames)}: {fname}")
                print(f"Error: {type(e).__name__}: {str(e)}")
                print("Continuing with next frame...")
                print("!" * 80)
                
                # Initialize arrays with NaN values for this failed frame
                if 'all_timepoint_data' in locals():
                    for param_type in self.all_params["data output"]["pages options"]:
                        if "Lawn dist indiv" not in param_type and param_type in all_timepoint_data:
                            if hasattr(all_timepoint_data[param_type], 'shape') and len(all_timepoint_data[param_type].shape) > 1:
                                if param_type in ["worm X positions", "worm Y positions"]:
                                    # For worm X/Y positions, use empty string instead of NaN
                                    all_timepoint_data[param_type][i, :] = ""
                                else:
                                    all_timepoint_data[param_type][i, :] = np.nan
                
                # Continue to next iteration
                continue

        # ============= ERROR SUMMARY AND REPORTING =============
        if processing_errors:
            print("\n" + "!" * 80)
            print("⚠️  ANALYSIS COMPLETED WITH ERRORS")
            print("!" * 80)
            print(f"Total frames processed: {len(filenames)}")
            print(f"Successful frames: {len(filenames) - len(processing_errors)}")
            print(f"Failed frames: {len(processing_errors)}")
            print("!" * 80)
            
            # Print summary of errors
            print("ERROR SUMMARY:")
            for error in processing_errors:
                print(f"  - Frame {error['frame_index']+1}: {error['filename']} - {error['error_type']}: {error['error_message'][:100]}...")
            
            # Save detailed error report to file
            try:
                fout = self.getOutNames()
                error_report_path = os.path.join(fout["pout"], "error_report.txt")
                
                with open(error_report_path, 'w') as f:
                    f.write("=" * 80 + "\n")
                    f.write("IMAGE ANALYSIS ERROR REPORT\n")
                    f.write("=" * 80 + "\n")
                    f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total frames: {len(filenames)}\n")
                    f.write(f"Successful: {len(filenames) - len(processing_errors)}\n")
                    f.write(f"Failed: {len(processing_errors)}\n\n")
                    
                    for i, error in enumerate(processing_errors, 1):
                        f.write(f"ERROR #{i}\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Filename: {error['filename']}\n")
                        f.write(f"Frame Index: {error['frame_index']}\n")
                        f.write(f"Timestamp: {error['timestamp']}\n")
                        f.write(f"Error Type: {error['error_type']}\n")
                        f.write(f"Error Message: {error['error_message']}\n")
                        f.write(f"Full Traceback:\n{error['full_traceback']}\n")
                        f.write("=" * 80 + "\n\n")
                
                print(f"📄 Detailed error report saved to: {error_report_path}")
                
            except Exception as report_error:
                print(f"⚠️  Could not save error report: {report_error}")
            
            print("!" * 80)
        else:
            print("\n" + "✅" * 20)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY - NO ERRORS!")
            print("✅" * 20)

        # Assemble indiv data
        # df_D_lawn_filt_indiv = pd.DataFrame({"time (hr)":t/3600})    
        # for i,arr in enumerate(all_timepoint_data["Lawn dist indiv"]):
        #     # arr: [N_wells, N_worms_max (e.g. 20)]
        #     for j,row in enumerate(arr):
        #         for k in range(5):
        #             df_D_lawn_filt_indiv.at[i,"Well " + str(j+1) + "."+str(k+1)]= row[k]
        #         pass


        # Determine the x-axis from the config
        data_frames = {}
        data_output_config = self.all_params["data output"]        
        x_axis = data_output_config["x axis"]
        if x_axis == "time": 
            x_axis_data = t/3600 
        else:
            x_axis_data = filenames

        # Prepare data frames based on the selected pages            
        for analysis_type in data_output_config["pages"]:
            data_frames[analysis_type] = pd.DataFrame({x_axis: x_axis_data})
            arr = all_timepoint_data[analysis_type]
            
            if analysis_type in ["worm X positions", "worm Y positions"]:
                # Special handling for worm X/Y positions (comma-separated numeric strings)
                for well_idx in range(arr.shape[1]):
                    well_num = well_idx + 1
                    data_frames[analysis_type]["Well " + str(well_num)] = arr[:,well_idx].astype(str)
            else:
                # Standard numeric data handling
                for well_idx in range(arr.shape[1]):
                    well_num = well_idx + 1
                    data_frames[analysis_type]["Well " + str(well_num)] = arr[:,well_idx]

        # ... You can add more conditions for other pages here

        # Save data to Excel
        while(True):
            try: 
                with pd.ExcelWriter(fout["xlsx raw"], engine='xlsxwriter') as writer:
                    for page, df in data_frames.items():
                        df.to_excel(writer, sheet_name=page, index=False)
                    break
            except PermissionError as err:
                retry = messagebox.askretrycancel("Please close Excel",f"Could not save data to file\n\n{fout['xlsx raw']}\n\nBecause excel is open. Please close and try again.\n\nIf you cancel your entire analysis will not be saved!")
                if not retry: break

        # JSON file with analysis parameters
        analysis_metadata = copy.deepcopy(self.all_params)
        to_write = json.dumps(analysis_metadata,indent=4)
        with open(fout["json"],"w") as outfile:
            outfile.write(to_write)

        # JSON file with all frames' metadata
        saveJSON(fout["total json"],total_jdata,indent=None)

        pass

    def fname2time(self,fname):
        # WWRecorder format
        if "t=" in fname:
            i0 = fname.find("t=")+2
            i1 = fname.find(" s")
            t_str = fname[i0:i1]
            t = float(t_str)
            return t
        else:
            return 0

    def checkIfAlreadyDone(self):
        pout = os.path.join(self.plate_params["paths"][self.path_key],"_Analysis")
        fout = os.path.join(pout,"analysis_metadata.json")
        return os.path.exists(fout) and not MultiWellAnalysis.skip_menus

    def getOutNames(self):
        pout = os.path.join(self.plate_params["paths"][self.path_key],"_Analysis")
        if not os.path.exists(pout): os.makedirs(pout)
        _,expt_head = os.path.split(self.plate_params["paths"][self.path_key])       
        outnames ={} 
        outnames["pout"] = pout
        outnames["xlsx raw"] = os.path.join(pout,expt_head+"_raw.xlsx")
        outnames["well img"] = os.path.join(pout,expt_head + " well segmentation.png")
        outnames["json"] = os.path.join(pout,"analysis_metadata.json")
        outnames["total json"] = os.path.join(pout,"analysis_dump.json")
        outnames["xlsx plotted"] = os.path.join(pout,"analysis_metadata.json")
        return outnames
    
    def _load_secondary_classifier(self):
        """Load the secondary worm classifier"""
        try:
            from WormClassifier import WormClassifier, get_default_config
            
            window_size = self.secondary_classifier_params["secondary_classifier_window_size"]
            config = get_default_config(window_size=window_size)
            config.update({
                'confidence_threshold': self.secondary_classifier_params["secondary_confidence_threshold"]
            })
            
            self.secondary_classifier = WormClassifier(config)
            self.secondary_classifier.load_model(self.secondary_classifier_params["secondary_classifier_path"])
            print(f"✅ Secondary classifier loaded: {self.secondary_classifier_params['secondary_classifier_path']}")
            
        except Exception as e:
            print(f"❌ Failed to load secondary classifier: {e}")
            self.secondary_classifier = None



    def generatePlots(self):

        # Find raw data excel file
        fout = self.getOutNames()
        if not os.path.exists(fout["xlsx raw"]): raise Exception("Generate plots could not find XLSX RAW in path:\n%s" % (fout['xlsx raw']))

        # Setup excel out file for grouped data
        writer = pd.ExcelWriter(fout["xlsx raw"].replace("raw","grouped"), engine='xlsxwriter')


        for fignum,sheet_name in enumerate(self.plot_params["sheets"]):
            df = pd.read_excel(fout['xlsx raw'],sheet_name=sheet_name)
            df_filt = copy.deepcopy(df) # output arr

            # Set time axis
            t = df['time (hr)']
            if self.plot_params['time unit'].lower() == "hr": pass
            elif self.plot_params['time unit'].lower() == "min": t*=60
            elif self.plot_params['time unit'].lower() == "sec": t*= 3600
            else: raise Exception('Invalid time unit [%s] - must be one of: %s' %(self.plot_params["time unit"],["hr", "min", "sec"]))
            t_key = 'time ' + self.plot_params['time unit'].lower() 

            fig = plt.figure(fignum, figsize = (12,7))

            # Select whether to plot by group or individual
            if "N worms" in sheet_name:
                wellnums = range(1,self.plate_params["N wells"]+1)
                groups = {}
                labels = []
                for i in range(self.plate_params["N wells"]): 
                    groups[str(i+1)] = [i+1]
                    labels.append(str(i+1))
            else:
                groups = self.plot_params["groups"]
                labels = list(self.plot_params["groups"])



            # Group data
            df_grouped = pd.DataFrame({"time (hr)":t/3600})
            for group_name in groups:
                group_list = groups[group_name]
                group_data = np.zeros((df.shape[0],len(group_list)))

                for c,well_num in enumerate(group_list):
                    key = "Well " + str(well_num)
                    well_data = scipy.signal.medfilt(df[key],kernel_size = self.plot_params["median kernel size"])
                    group_data[:,c] = well_data
                    

                # Plot average
                group_mean = np.mean(group_data,axis=1)
                plt.plot(t, group_mean,linewidth=2)

                df_grouped[group_name] = group_mean

            # Save plot results to xlsx
            if not "N worms" in sheet_name:
                df_grouped.to_excel(writer,sheet_name="Lawn dist",index=False)

            plt.legend(labels)
            plt.xlabel(t_key)
            plt.ylabel(sheet_name)
            plt.title("Mean " + sheet_name)
            plt.savefig(os.path.join(fout["pout"],sheet_name))

        writer.close()
        plt.show()

    def apply_secondary_classification_to_existing(self):
        """Apply secondary classifier to previously analyzed dataset"""
        if not self.checkIfAlreadyDone():
            print("❌ No existing analysis found. Run analyzeImageSeries() first.")
            return
        
        # Load secondary classifier
        if self.secondary_classifier is None:
            if self.seg_params.get("use_secondary_classifier", False):
                self._load_secondary_classifier()
            else:
                print("❌ Secondary classifier not enabled in configuration.")
                return
        
        if self.secondary_classifier is None:
            print("❌ Failed to load secondary classifier.")
            return
        
        print("🔄 Applying secondary classification to existing dataset...")
        
        # Load existing results
        fout = self.getOutNames()
        with open(fout["total json"], 'r') as f:
            existing_data = json.load(f)
        
        # Process each frame
        total_frames = len([k for k in existing_data.keys() if k != "t"])
        processed_frames = 0
        
        for filename, frame_data in existing_data.items():
            if filename == "t":  # Skip time metadata
                continue
                
            processed_frames += 1
            print(f"Processing frame {processed_frames}/{total_frames}: {filename}")
            
            # Load original image
            img_path = os.path.join(self.plate_params["paths"]["fluo"], filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"⚠️  Could not load image: {img_path}")
                continue
            
            # Recreate plate and well analysis for this frame
            if self.plate_params["segment wells in first frame only"]:
                plate = self.Plates[0] if hasattr(self, 'Plates') and len(self.Plates) > 0 else Plate(self.plate_params, use_file_idx=0)
            else:
                # Find the index for this filename
                filenames = [f for f in os.listdir(self.plate_params["paths"]["fluo"]) if f.endswith('.png')]
                try:
                    file_idx = filenames.index(filename)
                    plate = Plate(self.plate_params, use_file_idx=file_idx)
                except ValueError:
                    print(f"⚠️  Could not find filename in directory: {filename}")
                    continue
            
            # Create well analyses
            well_analyses = []
            for well in plate.wells:
                well_analysis = WellAnalysis(well, self.seg_params)
                well_analysis.secondary_classifier = self.secondary_classifier
                well_analysis.secondary_classifier_window_size = self.seg_params["secondary_classifier_window_size"]
                well_analysis.secondary_classification_mode = self.seg_params["secondary_classification_mode"]
                well_analysis.original_image = img.copy()
                well_analyses.append(well_analysis)
            
            # Apply secondary classification to each well
            for j, well_analysis in enumerate(well_analyses):
                well_key = f"Well {j+1}"
                if well_key in frame_data:
                    # Recreate worms from existing data
                    well_analysis.WT.worms = []
                    if "worms" in frame_data[well_key]:
                        # Recreate worm objects from saved data
                        from Worm import Worm
                        for worm_data in frame_data[well_key]["worms"]:
                            worm = Worm()  # Create empty worm
                            worm.fromDict(worm_data)  # Populate from saved data
                            well_analysis.WT.worms.append(worm)
                    
                    # Apply secondary classification
                    well_analysis._apply_secondary_classification()
                    
                    # Update frame data with new classifications
                    if hasattr(well_analysis, 'N_total_confirmed'):
                        frame_data[well_key]["N worms confirmed"] = well_analysis.N_total_confirmed
                        frame_data[well_key]["N worms rejected"] = well_analysis.N_total_rejected
                        frame_data[well_key]["N worms original"] = well_analysis.N_total_original
        
        # Save updated results
        with open(fout["total json"], 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print("✅ Secondary classification applied to existing dataset")
        print(f"Updated results saved to: {fout['total json']}")

    def secondaryObjectVerification(self):
        """Apply secondary classifier verification to all detected worms"""
        
        # Check if analysis has been done
        if not self.checkIfAlreadyDone():
            print("❌ No analysis found. Run analyzeImageSeries() first.")
            return
        
        # Load secondary classifier
        print("🔄 Loading secondary classifier...")
        self._load_secondary_classifier()
        
        if self.secondary_classifier is None:
            print("❌ Failed to load secondary classifier. Check configuration.")
            return
        
        print("🔄 Starting secondary object verification...")
        
        # Load existing results
        fout = self.getOutNames()
        with open(fout["total json"], 'r') as f:
            existing_data = json.load(f)
        
        # Find all image files
        filenames = [f for f in os.listdir(self.plate_params["paths"]["fluo"]) if f.endswith('.png')]
        total_frames = len(filenames)
        
        # Initialize data arrays for secondary classification results
        all_timepoint_data = {}
        all_timepoint_data["N worms confirmed"] = np.ones((total_frames, self.plate_params["N wells"])) * np.nan
        all_timepoint_data["N worms rejected"] = np.ones((total_frames, self.plate_params["N wells"])) * np.nan
        all_timepoint_data["Secondary confidence avg"] = np.ones((total_frames, self.plate_params["N wells"])) * np.nan
        # Add worm position tracking for confirmed worms
        all_timepoint_data["worm X positions confirmed"] = np.full((total_frames, self.plate_params["N wells"]), "", dtype=object)
        all_timepoint_data["worm Y positions confirmed"] = np.full((total_frames, self.plate_params["N wells"]), "", dtype=object)
        # Add touching/alone counts for confirmed worms
        all_timepoint_data["N touching confirmed"] = np.ones((total_frames, self.plate_params["N wells"])) * np.nan
        all_timepoint_data["N alone confirmed"] = np.ones((total_frames, self.plate_params["N wells"])) * np.nan
        
        # Process each frame
        for i, filename in enumerate(filenames):
            print(f"Processing frame {i+1}/{total_frames}: {filename}")
            
            # Load original image
            img_path = os.path.join(self.plate_params["paths"]["fluo"], filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is None:
                print(f"⚠️  Could not load image: {img_path}")
                continue
            
            # Recreate plate and well analysis for this frame
            if self.plate_params["segment wells in first frame only"]:
                plate = self.Plates[0] if hasattr(self, 'Plates') and len(self.Plates) > 0 else Plate(self.plate_params, use_file_idx=0)
            else:
                plate = Plate(self.plate_params, use_file_idx=i)
            
            # Create well analyses
            well_analyses = []
            for well in plate.wells:
                well_analysis = WellAnalysis(well, self.seg_params)
                well_analysis.secondary_classifier = self.secondary_classifier
                well_analysis.secondary_classifier_window_size = self.secondary_classifier_params["secondary_classifier_window_size"]
                well_analysis.secondary_classification_mode = self.secondary_classifier_params["secondary_classification_mode"]
                well_analysis.original_image = img.copy()
                well_analyses.append(well_analysis)
            
            # Apply secondary classification to each well
            for j, well_analysis in enumerate(well_analyses):
                well_key = f"Well {j+1}"
                
                # Recreate worms from existing data
                if filename in existing_data and well_key in existing_data[filename]:
                    well_analysis.WT.worms = []
                    if "worms" in existing_data[filename][well_key]:
                        from Worm import Worm
                        for worm_data in existing_data[filename][well_key]["worms"]:
                            worm = Worm()  # Create empty worm
                            worm.fromDict(worm_data)  # Populate from saved data
                            well_analysis.WT.worms.append(worm)
                
                # Apply secondary classification
                well_analysis._apply_secondary_classification()
                
                # Store results
                all_timepoint_data["N worms confirmed"][i,j] = getattr(well_analysis, 'N_total_confirmed', len(well_analysis.WT.worms))
                all_timepoint_data["N worms rejected"][i,j] = getattr(well_analysis, 'N_total_rejected', 0)
                
                # Calculate average secondary confidence for confirmed worms
                confirmed_worms = [w for w in well_analysis.WT.worms if hasattr(w, 'classification_status') and w.classification_status == "confirmed"]
                if confirmed_worms:
                    avg_confidence = np.mean([w.secondary_confidence for w in confirmed_worms])
                    all_timepoint_data["Secondary confidence avg"][i,j] = avg_confidence
                
                # Store confirmed worm X and Y positions (similar to original format)
                x_positions_confirmed = []
                y_positions_confirmed = []
                for worm in confirmed_worms:
                    x, y = worm.centroid
                    x_positions_confirmed.append(f"{x:.1f}")
                    y_positions_confirmed.append(f"{y:.1f}")
                all_timepoint_data["worm X positions confirmed"][i,j] = ",".join(x_positions_confirmed)
                all_timepoint_data["worm Y positions confirmed"][i,j] = ",".join(y_positions_confirmed)
                
                # Update existing data with new classifications
                if filename in existing_data and well_key in existing_data[filename]:
                    existing_data[filename][well_key]["N worms confirmed"] = all_timepoint_data["N worms confirmed"][i,j]
                    existing_data[filename][well_key]["N worms rejected"] = all_timepoint_data["N worms rejected"][i,j]
                    existing_data[filename][well_key]["N worms original"] = len(well_analysis.WT.worms)
                    
                    # Update individual worm objects with classification data
                    if "worms" in existing_data[filename][well_key]:
                        for k, worm_data in enumerate(existing_data[filename][well_key]["worms"]):
                            if k < len(well_analysis.WT.worms):
                                worm = well_analysis.WT.worms[k]
                                # Update worm data with classification information
                                worm_data["classification_status"] = worm.classification_status
                                worm_data["primary_confidence"] = worm.primary_confidence
                                worm_data["secondary_confidence"] = worm.secondary_confidence
                                worm_data["secondary_classification"] = worm.secondary_classification
        
        # Save updated results
        with open(fout["total json"], 'w') as f:
            json.dump(existing_data, f, indent=None)
        
        # NOTE: We deliberately do NOT modify the original Excel file to preserve the original analysis results
        print("📋 Original Excel file left unchanged to preserve original analysis results")
        
        # Create Excel file with secondary classification results
        # Use the same x-axis as the main data file
        data_output_config = self.all_params["data output"]        
        x_axis = data_output_config["x axis"]
        if x_axis == "time": 
            x_axis_data = [self.fname2time(f) for f in filenames]
        else:
            x_axis_data = filenames
        
        # Prepare data frames for secondary classification results
        data_frames = {}
        for analysis_type in ["N worms confirmed", "N worms rejected", "Secondary confidence avg", "worm X positions confirmed", "worm Y positions confirmed", "N touching confirmed", "N alone confirmed"]:
            data_frames[analysis_type] = pd.DataFrame({x_axis: x_axis_data})
            arr = all_timepoint_data[analysis_type]
            
            if analysis_type in ["worm X positions confirmed", "worm Y positions confirmed"]:
                # Special handling for worm position data (comma-separated numeric strings)
                for well_idx in range(arr.shape[1]):
                    well_num = well_idx + 1
                    data_frames[analysis_type]["Well " + str(well_num)] = arr[:,well_idx].astype(str)
            else:
                # Standard numeric data handling
                for well_idx in range(arr.shape[1]):
                    well_num = well_idx + 1
                    data_frames[analysis_type]["Well " + str(well_num)] = arr[:,well_idx]
        
        # Save secondary classification results to Excel
        secondary_results_file = fout["xlsx raw"].replace("_raw.xlsx", "_secondary_verification.xlsx")
        with pd.ExcelWriter(secondary_results_file, engine='xlsxwriter') as writer:
            for page, df in data_frames.items():
                df.to_excel(writer, sheet_name=page, index=False)
        
        print("✅ Secondary object verification completed!")
        print(f"Updated results saved to: {fout['total json']}")
        print(f"Secondary verification results saved to: {secondary_results_file}")
        print(f"   📍 Includes confirmed worm X and Y positions for each timepoint and well")
        
        # Print summary statistics
        total_confirmed = np.nansum(all_timepoint_data["N worms confirmed"])
        total_rejected = np.nansum(all_timepoint_data["N worms rejected"])
        total_original = total_confirmed + total_rejected
        
        total_confirmed_touching = np.nansum(all_timepoint_data["N touching confirmed"])
        total_confirmed_alone = np.nansum(all_timepoint_data["N alone confirmed"])
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total worms detected: {total_original}")
        print(f"   Worms confirmed: {total_confirmed}")
        print(f"   Worms rejected: {total_rejected}")
        print(f"   Rejection rate: {total_rejected/total_original*100:.1f}%" if total_original > 0 else "   Rejection rate: N/A")
        print(f"   Confirmed touching: {total_confirmed_touching}")
        print(f"   Confirmed alone: {total_confirmed_alone}")
        if total_confirmed > 0:
            print(f"   Confirmed touching rate: {total_confirmed_touching/total_confirmed*100:.1f}%")



    # Static function to draw all worms from a dict saved to disk
    def drawFromDict(img_color, D:dict, local = False, rad = None):

        # Dynamically set thickness and geometry params
        thick = 1#+int(img_color.shape[1]/2000) # e.g. 2 if size is >2000 or 3 if >3000 etc
        if rad is None:
            rad = 4+int(img_color.shape[1]/1000)


        for key in D:
            if not "Well" in key: continue
            worms= D[key]['worms']

            dx,dy,r = D[key]['well circle']
            for worm in worms:

                # extract contours
                contours = np.array(worm['contours'])
                centroid = copy.deepcopy(worm['centroid'])

                # Calculate actual crop offset (same logic as cropWell method)
                # This accounts for clipping when wells are at image boundaries
                actual_x_offset = max(0, dx - r)
                actual_y_offset = max(0, dy - r)
                
                # Shift to well positions using actual offsets
                contours[:,0,0] += actual_x_offset
                contours[:,0,1] += actual_y_offset
                centroid[0] = centroid[0] + actual_x_offset
                centroid[1] = centroid[1] + actual_y_offset

                # Determine color based on secondary classification status
                if 'classification_status' in worm:
                    if worm['classification_status'] == "rejected":
                        contour_color = (0, 0, 255)  # Red for rejected worms
                    elif worm['classification_status'] == "confirmed":
                        contour_color = (0, 255, 0)  # Green for confirmed worms
                    else:
                        contour_color = (0, 255, 0)  # Default green for pending/unknown
                else:
                    contour_color = (0, 255, 0)  # Default green for worms without secondary classification

                # Draw
                img_color = cv2.drawContours(img_color,[contours],0,contour_color,thick)
                img_color = cv2.circle(img_color,tuple(centroid),rad,(255,0,0),thickness=-1)

            pass


        return img_color


# QUick test of plotting
if __name__ == "__main__":
    MWA = MultiWellAnalysis()
    MWA.analyzeImageSeries()
    MWA.generatePlots()
