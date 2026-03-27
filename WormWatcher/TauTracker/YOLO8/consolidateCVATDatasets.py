"""
Consolidate Multiple CVAT Datasets
Consolidates multiple CVAT XML annotation datasets into a single folder of images
and a single VGG JSON annotation file.

Each input folder should contain:
- One XML file with CVAT bounding box annotations
- Images either in the same folder or in a subfolder (e.g., "images" or first subfolder)

The output will be:
- A single folder containing all images (with collision handling)
- A single VGG JSON file with all annotations consolidated

Usage:
    python consolidateCVATDatasets.py [config.json]
    
    If no config file is provided, a file dialog will open to select one.
"""

import json
import os
import sys
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Image extensions to look for
IMAGE_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.TIF', '.TIFF', '.PNG', '.JPG', '.JPEG']

# Global configuration variables (set from JSON config file)
TILE_SIZE = 512
TILE_OVERLAP = 0.1
MAX_IMAGE_SIZE_FOR_TILING = 512
CLASS_NAME = "worm"

# Check for PIL/Pillow and OpenCV for image conversion
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False


def find_cvat_xml_file(folder_path: str) -> Optional[str]:
    """
    Find the CVAT XML file in a folder.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        Path to the XML file, or None if not found
    """
    xml_files = list(Path(folder_path).glob("*.xml"))
    if len(xml_files) == 0:
        return None
    elif len(xml_files) == 1:
        return str(xml_files[0])
    else:
        # Multiple XML files - prefer one with "annotation" or "cvat" in name
        for xml_file in xml_files:
            name_lower = xml_file.name.lower()
            if "annotation" in name_lower or "cvat" in name_lower:
                return str(xml_file)
        # Return the first one if no match
        return str(xml_files[0])


def find_vgg_json_file(folder_path: str) -> Optional[str]:
    """
    Find the VGG JSON file in a folder.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        Path to the JSON file, or None if not found
    """
    json_files = list(Path(folder_path).glob("*.json"))
    if len(json_files) == 0:
        return None
    elif len(json_files) == 1:
        return str(json_files[0])
    else:
        # Multiple JSON files - prefer one with "vgg", "via", or "annotation" in name
        for json_file in json_files:
            name_lower = json_file.name.lower()
            if "vgg" in name_lower or "via" in name_lower or "annotation" in name_lower:
                return str(json_file)
        # Return the first one if no match
        return str(json_files[0])


def polygon_to_bounding_box(all_points_x: List[float], all_points_y: List[float]) -> Tuple[float, float, float, float]:
    """
    Convert polygon coordinates to bounding box (x, y, width, height).
    
    Args:
        all_points_x: List of x coordinates
        all_points_y: List of y coordinates
        
    Returns:
        Tuple of (x, y, width, height) for bounding box
    """
    if len(all_points_x) == 0 or len(all_points_y) == 0:
        return (0, 0, 0, 0)
    
    min_x = min(all_points_x)
    max_x = max(all_points_x)
    min_y = min(all_points_y)
    max_y = max(all_points_y)
    
    x = min_x
    y = min_y
    width = max_x - min_x
    height = max_y - min_y
    
    return (x, y, width, height)


def parse_vgg_region_to_bounding_box(region: Dict, class_name: str) -> Optional[Dict]:
    """
    Parse a VGG region and convert to bounding box format (rect).
    Handles polygons, rectangles, and circles.
    
    Args:
        region: VGG region dictionary
        class_name: Name of the class
        
    Returns:
        Dictionary with VGG rect format (x, y, width, height) or None if invalid
    """
    shape_attr = region.get('shape_attributes') or region
    region_attr = region.get('region_attributes', {})
    
    # Get class name
    if isinstance(region_attr, dict):
        label = region_attr.get('name', region_attr.get('class', class_name))
    else:
        label = class_name
    
    shape_name = shape_attr.get('name', '')
    
    # Handle polygon
    if shape_name == 'polygon' or ('all_points_x' in shape_attr and 'all_points_y' in shape_attr):
        all_points_x = shape_attr.get('all_points_x', [])
        all_points_y = shape_attr.get('all_points_y', [])
        
        if len(all_points_x) < 3 or len(all_points_y) < 3:
            return None
        
        # Convert polygon to bounding box
        x, y, width, height = polygon_to_bounding_box(all_points_x, all_points_y)
        
        if width <= 0 or height <= 0:
            return None
        
        return {
            "shape_attributes": {
                "name": "rect",
                "x": int(round(x)),
                "y": int(round(y)),
                "width": int(round(width)),
                "height": int(round(height))
            },
            "region_attributes": {
                "name": label
            }
        }
    
    # Handle rectangle (already in correct format)
    elif shape_name == 'rect' or ('x' in shape_attr and 'y' in shape_attr and 'width' in shape_attr and 'height' in shape_attr):
        x = shape_attr.get('x', 0)
        y = shape_attr.get('y', 0)
        width = shape_attr.get('width', 0)
        height = shape_attr.get('height', 0)
        
        if width <= 0 or height <= 0:
            return None
        
        return {
            "shape_attributes": {
                "name": "rect",
                "x": int(round(x)),
                "y": int(round(y)),
                "width": int(round(width)),
                "height": int(round(height))
            },
            "region_attributes": {
                "name": label
            }
        }
    
    # Handle circle - convert to bounding box
    elif shape_name == 'circle' or ('cx' in shape_attr and 'cy' in shape_attr and 'r' in shape_attr):
        cx = shape_attr.get('cx', 0)
        cy = shape_attr.get('cy', 0)
        r = shape_attr.get('r', 0)
        
        if r <= 0:
            return None
        
        x = cx - r
        y = cy - r
        width = 2 * r
        height = 2 * r
        
        return {
            "shape_attributes": {
                "name": "rect",
                "x": int(round(x)),
                "y": int(round(y)),
                "width": int(round(width)),
                "height": int(round(height))
            },
            "region_attributes": {
                "name": label
            }
        }
    
    return None


def find_images_directory(folder_path: str) -> str:
    """
    Find the images directory in a folder.
    Checks the folder itself, then looks for subfolders like "images" or first subfolder.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        Path to the images directory
    """
    # Check if images are directly in the folder
    folder_path_obj = Path(folder_path)
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(folder_path_obj.glob(f"*{ext}"))
    
    if len(image_files) > 0:
        return folder_path
    
    # Check for "images" subfolder
    images_subfolder = folder_path_obj / "images"
    if images_subfolder.exists() and images_subfolder.is_dir():
        return str(images_subfolder)
    
    # Check first subfolder
    subfolders = [f for f in folder_path_obj.iterdir() if f.is_dir()]
    if len(subfolders) > 0:
        return str(subfolders[0])
    
    # Return original folder if nothing found
    return folder_path


def handle_filename_collision(base_name: str, extension: str, used_names: set) -> str:
    """
    Handle filename collisions by appending a counter.
    
    Args:
        base_name: Base filename without extension
        extension: File extension (with dot)
        used_names: Set of already used filenames
        
    Returns:
        Unique filename
    """
    filename = base_name + extension
    if filename not in used_names:
        used_names.add(filename)
        return filename
    
    counter = 1
    while True:
        new_filename = f"{base_name}_{counter}{extension}"
        if new_filename not in used_names:
            used_names.add(new_filename)
            return new_filename
        counter += 1


def parse_cvat_box_to_vgg(box_elem: ET.Element, class_name: str) -> Optional[Dict]:
    """
    Parse a CVAT <box> element to VGG JSON format.
    
    CVAT boxes use xtl, ytl, xbr, ybr (top-left and bottom-right corners).
    VGG rectangles use x, y, width, height.
    
    Args:
        box_elem: XML Element for <box>
        class_name: Name of the class
        
    Returns:
        Dictionary with VGG region format or None if invalid
    """
    # Get label (class name)
    label = box_elem.get('label', class_name)
    
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
    
    # Convert CVAT format (xtl, ytl, xbr, ybr) to VGG format (x, y, width, height)
    x = xtl
    y = ytl
    width = xbr - xtl
    height = ybr - ytl
    
    # Create VGG region structure
    region = {
        "shape_attributes": {
            "name": "rect",
            "x": int(round(x)),
            "y": int(round(y)),
            "width": int(round(width)),
            "height": int(round(height))
        },
        "region_attributes": {
            "name": label
        }
    }
    
    return region


def get_output_filename_for_image(image_path: str) -> str:
    """
    Determine the output filename for an image (converting TIF to PNG extension).
    
    Args:
        image_path: Path to the source image
        
    Returns:
        Output filename (e.g., converts .tif to .png, keeps others as-is)
    """
    image_path_obj = Path(image_path)
    image_ext = image_path_obj.suffix.lower()
    base_name = image_path_obj.stem
    
    # If it's TIF/TIFF, convert extension to PNG
    if image_ext in ['.tif', '.tiff']:
        return base_name + '.png'
    
    # For other extensions, keep original name
    return image_path_obj.name


def validate_region_is_rect(region: Dict) -> bool:
    """
    Validate that a region is in rect format (not polygon or other format).
    
    Args:
        region: Region dictionary
        
    Returns:
        True if region is a rect, False otherwise
    """
    shape_attr = region.get('shape_attributes', {})
    shape_name = shape_attr.get('name', '')
    
    # Must be "rect"
    if shape_name != 'rect':
        return False
    
    # Must have x, y, width, height
    required_keys = ['x', 'y', 'width', 'height']
    if not all(key in shape_attr for key in required_keys):
        return False
    
    return True


def box_overlaps_tile(box_x: float, box_y: float, box_width: float, box_height: float,
                      tile_x: int, tile_y: int, tile_width: int, tile_height: int) -> bool:
    """
    Check if a bounding box overlaps with a tile region.
    
    Args:
        box_x: Bounding box left edge
        box_y: Bounding box top edge
        box_width: Bounding box width
        box_height: Bounding box height
        tile_x: Tile left edge
        tile_y: Tile top edge
        tile_width: Width of the tile
        tile_height: Height of the tile
        
    Returns:
        True if the box overlaps the tile, False otherwise
    """
    box_right = box_x + box_width
    box_bottom = box_y + box_height
    tile_right = tile_x + tile_width
    tile_bottom = tile_y + tile_height
    
    # Check for overlap
    return not (box_right <= tile_x or box_x >= tile_right or 
                box_bottom <= tile_y or box_y >= tile_bottom)


def adjust_box_for_tile(box_x: float, box_y: float, box_width: float, box_height: float,
                        tile_x: int, tile_y: int, tile_width: int, tile_height: int) -> Optional[Tuple[float, float, float, float]]:
    """
    Adjust bounding box coordinates to be relative to a tile.
    Clips the box to tile bounds and shifts coordinates.
    
    Args:
        box_x: Bounding box left edge (absolute coordinates)
        box_y: Bounding box top edge (absolute coordinates)
        box_width: Bounding box width
        box_height: Bounding box height
        tile_x: Tile left edge (absolute coordinates)
        tile_y: Tile top edge (absolute coordinates)
        tile_width: Width of the tile
        tile_height: Height of the tile
        
    Returns:
        Tuple of (x, y, width, height) relative to tile, or None if box doesn't overlap
    """
    box_right = box_x + box_width
    box_bottom = box_y + box_height
    tile_right = tile_x + tile_width
    tile_bottom = tile_y + tile_height
    
    # Clip box to tile bounds
    clipped_x = max(box_x, tile_x)
    clipped_y = max(box_y, tile_y)
    clipped_right = min(box_right, tile_right)
    clipped_bottom = min(box_bottom, tile_bottom)
    
    # Check if there's any overlap
    if clipped_right <= clipped_x or clipped_bottom <= clipped_y:
        return None
    
    # Convert to tile-relative coordinates
    relative_x = clipped_x - tile_x
    relative_y = clipped_y - tile_y
    relative_width = clipped_right - clipped_x
    relative_height = clipped_bottom - clipped_y
    
    return (relative_x, relative_y, relative_width, relative_height)


def split_image_into_tiles(image_path: str, img_width: int, img_height: int, 
                           tile_size: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """
    Generate tile positions for an image with overlap.
    
    Args:
        image_path: Path to the image (for context, not used in calculation)
        img_width: Image width
        img_height: Image height
        tile_size: Size of each tile
        overlap: Overlap ratio (e.g., 0.1 for 10%)
        
    Returns:
        List of tuples: (tile_x, tile_y, tile_width, tile_height) for each tile
    """
    tiles = []
    step_size = int(tile_size * (1 - overlap))
    
    y = 0
    while y < img_height:
        x = 0
        while x < img_width:
            # Calculate actual tile dimensions (may be smaller at edges)
            actual_width = min(tile_size, img_width - x)
            actual_height = min(tile_size, img_height - y)
            tiles.append((x, y, actual_width, actual_height))
            x += step_size
        y += step_size
    
    return tiles


def extract_tile_from_image(image_path: str, tile_x: int, tile_y: int, 
                            tile_width: int, tile_height: int, output_path: str) -> None:
    """
    Extract a tile from an image and save it.
    
    Args:
        image_path: Path to source image
        tile_x: Left edge of tile
        tile_y: Top edge of tile
        tile_width: Width of tile
        tile_height: Height of tile
        output_path: Path where tile should be saved
    """
    if PIL_AVAILABLE:
        img = PILImage.open(image_path)
        tile = img.crop((tile_x, tile_y, tile_x + tile_width, tile_y + tile_height))
        tile.save(output_path, 'PNG')
    elif CV2_AVAILABLE:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        tile = img[tile_y:tile_y + tile_height, tile_x:tile_x + tile_width]
        cv2.imwrite(output_path, tile)
    else:
        raise ValueError("PIL/Pillow or OpenCV required for image tiling")


def convert_tif_to_png(image_path: str, output_path: str) -> None:
    """
    Convert TIF/TIFF image to PNG.
    
    Args:
        image_path: Path to the source TIF/TIFF image
        output_path: Path where PNG should be saved
        
    Raises:
        ValueError: If conversion fails or libraries are not available
    """
    if not os.path.exists(output_path):
        try:
            if PIL_AVAILABLE:
                img = PILImage.open(image_path)
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output_path, 'PNG')
            elif CV2_AVAILABLE:
                import cv2
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Could not load TIF with OpenCV")
                cv2.imwrite(output_path, img)
            else:
                raise ValueError("PIL/Pillow or OpenCV required for TIF conversion")
        except Exception as e:
            raise ValueError(f"Failed to convert {image_path} to PNG: {e}")


def process_cvat_dataset(
    dataset_folder: str,
    output_images_dir: str,
    vgg_data: Dict,
    used_filenames: set,
    stats: Dict,
    class_name: str = "worm"
) -> None:
    """
    Process a single CVAT dataset folder and add its images/annotations to the consolidated output.
    
    Args:
        dataset_folder: Path to the CVAT dataset folder
        output_images_dir: Output directory for consolidated images
        vgg_data: Dictionary to add VGG entries to (modified in place)
        used_filenames: Set of used filenames for collision handling (modified in place)
        stats: Statistics dictionary (modified in place)
        class_name: Name of the annotation class
        
    Raises:
        FileNotFoundError: If XML file or images directory cannot be found
        ValueError: If images directory contains no images
    """
    print(f"\n📁 Processing dataset: {dataset_folder}")
    
    # Extract folder name (last element of path) for use in output filenames
    folder_name = os.path.basename(os.path.normpath(dataset_folder))
    # Sanitize folder name for use in filenames (remove invalid characters)
    folder_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in folder_name)
    folder_name = folder_name.strip()
    if not folder_name:
        folder_name = "dataset"  # Fallback if folder name is empty after sanitization
    
    # Find XML file - should already be validated, but double-check
    xml_file = find_cvat_xml_file(dataset_folder)
    if not xml_file:
        # This shouldn't happen if validation passed, but handle it anyway
        raise RuntimeError(
            f"XML file not found in validated dataset folder: {dataset_folder}\n"
            f"This should not happen - validation should have caught this."
        )
    
    print(f"  📄 Found XML: {os.path.basename(xml_file)}")
    
    # Find images directory - should already be validated
    images_dir = find_images_directory(dataset_folder)
    print(f"  🖼️  Images directory: {images_dir}")
    
    # Parse XML - this is required, not optional
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(
            f"Failed to parse XML file: {xml_file}\n"
            f"Error: {e}\n"
            f"This XML file appears to be corrupted or invalid."
        )
    
    # Process each image in the XML
    for image_elem in root.findall('image'):
        stats['total_images'] += 1
        
        try:
            # Get image attributes
            image_name = image_elem.get('name')
            if not image_name:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {image_elem.get('id', 'unknown')}: No name attribute")
                continue
            
            # Get image dimensions from XML
            img_width = int(image_elem.get('width', 0))
            img_height = int(image_elem.get('height', 0))
            
            if img_width <= 0 or img_height <= 0:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {image_name}: Invalid dimensions ({img_width}x{img_height})")
                continue
            
            # Find source image file
            source_image_path = os.path.join(images_dir, image_name)
            
            # Try different extensions if not found
            if not os.path.exists(source_image_path):
                base_name = os.path.splitext(image_name)[0]
                found = False
                for ext in IMAGE_EXTENSIONS:
                    alt_path = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        source_image_path = alt_path
                        image_name = base_name + ext
                        found = True
                        break
                
                if not found:
                    # Image file referenced in XML but doesn't exist - this is an error
                    stats['skipped_images'] += 1
                    error_msg = f"Image file not found: {image_name} (expected in {images_dir})"
                    stats['errors'].append(error_msg)
                    print(f"  ⚠️  Warning: {error_msg}")
                    continue
            
            # Parse bounding box annotations first (before tiling decision)
            original_regions = []
            for box_elem in image_elem.findall('box'):
                region = parse_cvat_box_to_vgg(box_elem, class_name)
                if region:
                    original_regions.append(region)
            
            # Skip images without annotations
            if len(original_regions) == 0:
                stats['skipped_images'] += 1
                continue
            
            # Check if image needs to be tiled (larger than threshold)
            needs_tiling = img_width > MAX_IMAGE_SIZE_FOR_TILING or img_height > MAX_IMAGE_SIZE_FOR_TILING
            
            # Ensure image is loaded/converted to PNG format for tiling
            # First, convert TIF to PNG if needed (create temporary file)
            source_ext = os.path.splitext(source_image_path)[1].lower()
            temp_image_path = source_image_path
            
            if source_ext in ['.tif', '.tiff']:
                # Create temporary PNG file
                temp_png_path = os.path.join(os.path.dirname(source_image_path), 
                                            os.path.splitext(image_name)[0] + '_temp.png')
                try:
                    convert_tif_to_png(source_image_path, temp_png_path)
                    temp_image_path = temp_png_path
                except Exception as e:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {image_name}: TIF conversion error - {e}")
                    continue
            
            if needs_tiling:
                # Split image into tiles
                original_base_filename = os.path.splitext(get_output_filename_for_image(source_image_path))[0]
                # Prepend folder name to base filename
                base_filename = f"{folder_name}_{original_base_filename}"
                tiles = split_image_into_tiles(temp_image_path, img_width, img_height, TILE_SIZE, TILE_OVERLAP)
                
                tiles_with_annotations = 0
                for tile_idx, (tile_x, tile_y, tile_width, tile_height) in enumerate(tiles):
                    # Check which bounding boxes overlap with this tile
                    tile_regions = []
                    for region in original_regions:
                        shape_attr = region['shape_attributes']
                        box_x = shape_attr['x']
                        box_y = shape_attr['y']
                        box_width = shape_attr['width']
                        box_height = shape_attr['height']
                        
                        # Check if box overlaps with tile (use actual tile dimensions)
                        if box_overlaps_tile(box_x, box_y, box_width, box_height, 
                                           tile_x, tile_y, tile_width, tile_height):
                            # Calculate original box area
                            original_area = box_width * box_height
                            
                            # Adjust box coordinates to be relative to tile (use actual tile dimensions)
                            adjusted = adjust_box_for_tile(box_x, box_y, box_width, box_height,
                                                          tile_x, tile_y, tile_width, tile_height)
                            if adjusted:
                                adj_x, adj_y, adj_width, adj_height = adjusted
                                
                                # Calculate clipped box area
                                clipped_area = adj_width * adj_height
                                
                                # Only include if >= 20% of original area
                                if original_area > 0 and (clipped_area / original_area) >= 0.2:
                                    tile_region = {
                                        "shape_attributes": {
                                            "name": "rect",
                                            "x": int(round(adj_x)),
                                            "y": int(round(adj_y)),
                                            "width": int(round(adj_width)),
                                            "height": int(round(adj_height))
                                        },
                                        "region_attributes": region['region_attributes']
                                    }
                                    tile_regions.append(tile_region)
                    
                    # Only create tile if it has annotations
                    if len(tile_regions) > 0:
                        # Generate tile filename: base_name_tile_row_col.png
                        # Calculate row and col indices
                        step_size = int(TILE_SIZE * (1 - TILE_OVERLAP))
                        row = tile_y // step_size
                        col = tile_x // step_size
                        tile_filename = f"{base_filename}_tile_{row:04d}_{col:04d}.png"
                        tile_filename = handle_filename_collision(
                            os.path.splitext(tile_filename)[0],
                            os.path.splitext(tile_filename)[1],
                            used_filenames
                        )
                        tile_output_path = os.path.join(output_images_dir, tile_filename)
                        
                        # Extract tile
                        try:
                            extract_tile_from_image(temp_image_path, tile_x, tile_y, 
                                                   tile_width, tile_height, tile_output_path)
                        except Exception as e:
                            stats['errors'].append(f"Tile {tile_idx} from {image_name}: Extraction error - {e}")
                            continue
                        
                        # Get file size
                        tile_file_size = os.path.getsize(tile_output_path)
                        
                        # Validate all regions are rects (safeguard against polygons)
                        validated_tile_regions = []
                        for region in tile_regions:
                            if validate_region_is_rect(region):
                                validated_tile_regions.append(region)
                            else:
                                # This shouldn't happen, but convert as fallback
                                bbox_region = parse_vgg_region_to_bounding_box(region, class_name)
                                if bbox_region:
                                    validated_tile_regions.append(bbox_region)
                        
                        # Create VGG entry for tile
                        vgg_img_id = f"{tile_filename}{tile_file_size}"
                        img_entry = {
                            "filename": tile_filename,
                            "size": tile_file_size,
                            "regions": validated_tile_regions,
                            "file_attributes": {}
                        }
                        
                        # Handle VGG ID collisions
                        original_vgg_id = vgg_img_id
                        counter = 0
                        while vgg_img_id in vgg_data:
                            counter += 1
                            vgg_img_id = f"{tile_filename}{tile_file_size}_{counter}"
                        
                        vgg_data[vgg_img_id] = img_entry
                        tiles_with_annotations += 1
                        stats['total_annotations'] += len(tile_regions)
                
                if tiles_with_annotations > 0:
                    stats['converted_images'] += 1
                    stats['images_tiled'] += 1
                    stats['tiles_created'] += tiles_with_annotations
                    print(f"    ✓ Split into {tiles_with_annotations} tiles with annotations")
                else:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {image_name}: No tiles with annotations after splitting")
                
                # Clean up temporary PNG file if created
                if temp_image_path != source_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass
            
            else:
                # Image is small enough - process normally without tiling
                # Determine output filename (convert TIF to PNG extension)
                original_filename = get_output_filename_for_image(source_image_path)
                # Prepend folder name to filename
                base_name, ext = os.path.splitext(original_filename)
                proposed_filename = f"{folder_name}_{base_name}{ext}"
                
                # Handle filename collision
                output_image_name = handle_filename_collision(
                    os.path.splitext(proposed_filename)[0],
                    os.path.splitext(proposed_filename)[1],
                    used_filenames
                )
                output_image_path = os.path.join(output_images_dir, output_image_name)
                
                # Copy or convert image to output directory
                try:
                    if source_ext in ['.tif', '.tiff']:
                        # Convert TIF to PNG
                        convert_tif_to_png(source_image_path, output_image_path)
                    else:
                        # Copy as-is
                        if not os.path.exists(output_image_path):
                            shutil.copy2(source_image_path, output_image_path)
                except Exception as e:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {image_name}: Copy/conversion error - {e}")
                    continue
                
                # Get file size
                file_size = os.path.getsize(output_image_path)
                
                # Ensure we use the actual output filename (in case file already existed with different name)
                actual_output_filename = os.path.basename(output_image_path)
                
                # Validate all regions are rects (safeguard against polygons)
                validated_regions = []
                for region in original_regions:
                    if validate_region_is_rect(region):
                        validated_regions.append(region)
                    else:
                        # This shouldn't happen if conversion worked correctly
                        # Try to convert again as a fallback
                        bbox_region = parse_vgg_region_to_bounding_box(region, class_name)
                        if bbox_region:
                            validated_regions.append(bbox_region)
                        else:
                            stats['errors'].append(f"Image {image_name}: Region is not rect format and conversion failed - skipped")
                
                # Create VGG image entry - use actual output filename to ensure consistency
                vgg_img_id = f"{actual_output_filename}{file_size}"
                img_entry = {
                    "filename": actual_output_filename,
                    "size": file_size,
                    "regions": validated_regions,
                    "file_attributes": {}
                }
                
                # Add to consolidated VGG data (handle collisions in VGG IDs)
                if vgg_img_id in vgg_data:
                    # Collision in VGG ID - append counter
                    counter = 1
                    while f"{actual_output_filename}{file_size}_{counter}" in vgg_data:
                        counter += 1
                    vgg_img_id = f"{actual_output_filename}{file_size}_{counter}"
                
                vgg_data[vgg_img_id] = img_entry
                stats['converted_images'] += 1
                stats['total_annotations'] += len(validated_regions)
            
        except Exception as e:
            stats['skipped_images'] += 1
            image_id = image_elem.get('id', 'unknown')
            stats['errors'].append(f"Image {image_id}: {str(e)}")
            continue


def process_vgg_json_dataset(
    dataset_folder: str,
    output_images_dir: str,
    vgg_data: Dict,
    used_filenames: set,
    stats: Dict,
    class_name: str = "worm"
) -> None:
    """
    Process a single VGG JSON dataset folder and add its images/annotations to the consolidated output.
    
    Args:
        dataset_folder: Path to the VGG JSON dataset folder
        output_images_dir: Output directory for consolidated images
        vgg_data: Dictionary to add VGG entries to (modified in place)
        used_filenames: Set of used filenames for collision handling (modified in place)
        stats: Statistics dictionary (modified in place)
        class_name: Name of the annotation class
        
    Raises:
        FileNotFoundError: If JSON file or images directory cannot be found
        ValueError: If images directory contains no images or JSON is invalid
    """
    print(f"\n📁 Processing VGG JSON dataset: {dataset_folder}")
    
    # Extract folder name for use in output filenames
    folder_name = os.path.basename(os.path.normpath(dataset_folder))
    folder_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in folder_name)
    folder_name = folder_name.strip()
    if not folder_name:
        folder_name = "dataset"
    
    # Find VGG JSON file
    vgg_json_file = find_vgg_json_file(dataset_folder)
    if not vgg_json_file:
        raise RuntimeError(
            f"VGG JSON file not found in validated dataset folder: {dataset_folder}\n"
            f"This should not happen - validation should have caught this."
        )
    
    print(f"  📄 Found VGG JSON: {os.path.basename(vgg_json_file)}")
    
    # Find images directory
    images_dir = find_images_directory(dataset_folder)
    print(f"  🖼️  Images directory: {images_dir}")
    
    # Load and parse VGG JSON
    try:
        with open(vgg_json_file, 'r', encoding='utf-8') as f:
            vgg_json_data = json.load(f)
    except Exception as e:
        raise ValueError(
            f"Failed to load VGG JSON file: {vgg_json_file}\n"
            f"Error: {e}"
        )
    
    # Parse VGG JSON structure
    if isinstance(vgg_json_data, dict):
        if '_via_img_metadata' in vgg_json_data:
            img_metadata = vgg_json_data['_via_img_metadata']
        else:
            img_metadata = vgg_json_data
    elif isinstance(vgg_json_data, list):
        # Convert list to dict format
        img_metadata = {}
        for item in vgg_json_data:
            if isinstance(item, dict) and 'filename' in item:
                img_id = item.get('filename', 'unknown')
                img_metadata[img_id] = item
    else:
        raise ValueError(f"Unsupported VGG JSON format in: {vgg_json_file}")
    
    # Process each image in the VGG JSON
    for img_id, img_data in img_metadata.items():
        stats['total_images'] += 1
        
        try:
            # Get filename
            image_name = img_data.get('filename') or img_data.get('fname')
            if not image_name:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {img_id}: No filename found")
                continue
            
            # Find source image file
            source_image_path = os.path.join(images_dir, image_name)
            
            # Try different extensions if not found
            if not os.path.exists(source_image_path):
                base_name = os.path.splitext(image_name)[0]
                found = False
                for ext in IMAGE_EXTENSIONS:
                    alt_path = os.path.join(images_dir, base_name + ext)
                    if os.path.exists(alt_path):
                        source_image_path = alt_path
                        image_name = base_name + ext
                        found = True
                        break
                
                if not found:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {image_name}: File not found in {images_dir}")
                    continue
            
            # Load image to get dimensions
            try:
                if PIL_AVAILABLE:
                    img = PILImage.open(source_image_path)
                    img_width, img_height = img.size
                elif CV2_AVAILABLE:
                    import cv2
                    img = cv2.imread(source_image_path)
                    if img is None:
                        raise ValueError("Could not load image")
                    img_height, img_width = img.shape[:2]
                else:
                    raise ValueError("PIL/Pillow or OpenCV required to load images")
            except Exception as e:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {image_name}: Could not load - {e}")
                continue
            
            # Parse regions and convert polygons to bounding boxes
            original_regions = []
            regions = img_data.get('regions') or img_data.get('via_region_data', {})
            
            if isinstance(regions, list):
                for region in regions:
                    # Ensure we always convert polygons to rects
                    bbox_region = parse_vgg_region_to_bounding_box(region, class_name)
                    if bbox_region:
                        original_regions.append(bbox_region)
                    else:
                        # Log if conversion fails (but don't add the region)
                        shape_attr = region.get('shape_attributes') or region
                        shape_name = shape_attr.get('name', 'unknown')
                        stats['errors'].append(f"Image {image_name}: Failed to convert {shape_name} region to bounding box (skipped)")
            elif isinstance(regions, dict):
                for region_id, region in regions.items():
                    # Ensure we always convert polygons to rects
                    bbox_region = parse_vgg_region_to_bounding_box(region, class_name)
                    if bbox_region:
                        original_regions.append(bbox_region)
                    else:
                        # Log if conversion fails (but don't add the region)
                        shape_attr = region.get('shape_attributes') or region
                        shape_name = shape_attr.get('name', 'unknown')
                        stats['errors'].append(f"Image {image_name}: Failed to convert {shape_name} region to bounding box (skipped)")
            
            # Skip images without annotations
            if len(original_regions) == 0:
                stats['skipped_images'] += 1
                continue
            
            # Check if image needs to be tiled (larger than threshold)
            needs_tiling = img_width > MAX_IMAGE_SIZE_FOR_TILING or img_height > MAX_IMAGE_SIZE_FOR_TILING
            
            # Ensure image is loaded/converted to PNG format for tiling
            source_ext = os.path.splitext(source_image_path)[1].lower()
            temp_image_path = source_image_path
            
            if source_ext in ['.tif', '.tiff']:
                # Create temporary PNG file
                temp_png_path = os.path.join(os.path.dirname(source_image_path), 
                                            os.path.splitext(image_name)[0] + '_temp.png')
                try:
                    convert_tif_to_png(source_image_path, temp_png_path)
                    temp_image_path = temp_png_path
                except Exception as e:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {image_name}: TIF conversion error - {e}")
                    continue
            
            if needs_tiling:
                # Split image into tiles (same logic as CVAT processing)
                original_base_filename = os.path.splitext(get_output_filename_for_image(source_image_path))[0]
                base_filename = f"{folder_name}_{original_base_filename}"
                tiles = split_image_into_tiles(temp_image_path, img_width, img_height, TILE_SIZE, TILE_OVERLAP)
                
                tiles_with_annotations = 0
                for tile_idx, (tile_x, tile_y, tile_width, tile_height) in enumerate(tiles):
                    # Check which bounding boxes overlap with this tile
                    tile_regions = []
                    for region in original_regions:
                        shape_attr = region['shape_attributes']
                        box_x = shape_attr['x']
                        box_y = shape_attr['y']
                        box_width = shape_attr['width']
                        box_height = shape_attr['height']
                        
                        # Check if box overlaps with tile
                        if box_overlaps_tile(box_x, box_y, box_width, box_height, 
                                           tile_x, tile_y, tile_width, tile_height):
                            # Calculate original box area
                            original_area = box_width * box_height
                            
                            # Adjust box coordinates to be relative to tile
                            adjusted = adjust_box_for_tile(box_x, box_y, box_width, box_height,
                                                          tile_x, tile_y, tile_width, tile_height)
                            if adjusted:
                                adj_x, adj_y, adj_width, adj_height = adjusted
                                
                                # Calculate clipped box area
                                clipped_area = adj_width * adj_height
                                
                                # Only include if >= 20% of original area
                                if original_area > 0 and (clipped_area / original_area) >= 0.2:
                                    tile_region = {
                                        "shape_attributes": {
                                            "name": "rect",
                                            "x": int(round(adj_x)),
                                            "y": int(round(adj_y)),
                                            "width": int(round(adj_width)),
                                            "height": int(round(adj_height))
                                        },
                                        "region_attributes": region['region_attributes']
                                    }
                                    tile_regions.append(tile_region)
                    
                    # Only create tile if it has annotations
                    if len(tile_regions) > 0:
                        # Generate tile filename
                        step_size = int(TILE_SIZE * (1 - TILE_OVERLAP))
                        row = tile_y // step_size
                        col = tile_x // step_size
                        tile_filename = f"{base_filename}_tile_{row:04d}_{col:04d}.png"
                        tile_filename = handle_filename_collision(
                            os.path.splitext(tile_filename)[0],
                            os.path.splitext(tile_filename)[1],
                            used_filenames
                        )
                        tile_output_path = os.path.join(output_images_dir, tile_filename)
                        
                        # Extract tile
                        try:
                            extract_tile_from_image(temp_image_path, tile_x, tile_y, 
                                                   tile_width, tile_height, tile_output_path)
                        except Exception as e:
                            stats['errors'].append(f"Tile {tile_idx} from {image_name}: Extraction error - {e}")
                            continue
                        
                        # Get file size
                        tile_file_size = os.path.getsize(tile_output_path)
                        
                        # Validate all regions are rects (safeguard against polygons)
                        validated_tile_regions = []
                        for region in tile_regions:
                            if validate_region_is_rect(region):
                                validated_tile_regions.append(region)
                            else:
                                # This shouldn't happen, but convert as fallback
                                bbox_region = parse_vgg_region_to_bounding_box(region, class_name)
                                if bbox_region:
                                    validated_tile_regions.append(bbox_region)
                        
                        # Create VGG entry for tile
                        vgg_img_id = f"{tile_filename}{tile_file_size}"
                        img_entry = {
                            "filename": tile_filename,
                            "size": tile_file_size,
                            "regions": validated_tile_regions,
                            "file_attributes": {}
                        }
                        
                        # Handle VGG ID collisions
                        counter = 0
                        while vgg_img_id in vgg_data:
                            counter += 1
                            vgg_img_id = f"{tile_filename}{tile_file_size}_{counter}"
                        
                        vgg_data[vgg_img_id] = img_entry
                        tiles_with_annotations += 1
                        stats['total_annotations'] += len(validated_tile_regions)
                
                if tiles_with_annotations > 0:
                    stats['converted_images'] += 1
                    stats['images_tiled'] += 1
                    stats['tiles_created'] += tiles_with_annotations
                    print(f"    ✓ Split into {tiles_with_annotations} tiles with annotations")
                else:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {image_name}: No tiles with annotations after splitting")
                
                # Clean up temporary PNG file if created
                if temp_image_path != source_image_path and os.path.exists(temp_image_path):
                    try:
                        os.remove(temp_image_path)
                    except:
                        pass
            
            else:
                # Image is small enough - process normally without tiling
                original_filename = get_output_filename_for_image(source_image_path)
                base_name, ext = os.path.splitext(original_filename)
                proposed_filename = f"{folder_name}_{base_name}{ext}"
                
                # Handle filename collision
                output_image_name = handle_filename_collision(
                    os.path.splitext(proposed_filename)[0],
                    os.path.splitext(proposed_filename)[1],
                    used_filenames
                )
                output_image_path = os.path.join(output_images_dir, output_image_name)
                
                # Copy or convert image to output directory
                try:
                    if source_ext in ['.tif', '.tiff']:
                        convert_tif_to_png(source_image_path, output_image_path)
                    else:
                        if not os.path.exists(output_image_path):
                            shutil.copy2(source_image_path, output_image_path)
                except Exception as e:
                    stats['skipped_images'] += 1
                    stats['errors'].append(f"Image {image_name}: Copy/conversion error - {e}")
                    continue
                
                # Get file size
                file_size = os.path.getsize(output_image_path)
                
                # Ensure we use the actual output filename (in case file already existed with different name)
                actual_output_filename = os.path.basename(output_image_path)
                
                # Validate all regions are rects (safeguard against polygons)
                validated_regions = []
                for region in original_regions:
                    if validate_region_is_rect(region):
                        validated_regions.append(region)
                    else:
                        # This shouldn't happen if conversion worked correctly
                        # Try to convert again as a fallback
                        bbox_region = parse_vgg_region_to_bounding_box(region, class_name)
                        if bbox_region:
                            validated_regions.append(bbox_region)
                        else:
                            stats['errors'].append(f"Image {image_name}: Region is not rect format and conversion failed - skipped")
                
                # Create VGG image entry - use actual output filename to ensure consistency
                vgg_img_id = f"{actual_output_filename}{file_size}"
                img_entry = {
                    "filename": actual_output_filename,
                    "size": file_size,
                    "regions": validated_regions,
                    "file_attributes": {}
                }
                
                # Handle VGG ID collisions
                if vgg_img_id in vgg_data:
                    counter = 1
                    while f"{actual_output_filename}{file_size}_{counter}" in vgg_data:
                        counter += 1
                    vgg_img_id = f"{actual_output_filename}{file_size}_{counter}"
                
                vgg_data[vgg_img_id] = img_entry
                stats['converted_images'] += 1
                stats['total_annotations'] += len(validated_regions)
            
        except Exception as e:
            stats['skipped_images'] += 1
            stats['errors'].append(f"Image {img_id}: {str(e)}")
            continue


def consolidate_cvat_datasets(
    dataset_folders: List[str],
    output_images_dir: str,
    output_json_file: str,
    class_name: str = "worm",
    config_json_path: Optional[str] = None
) -> Dict:
    """
    Consolidate multiple CVAT XML or VGG JSON datasets into a single folder and VGG JSON file.
    
    Supports both CVAT XML format (with bounding boxes) and VGG JSON format (with polygons,
    rectangles, or circles). Polygons and circles are converted to bounding boxes.
    
    Args:
        dataset_folders: List of paths to dataset folders (CVAT XML or VGG JSON)
        output_images_dir: Output directory for consolidated images
        output_json_file: Path to output VGG JSON file
        class_name: Name of the class (default: "worm")
        
    Returns:
        Dictionary with consolidation statistics
        
    Raises:
        ValueError: If output directory is not empty or input validation fails
        FileNotFoundError: If required input folders don't exist
    """
    # Validate output directory - must not contain image files (JSON files are allowed)
    if os.path.exists(output_images_dir):
        contents = list(Path(output_images_dir).iterdir())
        
        # Filter out only image files - allow JSON files and other files
        image_files = []
        for item in contents:
            # Skip directories, only check files
            if not item.is_file():
                continue
            
            # Check if it's an image file based on extension
            if item.suffix.lower() in [ext.lower() for ext in IMAGE_EXTENSIONS]:
                image_files.append(item)
        
        if len(image_files) > 0:
            raise ValueError(
                f"Output directory contains image files: {output_images_dir}\n"
                f"Found {len(image_files)} image file(s). Please use an empty directory or a directory without images to avoid conflicts.\n"
                f"(Note: JSON files are allowed in the output directory)"
            )
    else:
        # Create output directory if it doesn't exist
        try:
            os.makedirs(output_images_dir, exist_ok=True)
        except OSError as e:
            raise ValueError(
                f"Cannot create output directory: {output_images_dir}\n"
                f"Error: {e}"
            )
    
    # Check output JSON file path if it's outside the output directory
    if os.path.dirname(output_json_file):
        json_output_path = output_json_file
        json_dir = os.path.dirname(json_output_path)
        if os.path.exists(json_output_path):
            raise ValueError(
                f"Output JSON file already exists: {json_output_path}\n"
                f"Please remove it or choose a different output location."
            )
        # Ensure the directory for JSON file exists or can be created
        if json_dir and not os.path.exists(json_dir):
            try:
                os.makedirs(json_dir, exist_ok=True)
            except OSError as e:
                raise ValueError(
                    f"Cannot create directory for JSON output: {json_dir}\n"
                    f"Error: {e}"
                )
    
    # Statistics
    stats = {
        'total_datasets': len(dataset_folders),
        'processed_datasets': 0,
        'skipped_datasets': 0,
        'total_images': 0,
        'converted_images': 0,
        'skipped_images': 0,
        'total_annotations': 0,
        'tiles_created': 0,
        'images_tiled': 0,
        'errors': []
    }
    
    # Consolidated VGG data
    vgg_data = {}
    used_filenames = set()
    
    # Process each dataset
    print("=" * 80)
    print("Dataset Consolidation (CVAT XML / VGG JSON)")
    print("=" * 80)
    print(f"Processing {len(dataset_folders)} dataset(s)...")
    
    # Validate all input dataset folders exist and have required files before processing
    validation_errors = []
    
    for dataset_folder in dataset_folders:
        # Check folder exists
        if not os.path.exists(dataset_folder):
            validation_errors.append(f"Dataset folder does not exist: {dataset_folder}")
            continue
        
        # Check folder is actually a directory
        if not os.path.isdir(dataset_folder):
            validation_errors.append(f"Dataset path is not a directory: {dataset_folder}")
            continue
        
        # Check for annotation file (CVAT XML or VGG JSON)
        xml_file = find_cvat_xml_file(dataset_folder)
        vgg_json_file = find_vgg_json_file(dataset_folder)
        
        if not xml_file and not vgg_json_file:
            validation_errors.append(
                f"Dataset folder missing annotation file: {dataset_folder}\n"
                f"  Expected to find either a CVAT XML file (.xml) or VGG JSON file (.json) in this directory."
            )
            continue
        
        # Validate annotation file can be parsed
        if xml_file:
            # Validate XML file can be parsed (check for corruption)
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                # Check it has the expected structure (should have 'image' elements)
                if root.tag != 'annotations' and len(root.findall('image')) == 0:
                    validation_errors.append(
                        f"XML file has invalid structure: {xml_file}\n"
                        f"  Expected CVAT XML format with <image> elements.\n"
                        f"  For dataset: {dataset_folder}"
                    )
                    continue
            except ET.ParseError as e:
                validation_errors.append(
                    f"XML file is corrupted or invalid: {xml_file}\n"
                    f"  Parse error: {e}\n"
                    f"  For dataset: {dataset_folder}"
                )
                continue
            except Exception as e:
                validation_errors.append(
                    f"Cannot read XML file: {xml_file}\n"
                    f"  Error: {e}\n"
                    f"  For dataset: {dataset_folder}"
                )
                continue
        elif vgg_json_file:
            # Validate VGG JSON file can be parsed
            try:
                with open(vgg_json_file, 'r', encoding='utf-8') as f:
                    vgg_data = json.load(f)
                
                # Check it has the expected structure
                if isinstance(vgg_data, dict):
                    # Check if it has image entries (either direct or in _via_img_metadata)
                    if '_via_img_metadata' in vgg_data:
                        img_metadata = vgg_data['_via_img_metadata']
                    else:
                        img_metadata = vgg_data
                    
                    if len(img_metadata) == 0:
                        validation_errors.append(
                            f"VGG JSON file has no image entries: {vgg_json_file}\n"
                            f"  For dataset: {dataset_folder}"
                        )
                        continue
                elif not isinstance(vgg_data, (dict, list)):
                    validation_errors.append(
                        f"VGG JSON file has invalid structure: {vgg_json_file}\n"
                        f"  Expected a dictionary or list.\n"
                        f"  For dataset: {dataset_folder}"
                    )
                    continue
            except json.JSONDecodeError as e:
                validation_errors.append(
                    f"VGG JSON file is corrupted or invalid: {vgg_json_file}\n"
                    f"  JSON parse error: {e}\n"
                    f"  For dataset: {dataset_folder}"
                )
                continue
            except Exception as e:
                validation_errors.append(
                    f"Cannot read VGG JSON file: {vgg_json_file}\n"
                    f"  Error: {e}\n"
                    f"  For dataset: {dataset_folder}"
                )
                continue
        
        # Check for images directory with images
        images_dir = find_images_directory(dataset_folder)
        if not os.path.exists(images_dir):
            validation_errors.append(
                f"Dataset images directory does not exist: {images_dir}\n"
                f"  For dataset: {dataset_folder}"
            )
            continue
        
        # Verify images directory has image files
        images_dir_path = Path(images_dir)
        image_files = []
        for ext in IMAGE_EXTENSIONS:
            image_files.extend(images_dir_path.glob(f"*{ext}"))
        
        if len(image_files) == 0:
            validation_errors.append(
                f"Dataset images directory contains no image files: {images_dir}\n"
                f"  Expected to find image files (.png, .jpg, .tif, etc.) in this directory.\n"
                f"  For dataset: {dataset_folder}"
            )
    
    # Report all validation errors at once before processing
    if validation_errors:
        error_msg = "Validation failed for one or more dataset folders:\n\n"
        error_msg += "\n".join(f"  {i+1}. {error}" for i, error in enumerate(validation_errors))
        error_msg += "\n\nPlease fix these issues before running the consolidation."
        raise ValueError(error_msg)
    
    # All validation passed - process datasets
    for dataset_folder in dataset_folders:
        
        try:
            # Check if this is a CVAT XML or VGG JSON dataset
            xml_file = find_cvat_xml_file(dataset_folder)
            vgg_json_file = find_vgg_json_file(dataset_folder)
            
            if xml_file:
                process_cvat_dataset(
                    dataset_folder,
                    output_images_dir,
                    vgg_data,
                    used_filenames,
                    stats,
                    class_name
                )
            elif vgg_json_file:
                process_vgg_json_dataset(
                    dataset_folder,
                    output_images_dir,
                    vgg_data,
                    used_filenames,
                    stats,
                    class_name
                )
            else:
                # This shouldn't happen if validation passed
                raise RuntimeError(f"Neither XML nor JSON file found in validated dataset: {dataset_folder}")
            
            stats['processed_datasets'] += 1
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            # Critical errors - these should fail the whole operation
            error_msg = f"Dataset {dataset_folder}: {str(e)}"
            print(f"  ❌ Error processing dataset: {error_msg}")
            stats['skipped_datasets'] += 1
            stats['errors'].append(error_msg)
            # Re-raise to stop processing - these are critical errors
            raise
        except Exception as e:
            # Unexpected errors - log and fail
            error_msg = f"Dataset {dataset_folder}: Unexpected error - {str(e)}"
            print(f"  ❌ Error processing dataset: {error_msg}")
            stats['skipped_datasets'] += 1
            stats['errors'].append(error_msg)
            import traceback
            traceback.print_exc()
            raise
    
    # Write consolidated VGG JSON file
    # If output_json_file is just a filename (no directory), put it in the output_images_dir
    if os.path.dirname(output_json_file):
        # Full path provided - use it as-is
        json_output_path = output_json_file
    else:
        # Just a filename - put it in the output images directory
        json_output_path = os.path.join(output_images_dir, output_json_file)
    
    os.makedirs(os.path.dirname(json_output_path) if os.path.dirname(json_output_path) else '.', exist_ok=True)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(vgg_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Consolidation Statistics")
    print("=" * 80)
    print(f"Total datasets: {stats['total_datasets']}")
    print(f"Processed datasets: {stats['processed_datasets']}")
    print(f"Skipped datasets: {stats['skipped_datasets']}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successfully converted: {stats['converted_images']}")
    print(f"Skipped images: {stats['skipped_images']}")
    print(f"Images tiled: {stats['images_tiled']}")
    print(f"Tiles created: {stats['tiles_created']}")
    print(f"Total annotations: {stats['total_annotations']}")
    print(f"Output images directory: {os.path.abspath(output_images_dir)}")
    # Calculate the actual JSON output path
    if os.path.dirname(output_json_file):
        actual_json_path = output_json_file
    else:
        actual_json_path = os.path.join(output_images_dir, output_json_file)
    print(f"Output JSON file: {os.path.abspath(actual_json_path)}")
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:20]:  # Show first 20 errors
            print(f"  - {error}")
        if len(stats['errors']) > 20:
            print(f"  ... and {len(stats['errors']) - 20} more errors")
    print("=" * 80)
    print("\n✅ Consolidation complete!")
    print("=" * 80)
    
    return stats


def load_config(config_json_path: str) -> Dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_json_path: Path to the configuration JSON file
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if not os.path.exists(config_json_path):
        raise FileNotFoundError(f"Configuration file not found: {config_json_path}")
    
    with open(config_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate required fields
    required_fields = ['cvat_dataset_folders', 'output_images_dir']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")
    
    # Set defaults for optional fields
    config.setdefault('output_json_file', 'consolidated_annotations.json')
    config.setdefault('class_name', 'worm')
    config.setdefault('tile_size', 512)
    config.setdefault('tile_overlap', 0.1)
    config.setdefault('max_image_size_for_tiling', 512)
    
    return config


def select_config_file_gui() -> Optional[str]:
    """
    Show GUI file dialog to select configuration JSON file.
    
    Returns:
        Path to selected config file, or None if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("❌ Error: tkinter is not available. Please install it or provide config file path.")
        print("   On Linux, install with: sudo apt-get install python3-tk")
        return None
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    print("📁 Please select the configuration JSON file...")
    config_file = filedialog.askopenfilename(
        title="Select Configuration JSON File",
        filetypes=[
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ],
        initialdir=os.getcwd()
    )
    
    root.destroy()
    
    if not config_file:
        print("❌ No configuration file selected. Cancelled.")
        return None
    
    return config_file


def main():
    """Main function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Consolidate multiple CVAT XML annotation datasets into a single folder and VGG JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use GUI to select config file
  python consolidateCVATDatasets.py
  
  # Specify config file directly
  python consolidateCVATDatasets.py config.json
        """
    )
    
    parser.add_argument(
        'config_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to configuration JSON file (optional, will open file dialog if not provided)'
    )
    
    args = parser.parse_args()
    
    # Get config file path
    if args.config_file:
        config_json_path = args.config_file
    else:
        # Use GUI to select config file
        config_json_path = select_config_file_gui()
        if not config_json_path:
            return 1
    
    # Load configuration
    try:
        config = load_config(config_json_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in configuration file: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # Extract configuration values
    cvat_dataset_folders = config['cvat_dataset_folders']
    output_images_dir = config['output_images_dir']
    output_json_file = config['output_json_file']
    class_name = config['class_name']
    
    # Make tiling config available globally (used in functions)
    global TILE_SIZE, TILE_OVERLAP, MAX_IMAGE_SIZE_FOR_TILING
    TILE_SIZE = config['tile_size']
    TILE_OVERLAP = config['tile_overlap']
    MAX_IMAGE_SIZE_FOR_TILING = config['max_image_size_for_tiling']
    
    # Validate dataset folders
    if not cvat_dataset_folders or len(cvat_dataset_folders) == 0:
        print("⚠️  Warning: No dataset folders configured in configuration file!")
        return 1
    
    # Check for image conversion libraries
    if not PIL_AVAILABLE and not CV2_AVAILABLE:
        print("⚠️  Warning: PIL/Pillow or OpenCV not available. TIF to PNG conversion will fail.")
        print("   Install with: pip install Pillow  or  pip install opencv-python")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Validate configuration before running
    if not cvat_dataset_folders or len(cvat_dataset_folders) == 0:
        print("❌ Error: No dataset folders specified in configuration file!")
        return 1
    
    # Run consolidation
    try:
        stats = consolidate_cvat_datasets(
            cvat_dataset_folders,
            output_images_dir,
            output_json_file,
            class_name,
            config_json_path
        )
        
        # Check if we actually processed anything
        if stats['total_images'] == 0:
            print("\n❌ Error: No images were found to process!")
            print("   Please verify that:")
            print("   - Dataset folders contain valid CVAT XML files")
            print("   - XML files reference images that exist in the image directories")
            return 1
        
        if stats['converted_images'] == 0:
            print("\n❌ Error: No images were successfully consolidated!")
            print(f"   Processed {stats['total_images']} images but none were converted.")
            if stats['errors']:
                print(f"   Found {len(stats['errors'])} error(s). See details above.")
            return 1
        
        if stats['skipped_datasets'] > 0:
            print(f"\n⚠️  Warning: {stats['skipped_datasets']} dataset(s) were skipped due to errors!")
            print("   See error details above.")
            # Still return error if any datasets were skipped
            if stats['processed_datasets'] == 0:
                return 1
        
        print(f"\n✅ Successfully consolidated {stats['converted_images']} images!")
        return 0
            
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during consolidation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

