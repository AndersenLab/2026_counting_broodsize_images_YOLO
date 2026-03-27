"""
CVAT XML to VGG JSON Converter
Converts CVAT XML annotation format to VGG Image Annotator (VIA) JSON format

This script converts bounding box annotations from CVAT XML format to VGG JSON format,
which can then be used with tools that expect VGG JSON annotations.

Usage:
    python cvat_to_vgg.py input.xml output.json [--images_dir path/to/images]
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
import argparse
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    try:
        import cv2
        CV2_AVAILABLE = True
        PIL_AVAILABLE = False
    except ImportError:
        PIL_AVAILABLE = False
        CV2_AVAILABLE = False


def convert_cvat_xml_to_vgg_json(
    cvat_xml_path: str,
    output_json_path: str,
    images_dir: Optional[str] = None,
    class_name: str = "worm"
) -> Dict:
    """
    Convert CVAT XML annotations to VGG JSON format.
    
    Args:
        cvat_xml_path: Path to CVAT XML annotation file
        output_json_path: Path to output VGG JSON file
        images_dir: Optional directory containing images (for size calculation)
        class_name: Name of the class (default: "worm")
    
    Returns:
        Dictionary with conversion statistics
    """
    # Parse CVAT XML
    try:
        tree = ET.parse(cvat_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse CVAT XML file: {e}")
    
    # Statistics
    stats = {
        'total_images': 0,
        'converted_images': 0,
        'skipped_images': 0,
        'total_annotations': 0,
        'errors': []
    }
    
    # Build VGG JSON structure (simple format - just image metadata)
    vgg_data = {}
    
    # Check if image conversion is available
    if images_dir and not PIL_AVAILABLE and not CV2_AVAILABLE:
        print("⚠️  Warning: PIL/Pillow or OpenCV not available. TIF to PNG conversion will be skipped.")
        print("   Install with: pip install Pillow  or  pip install opencv-python")
    
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
            
            # Get image dimensions from XML
            img_width = int(image_elem.get('width', 0))
            img_height = int(image_elem.get('height', 0))
            
            if img_width <= 0 or img_height <= 0:
                stats['skipped_images'] += 1
                stats['errors'].append(f"Image {image_name}: Invalid dimensions ({img_width}x{img_height})")
                continue
            
            # Find and convert image if needed
            file_size = 0
            output_image_name = image_name  # Will be updated if conversion happens
            
            if images_dir:
                image_path = os.path.join(images_dir, image_name)
                original_path = None
                
                # Try to find the image file
                if not os.path.exists(image_path):
                    # Try different extensions
                    base_name = os.path.splitext(image_name)[0]
                    found = False
                    for ext in ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.TIF', '.TIFF', '.PNG', '.JPG', '.JPEG']:
                        alt_path = os.path.join(images_dir, base_name + ext)
                        if os.path.exists(alt_path):
                            original_path = alt_path
                            image_path = alt_path
                            found = True
                            break
                    
                    if not found:
                        stats['skipped_images'] += 1
                        stats['errors'].append(f"Image {image_name}: File not found in {images_dir}")
                        continue
                else:
                    original_path = image_path
                
                # Check if image is TIF/TIFF and convert to PNG
                image_ext = os.path.splitext(original_path)[1].lower()
                if image_ext in ['.tif', '.tiff']:
                    # Convert TIF to PNG
                    png_path = os.path.join(images_dir, os.path.splitext(image_name)[0] + '.png')
                    
                    if not os.path.exists(png_path):
                        # Perform conversion
                        try:
                            if PIL_AVAILABLE:
                                # Use PIL/Pillow
                                img = Image.open(original_path)
                                img.save(png_path, 'PNG')
                                print(f"  ✅ Converted {image_name} -> {os.path.basename(png_path)}")
                            elif CV2_AVAILABLE:
                                # Use OpenCV
                                img = cv2.imread(original_path)
                                cv2.imwrite(png_path, img)
                                print(f"  ✅ Converted {image_name} -> {os.path.basename(png_path)}")
                            else:
                                stats['skipped_images'] += 1
                                stats['errors'].append(f"Image {image_name}: Cannot convert TIF to PNG (PIL/Pillow or OpenCV required)")
                                continue
                        except Exception as e:
                            stats['skipped_images'] += 1
                            stats['errors'].append(f"Image {image_name}: Conversion error - {str(e)}")
                            continue
                    
                    # Update reference to PNG file
                    output_image_name = os.path.splitext(image_name)[0] + '.png'
                    image_path = png_path
                
                # Get file size of the final image (PNG if converted, original otherwise)
                file_size = os.path.getsize(image_path)
            else:
                # No images directory - check if filename suggests TIF conversion needed
                if os.path.splitext(image_name)[1].lower() in ['.tif', '.tiff']:
                    output_image_name = os.path.splitext(image_name)[0] + '.png'
            
            # Create VGG image ID (format: filename + filesize)
            # VGG uses filename + filesize as unique ID
            # Use the output image name (PNG if converted, original otherwise)
            vgg_img_id = f"{output_image_name}{file_size}"
            
            # Parse bounding box annotations
            regions = []
            for box_elem in image_elem.findall('box'):
                region = _parse_cvat_box_to_vgg(box_elem, class_name)
                if region:
                    regions.append(region)
                    stats['total_annotations'] += 1
            
            # Skip images without annotations
            if len(regions) == 0:
                stats['skipped_images'] += 1
                continue
            
            # Create VGG image entry (use output_image_name which may be PNG if converted)
            img_entry = {
                "filename": output_image_name,
                "size": file_size,
                "regions": regions,
                "file_attributes": {}
            }
            
            # VGG format uses image ID (filename + size) as the key directly
            vgg_data[vgg_img_id] = img_entry
            stats['converted_images'] += 1
            
        except Exception as e:
            stats['skipped_images'] += 1
            image_id = image_elem.get('id', 'unknown')
            stats['errors'].append(f"Image {image_id}: {str(e)}")
            continue
    
    # Write output JSON file
    os.makedirs(os.path.dirname(output_json_path) if os.path.dirname(output_json_path) else '.', exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(vgg_data, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("CVAT XML to VGG JSON Conversion Statistics")
    print("=" * 80)
    print(f"Input file: {cvat_xml_path}")
    print(f"Output file: {output_json_path}")
    print(f"Total images processed: {stats['total_images']}")
    print(f"Successfully converted: {stats['converted_images']}")
    print(f"Skipped: {stats['skipped_images']}")
    print(f"Total annotations: {stats['total_annotations']}")
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    print("=" * 80)
    print("\n✅ Conversion complete!")
    print("=" * 80)
    
    return stats


def _parse_cvat_box_to_vgg(box_elem: ET.Element, class_name: str) -> Optional[Dict]:
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


def select_files_gui() -> Optional[Tuple[str, str, Optional[str], str]]:
    """
    Show GUI file dialogs to select input XML, output JSON, and optionally images directory.
    
    Returns:
        Tuple of (input_xml_path, output_json_path, images_dir, class_name) or None if cancelled
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog
    except ImportError:
        print("❌ Error: tkinter is not available. Please install it or use command-line arguments.")
        print("   On Linux, install with: sudo apt-get install python3-tk")
        return None
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Select input XML file
    print("📁 Please select the CVAT XML annotation file...")
    input_xml = filedialog.askopenfilename(
        title="Select CVAT XML Annotation File",
        filetypes=[
            ("XML files", "*.xml"),
            ("All files", "*.*")
        ]
    )
    
    if not input_xml:
        print("❌ No input file selected. Cancelled.")
        return None
    
    # Select output JSON file
    print("💾 Please select where to save the VGG JSON file...")
    output_json = filedialog.asksaveasfilename(
        title="Save VGG JSON File As",
        defaultextension=".json",
        filetypes=[
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ],
        initialfile=os.path.splitext(os.path.basename(input_xml))[0] + "_vgg.json"
    )
    
    if not output_json:
        print("❌ No output file selected. Cancelled.")
        return None
    
    # Optionally select images directory
    print("📂 (Optional) Select images directory for file size calculation...")
    images_dir = filedialog.askdirectory(
        title="Select Images Directory (Optional - Cancel to skip)",
        mustexist=False
    )
    
    if not images_dir or not os.path.exists(images_dir):
        images_dir = None
        print("   Skipping images directory selection.")
    
    # Ask for class name
    root.deiconify()  # Show window for dialog
    class_name = simpledialog.askstring(
        "Class Name",
        "Enter the class name (e.g., 'worm'):",
        initialvalue="worm"
    )
    root.withdraw()
    
    if not class_name:
        class_name = "worm"
    
    root.destroy()
    
    return (input_xml, output_json, images_dir, class_name)


def main():
    """Main function for command-line or GUI usage"""
    # Check if running with arguments
    if len(sys.argv) > 1:
        # Command-line mode
        parser = argparse.ArgumentParser(
            description="Convert CVAT XML annotations to VGG JSON format",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Basic conversion
  python cvat_to_vgg.py annotations.xml output.json
  
  # With images directory (for file size calculation)
  python cvat_to_vgg.py annotations.xml output.json --images_dir ./images
  
  # Custom class name
  python cvat_to_vgg.py annotations.xml output.json --class_name "worm"
            """
        )
        
        parser.add_argument(
            "input_xml",
            type=str,
            nargs='?',
            help="Path to CVAT XML annotation file"
        )
        
        parser.add_argument(
            "output_json",
            type=str,
            nargs='?',
            help="Path to output VGG JSON file"
        )
        
        parser.add_argument(
            "--images_dir",
            type=str,
            default=None,
            help="Directory containing images (optional, for file size calculation)"
        )
        
        parser.add_argument(
            "--class_name",
            type=str,
            default="worm",
            help="Name of the class (default: 'worm')"
        )
        
        args = parser.parse_args()
        
        # If required args are missing, fall back to GUI
        if not args.input_xml or not args.output_json:
            print("📋 No command-line arguments provided. Opening GUI...")
            result = select_files_gui()
            if not result:
                return 1
            input_xml, output_json, images_dir, class_name = result
        else:
            input_xml = args.input_xml
            output_json = args.output_json
            images_dir = args.images_dir
            class_name = args.class_name
    else:
        # GUI mode
        print("=" * 80)
        print("CVAT XML to VGG JSON Converter - GUI Mode")
        print("=" * 80)
        result = select_files_gui()
        if not result:
            return 1
        input_xml, output_json, images_dir, class_name = result
    
    # Validate input file
    if not os.path.exists(input_xml):
        print(f"❌ Error: Input file not found: {input_xml}")
        return 1
    
    # Convert
    try:
        stats = convert_cvat_xml_to_vgg_json(
            input_xml,
            output_json,
            images_dir,
            class_name
        )
        
        if stats['converted_images'] > 0:
            print(f"\n✅ Successfully converted {stats['converted_images']} images!")
            return 0
        else:
            print("\n⚠️  Warning: No images were converted!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

