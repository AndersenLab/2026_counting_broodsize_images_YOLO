import os
import json
import random
import shutil
import argparse
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

def prompt_folder_selection(title="Select folder"):
    """Prompt user to select a folder"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folder_path = filedialog.askdirectory(title=title)
    
    root.destroy()
    return folder_path

def prompt_file_selection(title="Select file", filetypes=[("JSON files", "*.json")]):
    """Prompt user to select a file"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    
    root.destroy()
    return file_path

def load_vgg_json_excluded_images(json_path):
    """
    Load VGG JSON file and extract list of image filenames that have annotations
    Only excludes images that actually have objects/regions annotated, not just any image in the JSON
    """
    if not json_path or not os.path.exists(json_path):
        return set()
    
    try:
        with open(json_path, 'r') as f:
            vgg_data = json.load(f)
        
        excluded_images = set()
        total_images_in_json = 0
        
        # Handle different VGG JSON structures
        if isinstance(vgg_data, dict):
            # VIA v3.x format: check for '_via_img_metadata' key first
            if '_via_img_metadata' in vgg_data:
                img_metadata = vgg_data['_via_img_metadata']
                for img_id, img_data in img_metadata.items():
                    total_images_in_json += 1
                    filename = None
                    
                    # Get filename
                    if 'filename' in img_data:
                        filename = img_data['filename']
                    elif 'fname' in img_data:
                        filename = img_data['fname']
                    
                    if filename:
                        # Check if this image has any regions/annotations
                        has_annotations = False
                        
                        # Check for regions in the image data
                        if 'regions' in img_data:
                            regions = img_data['regions']
                            if isinstance(regions, list) and len(regions) > 0:
                                has_annotations = True
                            elif isinstance(regions, dict) and len(regions) > 0:
                                has_annotations = True
                        
                        # Also check for via_region_data (alternative structure)
                        if 'via_region_data' in img_data and img_data['via_region_data']:
                            has_annotations = True
                        
                        if has_annotations:
                            excluded_images.add(filename)
            
            else:
                # VIA v2.x format: keys might be image identifiers
                for key, value in vgg_data.items():
                    if isinstance(value, dict):
                        total_images_in_json += 1
                        filename = None
                        
                        # Get filename
                        if 'filename' in value:
                            filename = value['filename']
                        elif 'fname' in value:
                            filename = value['fname']
                        # Sometimes the key itself is the filename
                        elif isinstance(key, str) and any(ext in key.lower() for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']):
                            filename = key
                        
                        if filename:
                            # Check if this image has any regions/annotations
                            has_annotations = False
                            
                            # Check for regions
                            if 'regions' in value:
                                regions = value['regions']
                                if isinstance(regions, list) and len(regions) > 0:
                                    has_annotations = True
                                elif isinstance(regions, dict) and len(regions) > 0:
                                    has_annotations = True
                            
                            # Check for size > 0 in regions (some formats store empty objects)
                            if 'regions' in value and isinstance(value['regions'], list):
                                non_empty_regions = [r for r in value['regions'] if r and (
                                    ('shape_attributes' in r and r['shape_attributes']) or 
                                    ('region_attributes' in r and r['region_attributes'])
                                )]
                                has_annotations = len(non_empty_regions) > 0
                            
                            if has_annotations:
                                excluded_images.add(filename)
        
        elif isinstance(vgg_data, list):
            # List format: each item might contain filename and regions
            for item in vgg_data:
                if isinstance(item, dict):
                    total_images_in_json += 1
                    filename = None
                    
                    if 'filename' in item:
                        filename = item['filename']
                    elif 'fname' in item:
                        filename = item['fname']
                    
                    if filename:
                        # Check for annotations
                        has_annotations = False
                        if 'regions' in item:
                            regions = item['regions']
                            if isinstance(regions, list) and len(regions) > 0:
                                has_annotations = True
                            elif isinstance(regions, dict) and len(regions) > 0:
                                has_annotations = True
                        
                        if has_annotations:
                            excluded_images.add(filename)
        
        print(f"VGG JSON Analysis:")
        print(f"  Total images in JSON: {total_images_in_json}")
        print(f"  Images with annotations (to exclude): {len(excluded_images)}")
        print(f"  Images without annotations (available): {total_images_in_json - len(excluded_images)}")
        
        if excluded_images:
            print("Sample excluded images with annotations:", list(excluded_images)[:5])
        
        return excluded_images
        
    except Exception as e:
        print(f"Error loading VGG JSON file: {e}")
        return set()

def get_image_files(folder_path):
    """Get list of all image files in the folder"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'}
    image_files = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return image_files
    
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file_name)
            if ext.lower() in image_extensions:
                image_files.append(file_name)
    
    return sorted(image_files)

def select_images_with_spread(available_images, num_to_extract):
    """
    Select images with semi-random spread distribution across the list
    This ensures images are sampled from throughout the folder rather than clustered randomly
    
    Args:
        available_images (list): List of available image filenames (should be sorted)
        num_to_extract (int): Number of images to extract
    
    Returns:
        list: Selected image filenames
    """
    if num_to_extract >= len(available_images):
        return available_images.copy()
    
    if num_to_extract <= 0:
        return []
    
    # Sort images to ensure consistent ordering
    sorted_images = sorted(available_images)
    total_images = len(sorted_images)
    
    # Calculate the ideal stride for even distribution
    stride = total_images / num_to_extract
    
    selected_images = []
    
    # Use semi-random selection with spread
    for i in range(num_to_extract):
        # Calculate the ideal position for this selection
        ideal_pos = i * stride
        
        # Add some randomness around the ideal position (±25% of stride)
        randomness_range = max(1, int(stride * 0.25))
        min_pos = max(0, int(ideal_pos - randomness_range))
        max_pos = min(total_images - 1, int(ideal_pos + randomness_range))
        
        # Randomly select within the allowed range
        actual_pos = random.randint(min_pos, max_pos)
        
        # Ensure we don't select the same image twice
        attempts = 0
        while sorted_images[actual_pos] in selected_images and attempts < 10:
            actual_pos = random.randint(min_pos, max_pos)
            attempts += 1
        
        # If we couldn't find a unique image in the preferred range, 
        # fall back to finding any available image
        if sorted_images[actual_pos] in selected_images:
            for pos in range(total_images):
                if sorted_images[pos] not in selected_images:
                    actual_pos = pos
                    break
        
        selected_images.append(sorted_images[actual_pos])
    
    print(f"Selected images with spread distribution:")
    print(f"  Image positions: {[sorted_images.index(img) for img in selected_images[:10]]}{'...' if len(selected_images) > 10 else ''}")
    print(f"  Coverage: {min([sorted_images.index(img) for img in selected_images])}-{max([sorted_images.index(img) for img in selected_images])} of {total_images-1}")
    
    return selected_images

def extract_image_subset(input_folder, fraction, excluded_images=None, output_folder=None, seed=None):
    """
    Extract a subset of images from input folder
    
    Args:
        input_folder (str): Path to input folder containing images
        fraction (float): Fraction of images to extract (0.0 to 1.0)
        excluded_images (set): Set of image filenames to exclude
        output_folder (str): Path to output folder (if None, will be auto-generated)
        seed (int): Random seed for reproducible selection
    
    Returns:
        str: Path to output folder
    """
    if excluded_images is None:
        excluded_images = set()
    
    # Set random seed for reproducible results
    if seed is not None:
        random.seed(seed)
    
    # Get all image files
    print(f"Scanning for images in: {input_folder}")
    all_images = get_image_files(input_folder)
    print(f"Found {len(all_images)} total images")
    
    if not all_images:
        print("No image files found in the input folder")
        return None
    
    # Filter out excluded images
    available_images = [img for img in all_images if img not in excluded_images]
    excluded_count = len(all_images) - len(available_images)
    
    if excluded_count > 0:
        print(f"Excluded {excluded_count} images based on VGG JSON")
    
    print(f"Available images for selection: {len(available_images)}")
    
    if not available_images:
        print("No images available after exclusion")
        return None
    
    # Calculate number of images to extract
    num_to_extract = max(1, int(len(available_images) * fraction))
    print(f"Extracting {num_to_extract} images ({fraction:.1%} of available)")
    
    # Semi-randomly select images with spread distribution
    selected_images = select_images_with_spread(available_images, num_to_extract)
    
    # Create output folder
    if output_folder is None:
        input_path = Path(input_folder)
        parent_dir = input_path.parent
        folder_name = input_path.name
        output_folder = parent_dir / f"{folder_name}_subset_{int(fraction*100)}pct"
        
        # Add timestamp if folder exists
        counter = 1
        base_output_folder = output_folder
        while output_folder.exists():
            output_folder = Path(str(base_output_folder) + f"_{counter}")
            counter += 1
    
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    
    # Copy selected images
    print("Copying images...")
    copied_count = 0
    failed_count = 0
    
    for img_name in selected_images:
        src_path = os.path.join(input_folder, img_name)
        dst_path = output_folder / img_name
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying {img_name}: {e}")
            failed_count += 1
    
    print(f"\nExtraction complete!")
    print(f"Successfully copied: {copied_count} images")
    if failed_count > 0:
        print(f"Failed to copy: {failed_count} images")
    
    # Save extraction log
    log_path = output_folder / "extraction_log.txt"
    with open(log_path, 'w') as f:
        f.write(f"Image Subset Extraction Log\n")
        f.write(f"{'='*40}\n")
        f.write(f"Input folder: {input_folder}\n")
        f.write(f"Output folder: {output_folder}\n")
        f.write(f"Extraction fraction: {fraction:.1%}\n")
        f.write(f"Random seed: {seed}\n")
        f.write(f"Total images in input: {len(all_images)}\n")
        f.write(f"Excluded images: {excluded_count}\n")
        f.write(f"Available for selection: {len(available_images)}\n")
        f.write(f"Selected for extraction: {len(selected_images)}\n")
        f.write(f"Successfully copied: {copied_count}\n")
        f.write(f"Failed to copy: {failed_count}\n")
        f.write(f"\nSelected images:\n")
        for img in sorted(selected_images):
            f.write(f"  {img}\n")
    
    print(f"Extraction log saved to: {log_path}")
    
    return str(output_folder)

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Extract a subset of images from a folder with semi-random spread distribution")
    parser.add_argument("--input-folder", "-i", type=str, help="Input folder containing images")
    parser.add_argument("--fraction", "-f", type=float, help="Fraction of images to extract (0.0 to 1.0)")
    parser.add_argument("--exclude-json", "-e", type=str, help="VGG JSON file with annotated images to exclude (only excludes images with actual annotations)")
    parser.add_argument("--output-folder", "-o", type=str, help="Output folder (optional)")
    parser.add_argument("--seed", "-s", type=int, help="Random seed for reproducible selection")
    
    args = parser.parse_args()
    
    # Get input folder
    if args.input_folder:
        input_folder = args.input_folder
    else:
        print("Select input folder containing images:")
        input_folder = prompt_folder_selection("Select input folder with images")
        if not input_folder:
            print("No folder selected. Exiting.")
            return
    
    # Get fraction
    if args.fraction:
        fraction = args.fraction
    else:
        while True:
            try:
                fraction_input = input("Enter fraction of images to extract (e.g., 0.1 for 10%): ")
                fraction = float(fraction_input)
                if 0.0 < fraction <= 1.0:
                    break
                else:
                    print("Fraction must be between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")
    
    # Get exclusion file
    excluded_images = set()
    if args.exclude_json:
        exclude_json = args.exclude_json
    else:
        use_exclusion = input("Do you want to exclude images that have annotations in a VGG JSON file? (y/n): ").lower().strip()
        if use_exclusion in ['y', 'yes']:
            print("Select VGG JSON file (only images with annotations will be excluded):")
            exclude_json = prompt_file_selection("Select VGG JSON file", [("JSON files", "*.json")])
        else:
            exclude_json = None
    
    if exclude_json:
        excluded_images = load_vgg_json_excluded_images(exclude_json)
    
    # Extract subset
    print("\nStarting image extraction...")
    output_folder = extract_image_subset(
        input_folder=input_folder,
        fraction=fraction,
        excluded_images=excluded_images,
        output_folder=args.output_folder,
        seed=args.seed
    )
    
    if output_folder:
        print(f"\n✅ Image subset extraction completed successfully!")
        print(f"Output saved to: {output_folder}")
    else:
        print("\n❌ Image subset extraction failed!")

if __name__ == "__main__":
    main() 