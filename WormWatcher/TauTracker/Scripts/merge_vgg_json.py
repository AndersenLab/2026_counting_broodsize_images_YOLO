# merge_vgg_json()
#
# Merges multiple VGG JSON files (from myvision.ai labeling) into a single JSON file
# for use with RCNN_json2mask.py. Handles duplicate images by keeping the last
# labeled version and tracking all duplicates for reporting.
#
# Features:
# - UI for selecting multiple directories containing JSON files
# - Outputs merged JSON and duplicate report to the same folder as input
# - Prevents file conflicts by adding timestamps to output files
#
# Input: Multiple JSON files in user-selected directories
# Output: Single merged JSON file with duplicate report in the same folder

#####   LIBRARIES   #####

import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from collections import defaultdict
from datetime import datetime

#####   UI FUNCTIONS   #####

def select_directories():
    """Open UI to select multiple directories containing JSON files."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    print("Please select directories containing JSON files to merge...")
    print("(You can select multiple directories one by one)")
    
    directories = []
    
    while True:
        # Ask for a directory
        directory = filedialog.askdirectory(
            title="Select a directory containing JSON files (Cancel to finish selection)"
        )
        
        if not directory:
            break
            
        directories.append(directory)
        print(f"Added: {directory}")
        
        # Ask if user wants to add more
        response = messagebox.askyesno(
            "Add More Directories", 
            f"Added {len(directories)} directory(ies) so far.\n\nAdd another directory?"
        )
        
        if not response:
            break
    
    if not directories:
        print("No directories selected. Exiting.")
        return None
    
    print(f"\nSelected {len(directories)} directories:")
    for i, dir_path in enumerate(directories, 1):
        print(f"  {i}. {dir_path}")
    
    return directories

def get_output_paths(base_dir):
    """Generate output file paths with timestamps to prevent conflicts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_json = os.path.join(base_dir, f"merged_labels_{timestamp}.json")
    duplicate_report = os.path.join(base_dir, f"duplicate_report_{timestamp}.txt")
    
    return output_json, duplicate_report

#####   MAIN MERGING LOGIC   #####

def merge_vgg_json_files(json_dirs, output_json, duplicate_report=None):
    """
    Merge multiple VGG JSON files into a single file.
    
    Args:
        json_dirs: List of directories containing JSON files
        output_json: Output path for merged JSON file
        duplicate_report: Optional path for duplicate report file
    """
    
    merged_data = {}
    duplicate_tracker = defaultdict(list)  # filename -> list of source files
    processed_files = []
    
    print("Starting VGG JSON merge process...")
    print(f"Processing {len(json_dirs)} directories")
    
    # Process each directory
    for dir_idx, json_dir in enumerate(json_dirs):
        if not os.path.exists(json_dir):
            print(f"WARNING: Directory does not exist: {json_dir}")
            continue
            
        print(f"\nProcessing directory {dir_idx + 1}/{len(json_dirs)}: {json_dir}")
        
        # Find all JSON files in the directory
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
        
        if not json_files:
            print(f"  No JSON files found in {json_dir}")
            continue
            
        print(f"  Found {len(json_files)} JSON files")
        
        # Process each JSON file in the directory
        for json_file in json_files:
            json_path = os.path.join(json_dir, json_file)
            print(f"    Processing: {json_file}")
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                processed_files.append(json_path)
                
                # Merge data from this JSON file
                for image_key, image_data in data.items():
                    filename = image_data.get('filename', 'unknown')
                    
                    # Check if this image already exists in merged data
                    if image_key in merged_data:
                        # Track duplicate
                        duplicate_tracker[filename].append({
                            'source_file': json_path,
                            'image_key': image_key,
                            'regions_count': len(image_data.get('regions', []))
                        })
                        print(f"      DUPLICATE: {filename} (overwriting previous version)")
                    else:
                        # Track first occurrence
                        duplicate_tracker[filename].append({
                            'source_file': json_path,
                            'image_key': image_key,
                            'regions_count': len(image_data.get('regions', []))
                        })
                    
                    # Always update with latest version (overwrites if duplicate)
                    merged_data[image_key] = image_data
                    
            except Exception as e:
                print(f"    ERROR processing {json_file}: {str(e)}")
                continue
    
    # Write merged JSON file
    print(f"\nWriting merged JSON to: {output_json}")
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    with open(output_json, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    # Generate duplicate report
    if duplicate_report:
        print(f"Generating duplicate report: {duplicate_report}")
        generate_duplicate_report(duplicate_tracker, duplicate_report, processed_files)
    
    # Print summary
    print(f"\n=== MERGE SUMMARY ===")
    print(f"Total images processed: {len(merged_data)}")
    print(f"Total JSON files processed: {len(processed_files)}")
    
    duplicates_found = sum(1 for filename, occurrences in duplicate_tracker.items() if len(occurrences) > 1)
    print(f"Images with duplicates: {duplicates_found}")
    
    if duplicates_found > 0:
        print(f"\nDuplicate details:")
        for filename, occurrences in duplicate_tracker.items():
            if len(occurrences) > 1:
                print(f"  {filename}: {len(occurrences)} duplicates")
                for i, occ in enumerate(occurrences):
                    status = "KEPT" if i == len(occurrences) - 1 else "OVERWRITTEN"
                    print(f"    {status}: {os.path.basename(occ['source_file'])} ({occ['regions_count']} regions)")
    
    print(f"\nMerged JSON saved to: {output_json}")
    if duplicate_report:
        print(f"Duplicate report saved to: {duplicate_report}")

def generate_duplicate_report(duplicate_tracker, report_path, processed_files):
    """Generate a detailed duplicate report file."""
    
    with open(report_path, 'w') as f:
        f.write("VGG JSON MERGE - DUPLICATE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated on: {os.path.basename(__file__)}\n")
        f.write(f"Total JSON files processed: {len(processed_files)}\n\n")
        
        f.write("PROCESSED FILES:\n")
        f.write("-" * 20 + "\n")
        for i, file_path in enumerate(processed_files, 1):
            f.write(f"{i:3d}. {file_path}\n")
        
        f.write(f"\nDUPLICATE ANALYSIS:\n")
        f.write("-" * 20 + "\n")
        
        duplicates_found = 0
        total_images = len(duplicate_tracker)
        
        for filename, occurrences in duplicate_tracker.items():
            if len(occurrences) > 1:
                duplicates_found += 1
                f.write(f"\n{filename}:\n")
                f.write(f"  N duplicates: {len(occurrences)}\n")
                
                for i, occ in enumerate(occurrences):
                    status = "KEPT (last version)" if i == len(occurrences) - 1 else "OVERWRITTEN"
                    f.write(f"  {status}: {os.path.basename(occ['source_file'])} ({occ['regions_count']} regions)\n")
        
        f.write(f"\nSUMMARY:\n")
        f.write("-" * 10 + "\n")
        f.write(f"Total images: {total_images}\n")
        f.write(f"Images with duplicates: {duplicates_found}\n")
        f.write(f"Duplicate rate: {duplicates_found/total_images*100:.1f}%\n")

#####   EXECUTION   #####

if __name__ == "__main__":
    # Select directories using UI
    json_dirs = select_directories()
    
    if json_dirs is None:
        exit()
    
    # Determine output directory (use first selected directory as base)
    base_dir = json_dirs[0]
    print(f"\nOutput files will be saved to: {base_dir}")
    
    # Generate output paths with timestamps to prevent conflicts
    output_json, duplicate_report = get_output_paths(base_dir)
    
    print(f"Output JSON: {os.path.basename(output_json)}")
    print(f"Duplicate report: {os.path.basename(duplicate_report)}")
    
    # Confirm before proceeding
    response = input("\nProceed with merge? (y/n): ").lower().strip()
    if response not in ['y', 'yes']:
        print("Merge cancelled.")
        exit()
    
    # Run the merge
    merge_vgg_json_files(json_dirs, output_json, duplicate_report)
