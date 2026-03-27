#!/usr/bin/env python3
"""
Combine Worm Classifier Datasets

This script combines multiple worm classifier datasets into a single combined dataset.
It uses all samples from the smallest dataset and proportionally samples from others
based on specified fractions.

Usage:
    python combine_worm_classifier_datasets.py

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import shutil
from datetime import datetime
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION - Edit these values
# ============================================================================

# List of dataset entries, where each entry is [folder, fraction]
# - folder: Path to a single dataset folder (must contain Classifier/Worm and Classifier/Background)
# - fraction: Relative fraction determining how much this dataset contributes relative to the smallest one
# Example:
#   DATASET_ENTRIES = [
#       ["path/to/dataset1", 1.0],   # Single folder with fraction 1.0
#       ["path/to/dataset2", 0.5],   # Single folder with fraction 0.5
#       ["path/to/dataset3", 0.75],  # Single folder with fraction 0.75
#   ]
DATASET_ENTRIES = [
    [r"I:\Tau worm mask RCNN\Model 4c - combinaed 4a+4b\Augmented", 0.72],
    [r"E:\Nikita JHU Plates\Validation crops marked TP-TN-FP-FN", 0.28],
]

# Output folder for combined dataset
OUTPUT_FOLDER = r"I:\Tau worm mask RCNN\Model 4e - combined 4d and 4c classifier"

# Random seed for reproducibility
# This seed ensures that random sampling produces the same results each time you run the script.
# If you want different random samples each run, change this value or set it to None.
# Same seed = same random samples selected (reproducible)
# Different seed = different random samples selected
# None = uses system time (non-reproducible, different each run)
RANDOM_SEED = 42

# ============================================================================
# END CONFIGURATION
# ============================================================================


def validate_dataset_folder(folder_path):
    """
    Validate that a dataset folder has the expected structure
    
    Returns:
        tuple: (worm_dir, background_dir, worm_count, background_count) or None if invalid
    """
    folder_path = Path(folder_path)
    worm_dir = folder_path / "Classifier" / "Worm"
    background_dir = folder_path / "Classifier" / "Background"
    
    if not worm_dir.exists():
        print(f"❌ Warning: {worm_dir} not found")
        return None
    if not background_dir.exists():
        print(f"❌ Warning: {background_dir} not found")
        return None
    
    # Count image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
    worm_files = [f for f in worm_dir.iterdir() 
                  if f.suffix.lower() in image_extensions]
    bg_files = [f for f in background_dir.iterdir() 
                if f.suffix.lower() in image_extensions]
    
    return worm_dir, background_dir, len(worm_files), len(bg_files)




def get_all_image_files(directory):
    """Get all image files from a directory"""
    image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
    return [f for f in directory.iterdir() 
            if f.suffix.lower() in image_extensions]


def copy_file_with_unique_name(source_file, dest_dir, prefix="", counter=0):
    """
    Copy a file to destination directory with a unique name
    
    Returns:
        tuple: (new_filepath, counter) where counter is updated
    """
    source_path = Path(source_file)
    dest_path = Path(dest_dir)
    
    # Create unique filename
    base_name = source_path.stem
    extension = source_path.suffix
    
    # Try original name first
    if counter == 0:
        new_name = f"{prefix}{base_name}{extension}"
    else:
        new_name = f"{prefix}{base_name}_{counter:04d}{extension}"
    
    new_path = dest_path / new_name
    
    # If file exists, increment counter
    if new_path.exists():
        counter += 1
        return copy_file_with_unique_name(source_file, dest_dir, prefix, counter)
    
    # Copy file
    shutil.copy2(source_file, new_path)
    return new_path, counter


def main():
    """Main execution function"""
    print("🔀 Worm Classifier Dataset Combiner")
    print("=" * 50)
    
    # Validate configuration
    if len(DATASET_ENTRIES) == 0:
        print("❌ Error: No dataset entries specified in DATASET_ENTRIES")
        print("   Please edit the script and add dataset entries at the top")
        print("   Format: [folder(s), fraction]")
        sys.exit(1)
    
    # Validate all dataset entries
    print(f"\n📁 Validating {len(DATASET_ENTRIES)} dataset entries...")
    dataset_info = []
    
    for i, entry in enumerate(DATASET_ENTRIES):
        # Validate entry format
        if not isinstance(entry, list) or len(entry) != 2:
            print(f"❌ Error: Entry {i+1} must be [folder, fraction], got: {entry}")
            sys.exit(1)
        
        folder_path, fraction = entry
        
        # Validate folder is a string
        if not isinstance(folder_path, str):
            print(f"❌ Error: Entry {i+1} folder must be a string path, got: {type(folder_path)}")
            sys.exit(1)
        
        # Validate fraction
        if not isinstance(fraction, (int, float)) or fraction <= 0:
            print(f"❌ Error: Entry {i+1} fraction must be a positive number, got: {fraction}")
            sys.exit(1)
        
        print(f"\n   Dataset Entry {i+1}: {folder_path} (fraction: {fraction})")
        
        # Validate folder structure
        result = validate_dataset_folder(folder_path)
        if result is None:
            print(f"❌ Error: Invalid dataset folder: {folder_path}")
            sys.exit(1)
        
        worm_dir, bg_dir, worm_count, bg_count = result
        total_count = worm_count + bg_count
        
        dataset_info.append({
            'folder': folder_path,
            'worm_dir': worm_dir,
            'background_dir': bg_dir,
            'worm_count': worm_count,
            'background_count': bg_count,
            'total_count': total_count,
            'fraction': fraction
        })
        
        print(f"      ✅ Worm: {worm_count}, Background: {bg_count}, Total: {total_count}")
    
    # Validate that fractions sum to approximately 1.0
    total_fraction = sum(info['fraction'] for info in dataset_info)
    if abs(total_fraction - 1.0) > 0.01:  # Allow small floating point errors
        print(f"\n❌ Error: Fractions must sum to 1.0, but they sum to {total_fraction:.3f}")
        print(f"   Please adjust fractions so they add up to 1.0")
        for i, info in enumerate(dataset_info):
            print(f"   Dataset {i+1}: {info['fraction']}")
        sys.exit(1)
    
    # Find smallest dataset (this one will use all samples)
    smallest_dataset = min(dataset_info, key=lambda x: x['total_count'])
    smallest_total = smallest_dataset['total_count']
    smallest_index = dataset_info.index(smallest_dataset)
    smallest_fraction = smallest_dataset['fraction']
    
    # Calculate total dataset size based on smallest dataset
    # If smallest has fraction F and total T, then: T / F = total_combined
    total_combined = int(smallest_total / smallest_fraction)
    
    print(f"\n📊 Dataset Analysis:")
    print(f"   Smallest dataset: Entry {smallest_index + 1} - {Path(smallest_dataset['folder']).name}")
    print(f"   Smallest total samples: {smallest_total}")
    print(f"   Smallest fraction: {smallest_fraction:.3f}")
    print(f"   Calculated total combined dataset: {smallest_total} / {smallest_fraction:.3f} = {total_combined}")
    print(f"   This dataset will use ALL {smallest_total} samples ({smallest_fraction:.1%} of total)")
    
    # Calculate target sample counts for each dataset based on fractions
    print(f"\n📐 Calculating proportional sampling...")
    target_counts = []
    
    for i, info in enumerate(dataset_info):
        if info == smallest_dataset:
            # Use all samples from smallest dataset
            target_worm = info['worm_count']
            target_bg = info['background_count']
            target_total = info['total_count']
            use_all = True
        else:
            # Calculate target based on fraction of total combined dataset
            # Target = total_combined * fraction
            target_total = int(total_combined * info['fraction'])
            
            # Distribute proportionally between worm and background
            if info['total_count'] > 0:
                worm_ratio = info['worm_count'] / info['total_count']
                bg_ratio = info['background_count'] / info['total_count']
            else:
                worm_ratio = 0.5
                bg_ratio = 0.5
            
            target_worm = int(target_total * worm_ratio)
            target_bg = target_total - target_worm  # Ensure they sum to target_total
            use_all = False
        
        target_counts.append({
            'target_worm': target_worm,
            'target_bg': target_bg,
            'target_total': target_total,
            'use_all': use_all
        })
        
        if use_all:
            print(f"   Dataset {i+1}: Use ALL ({target_worm} worm, {target_bg} background) = {target_total} total ({info['fraction']:.1%} of {total_combined})")
        else:
            print(f"   Dataset {i+1}: Sample {target_worm} worm, {target_bg} background = {target_total} total ({info['fraction']:.1%} of {total_combined})")
            print(f"      Calculation: {total_combined} × {info['fraction']:.3f} = {target_total}")
            print(f"      Available: {info['worm_count']} worm, {info['background_count']} background")
    
    print(f"\n   Expected total combined dataset: {total_combined} samples")
    print(f"   Sum of targets: {sum(t['target_total'] for t in target_counts)}")
    
    # Create output directory structure
    output_path = Path(OUTPUT_FOLDER)
    output_worm_dir = output_path / "Classifier" / "Worm"
    output_bg_dir = output_path / "Classifier" / "Background"
    
    os.makedirs(output_worm_dir, exist_ok=True)
    os.makedirs(output_bg_dir, exist_ok=True)
    
    print(f"\n📦 Creating combined dataset in: {output_path}")
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Process each dataset
    total_copied_worm = 0
    total_copied_bg = 0
    
    for i, (info, targets) in enumerate(zip(dataset_info, target_counts)):
        display_name = Path(info['folder']).name
        print(f"\n📂 Processing dataset {i+1}: {display_name}")
        
        # Get all image files
        worm_files = get_all_image_files(info['worm_dir'])
        bg_files = get_all_image_files(info['background_dir'])
        
        # Process worm files
        if targets['use_all']:
            # Use all worm files
            selected_worm = worm_files
            print(f"   Worm: Using all {len(selected_worm)} files")
        else:
            # Randomly sample
            if len(worm_files) > targets['target_worm']:
                selected_indices = np.random.choice(len(worm_files), targets['target_worm'], replace=False)
                selected_worm = [worm_files[idx] for idx in selected_indices]
                print(f"   Worm: Randomly selected {len(selected_worm)} from {len(worm_files)}")
            else:
                selected_worm = worm_files
                print(f"   Worm: Using all {len(selected_worm)} (less than target)")
        
        # Process background files
        if targets['use_all']:
            # Use all background files
            selected_bg = bg_files
            print(f"   Background: Using all {len(selected_bg)} files")
        else:
            # Randomly sample
            if len(bg_files) > targets['target_bg']:
                selected_indices = np.random.choice(len(bg_files), targets['target_bg'], replace=False)
                selected_bg = [bg_files[idx] for idx in selected_indices]
                print(f"   Background: Randomly selected {len(selected_bg)} from {len(bg_files)}")
            else:
                selected_bg = bg_files
                print(f"   Background: Using all {len(selected_bg)} (less than target)")
        
        # Copy worm files
        print(f"   Copying {len(selected_worm)} worm files...")
        prefix = f"ds{i+1}_"
        counter = 0
        for source_file in selected_worm:
            _, counter = copy_file_with_unique_name(source_file, output_worm_dir, prefix, counter)
            total_copied_worm += 1
        
        # Copy background files
        print(f"   Copying {len(selected_bg)} background files...")
        prefix = f"ds{i+1}_"
        counter = 0
        for source_file in selected_bg:
            _, counter = copy_file_with_unique_name(source_file, output_bg_dir, prefix, counter)
            total_copied_bg += 1
    
    # Create summary
    print(f"\n✅ Dataset combination complete!")
    print(f"   Total worm samples: {total_copied_worm}")
    print(f"   Total background samples: {total_copied_bg}")
    print(f"   Total samples: {total_copied_worm + total_copied_bg}")
    print(f"   Output directory: {output_path}")
    
    # Save combination summary
    summary_file = output_path / "combination_summary.json"
    summary = {
        'datasets': [
            {
                'folder': info['folder'],
                'original_worm': info['worm_count'],
                'original_background': info['background_count'],
                'original_total': info['total_count'],
                'fraction': info['fraction'],
                'copied_worm': targets['target_worm'] if not targets['use_all'] else info['worm_count'],
                'copied_background': targets['target_bg'] if not targets['use_all'] else info['background_count'],
                'used_all': targets['use_all']
            }
            for info, targets in zip(dataset_info, target_counts)
        ],
        'output_folder': str(output_path),
        'total_worm_samples': total_copied_worm,
        'total_background_samples': total_copied_bg,
        'total_samples': total_copied_worm + total_copied_bg,
        'random_seed': RANDOM_SEED,
        'combination_timestamp': datetime.now().isoformat()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   Summary saved to: {summary_file}")
    
    print(f"\n📋 Next Steps:")
    print(f"   1. Review combined dataset in: {output_path}")
    print(f"   2. Train classifier using: train_worm_classifier.py")
    print(f"   3. Dataset structure: {output_path}/Classifier/Worm/ and {output_path}/Classifier/Background/")


if __name__ == "__main__":
    main()

