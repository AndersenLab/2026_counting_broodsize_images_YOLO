#!/usr/bin/env python3
"""
Generate Training Dataset for Worm Classifier

This script extracts small crops from DatasetImages using DatasetMasks for labeling,
creating a balanced dataset for training a worm vs background classifier.

Usage:
    python generate_worm_classifier_dataset.py --dataset_dir /path/to/dataset --output_dir /path/to/output

Author: AI Assistant
Date: 2024
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog

# Add current directory to path to import WormClassifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from WormClassifier import WormClassifier, get_default_config
except ImportError as e:
    print(f"❌ Error importing WormClassifier: {e}")
    print("Make sure WormClassifier.py is in the same directory as this script.")
    sys.exit(1)

try:
    from Scripts.file_cache import file_cache
except ImportError as e:
    print(f"❌ Error importing file_cache: {e}")
    print("Make sure Scripts/file_cache.py exists.")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate training dataset for worm classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # GUI mode (default) - will prompt for folder selection
    python generate_worm_classifier_dataset.py
    
    # Command line mode with custom parameters
    python generate_worm_classifier_dataset.py \\
        --dataset_dir ./dataset \\
        --output_dir ./classifier_data \\
        --window_size 64 \\
        --samples_per_image 200 \\
        --min_worm_pixels 75 \\
        --max_attempts 1000
    
    # Different balance ratio (more background samples)
    python generate_worm_classifier_dataset.py \\
        --dataset_dir ./dataset \\
        --output_dir ./classifier_data \\
        --balance_ratio 0.5
        """
    )
    
    # Optional arguments (GUI mode is default)
    parser.add_argument(
        '--dataset_dir', 
        type=str, 
        help='Parent directory containing DatasetImages and DatasetMasks folders (GUI mode if not specified)'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        help='Output directory for generated classifier dataset (same as dataset_dir if not specified)'
    )
    
    # Optional extraction parameters
    parser.add_argument(
        '--window_size', 
        type=int, 
        default=24,
        help='Size of extracted crop windows (default: 64)'
    )
    
    parser.add_argument(
        '--samples_per_image', 
        type=int, 
        default=20,
        help='Number of crops to extract per image (default: 100)'
    )
    
    parser.add_argument(
        '--min_worm_pixels', 
        type=int, 
        default=50,
        help='Minimum worm pixels to classify as worm (default: 50)'
    )
    
    parser.add_argument(
        '--balance_ratio', 
        type=float, 
        default=1.0,
        help='Target ratio of worm:background samples (default: 1.0)'
    )
    
    parser.add_argument(
        '--max_attempts', 
        type=int, 
        default=500,
        help='Maximum attempts to find valid crops per image (default: 500)'
    )
    
    parser.add_argument(
        '--random_seed', 
        type=int, 
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Output options
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Show what would be done without actually generating files'
    )
    
    return parser.parse_args()


def validate_dataset_structure(dataset_dir):
    """Validate that the dataset directory has the expected structure"""
    images_dir = os.path.join(dataset_dir, "DatasetImages")
    masks_dir = os.path.join(dataset_dir, "DatasetMasks")
    
    # Check if directories exist
    if not os.path.exists(images_dir):
        raise ValueError(f"DatasetImages directory not found: {images_dir}")
    if not os.path.exists(masks_dir):
        raise ValueError(f"DatasetMasks directory not found: {masks_dir}")
    
    # Check if directories contain files
    image_files = [f for f in os.listdir(images_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    mask_files = [f for f in os.listdir(masks_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    if len(mask_files) == 0:
        raise ValueError(f"No mask files found in {masks_dir}")
    
    print(f"✅ Dataset validation passed:")
    print(f"   Images: {len(image_files)} files in {images_dir}")
    print(f"   Masks: {len(mask_files)} files in {masks_dir}")
    
    return images_dir, masks_dir, image_files, mask_files


def estimate_dataset_size(image_files, samples_per_image, balance_ratio):
    """Estimate the final dataset size"""
    total_images = len(image_files)
    total_crops = total_images * samples_per_image
    
    # Estimate class distribution (assuming roughly equal)
    estimated_worm = total_crops // 2
    estimated_background = total_crops // 2
    
    # Apply balance ratio
    if balance_ratio < 1.0:
        # More background samples
        final_worm = estimated_worm
        final_background = int(estimated_worm / balance_ratio)
    else:
        # More worm samples or equal
        final_background = estimated_background
        final_worm = int(estimated_background * balance_ratio)
    
    return {
        'total_images': total_images,
        'estimated_total_crops': total_crops,
        'estimated_worm_samples': final_worm,
        'estimated_background_samples': final_background,
        'estimated_total_samples': final_worm + final_background
    }


def select_dataset_folder():
    """Select dataset folder using GUI with cache support"""
    script_name = os.path.basename(__file__)
    
    # Try to load cached folder path
    cached_paths = file_cache.load_file_paths(script_name)
    
    if cached_paths and 'dataset_folder' in cached_paths:
        # Show cache option to user
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        cached_folder = cached_paths['dataset_folder']
        folder_name = os.path.basename(cached_folder)
        
        use_cache = messagebox.askyesno(
            "Use Cached Folder?",
            f"Found previously selected folder:\n\n{folder_name}\n\nWould you like to use this folder again?",
            parent=root
        )
        
        root.destroy()
        
        if use_cache:
            return cached_folder
    
    # Select new folder
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    folder_path = filedialog.askdirectory(
        title="Select Dataset Folder (containing DatasetImages and DatasetMasks)"
    )
    
    root.destroy()
    
    if not folder_path:
        print("❌ No folder selected. Exiting.")
        sys.exit(1)
    
    # Save to cache
    file_cache.save_file_paths(script_name, [folder_path], ['dataset_folder'])
    
    return folder_path


def main():
    """Main execution function"""
    print("🪱 Worm Classifier Dataset Generator")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine dataset and output directories
    if args.dataset_dir:
        # Command line mode
        dataset_dir = args.dataset_dir
        output_dir = args.output_dir if args.output_dir else dataset_dir
        print(f"📁 Command line mode:")
        print(f"   Dataset directory: {dataset_dir}")
        print(f"   Output directory: {output_dir}")
    else:
        # GUI mode
        print(f"📁 GUI mode - selecting folder...")
        dataset_dir = select_dataset_folder()
        output_dir = dataset_dir  # Use same folder for output
        print(f"   Selected folder: {dataset_dir}")
        print(f"   Output will be saved to: {output_dir}")
    
    # Validate dataset structure
    try:
        images_dir, masks_dir, image_files, mask_files = validate_dataset_structure(dataset_dir)
    except ValueError as e:
        print(f"❌ Dataset validation failed: {e}")
        sys.exit(1)
    
    # Estimate dataset size
    size_estimate = estimate_dataset_size(image_files, args.samples_per_image, args.balance_ratio)
    
    print(f"\n📊 Dataset Generation Plan:")
    print(f"   Window size: {args.window_size}x{args.window_size}")
    print(f"   Samples per image: {args.samples_per_image}")
    print(f"   Min worm pixels: {args.min_worm_pixels}")
    print(f"   Balance ratio (worm:background): {args.balance_ratio}")
    print(f"   Max attempts per image: {args.max_attempts}")
    print(f"   Random seed: {args.random_seed}")
    print(f"   Estimated final dataset: {size_estimate['estimated_total_samples']} samples")
    print(f"     - Worm: ~{size_estimate['estimated_worm_samples']}")
    print(f"     - Background: ~{size_estimate['estimated_background_samples']}")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No files will be generated")
        print(f"   Output directory: {output_dir}")
        print(f"   Would create: {output_dir}/Classifier/Worm/")
        print(f"   Would create: {output_dir}/Classifier/Background/")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure classifier
    config = get_default_config(window_size=args.window_size)
    
    # Initialize classifier
    print(f"\n🔧 Initializing WormClassifier...")
    classifier = WormClassifier(config)
    
    # Prepare extraction configuration
    extraction_config = {
        'samples_per_image': args.samples_per_image,
        'min_worm_pixels': args.min_worm_pixels,
        'target_balance_ratio': args.balance_ratio,
        'max_attempts_per_image': args.max_attempts,
        'random_seed': args.random_seed
    }
    
    # Generate dataset
    print(f"\n🚀 Starting dataset generation...")
    start_time = datetime.now()
    
    try:
        stats = classifier.generate_training_dataset(
            dataset_parent_dir=dataset_dir,
            output_dir=output_dir,
            extraction_config=extraction_config
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Dataset generation completed successfully!")
        print(f"   Duration: {duration}")
        print(f"   Total crops extracted: {stats['total_crops_extracted']}")
        print(f"   Worm crops: {stats['worm_crops']}")
        print(f"   Background crops: {stats['background_crops']}")
        print(f"   Images processed: {stats['images_processed']}")
        print(f"   Extraction failures: {stats['extraction_failures']}")
        
        # Save summary report
        summary_file = os.path.join(output_dir, "generation_summary.json")
        summary = {
            'command_line_args': vars(args),
            'extraction_config': extraction_config,
            'statistics': stats,
            'size_estimate': size_estimate,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   Summary saved to: {summary_file}")
        
        # Show next steps
        print(f"\n📋 Next Steps:")
        print(f"   1. Review generated crops in: {output_dir}/Classifier/")
        print(f"   2. Train classifier using: train_worm_classifier.py")
        print(f"   3. Or use the classifier directly in your analysis pipeline")
        
    except Exception as e:
        print(f"\n❌ Dataset generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 