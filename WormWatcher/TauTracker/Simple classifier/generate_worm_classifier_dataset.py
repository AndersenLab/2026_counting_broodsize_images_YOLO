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
import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# Add current directory and parent directory to path to import WormClassifier
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

try:
    from WormClassifier import WormClassifier, get_default_config
except ImportError as e:
    print(f"❌ Error importing WormClassifier: {e}")
    print("Make sure WormClassifier.py is in the same directory or parent directory of this script.")
    print(f"   Checked: {current_dir}")
    print(f"   Checked: {parent_dir}")
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
        default=40,
        help='Size of extracted crop windows (default: 40)'
    )
    
    parser.add_argument(
        '--samples_per_image', 
        type=int, 
        default=35,
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


def select_cropped_folder():
    """Select folder containing TP/TN/FP/FN subfolders using GUI with cache support"""
    script_name = os.path.basename(__file__)
    
    # Try to load cached folder path
    cached_paths = file_cache.load_file_paths(script_name)
    
    if cached_paths and 'cropped_folder' in cached_paths:
        # Show cache option to user
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        cached_folder = cached_paths['cropped_folder']
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
        title="Select Folder (containing True Positive, True Negative, False Positive, False Negative subfolders)"
    )
    
    root.destroy()
    
    if not folder_path:
        print("❌ No folder selected. Exiting.")
        sys.exit(1)
    
    # Save to cache
    file_cache.save_file_paths(script_name, [folder_path], ['cropped_folder'])
    
    return folder_path


def validate_cropped_structure(cropped_dir):
    """Validate that the cropped directory has TP/TN/FP/FN subfolders"""
    cropped_path = Path(cropped_dir)
    
    tp_dir = cropped_path / "True Positive"
    tn_dir = cropped_path / "True Negative"
    fp_dir = cropped_path / "False Positive"
    fn_dir = cropped_path / "False Negative"
    
    # Check if at least one folder exists
    if not any([tp_dir.exists(), tn_dir.exists(), fp_dir.exists(), fn_dir.exists()]):
        raise ValueError(
            f"Expected at least one of the following folders:\n"
            f"  - True Positive\n"
            f"  - True Negative\n"
            f"  - False Positive\n"
            f"  - False Negative\n"
            f"Found in: {cropped_dir}"
        )
    
    # Count files in each folder
    counts = {}
    for folder_name, folder_path in [("True Positive", tp_dir), ("True Negative", tn_dir),
                                     ("False Positive", fp_dir), ("False Negative", fn_dir)]:
        if folder_path.exists():
            files = [f for f in folder_path.iterdir() 
                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']]
            counts[folder_name] = len(files)
        else:
            counts[folder_name] = 0
    
    total_files = sum(counts.values())
    if total_files == 0:
        raise ValueError(f"No image files found in any of the TP/TN/FP/FN folders in {cropped_dir}")
    
    print(f"✅ Cropped folder validation passed:")
    for folder_name, count in counts.items():
        if count > 0:
            print(f"   {folder_name}: {count} files")
    
    return counts


def generate_from_cropped_files(cropped_dir, output_dir, window_size, random_seed=42, augmentations_per_image=5):
    """
    Generate training dataset from already cropped files in TP/TN/FP/FN folders
    
    Args:
        cropped_dir: Directory containing TP/TN/FP/FN subfolders
        output_dir: Output directory for Classifier/Worm and Classifier/Background
        window_size: Target window size for resizing
        random_seed: Random seed for reproducibility
        augmentations_per_image: Number of augmented versions to create per original image
    
    Returns:
        dict: Statistics about the generated dataset
    """
    np.random.seed(random_seed)
    
    cropped_path = Path(cropped_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    worm_output_dir = output_path / "Classifier" / "Worm"
    background_output_dir = output_path / "Classifier" / "Background"
    os.makedirs(worm_output_dir, exist_ok=True)
    os.makedirs(background_output_dir, exist_ok=True)
    
    # Define augmentation transforms (preserving worm smoothness)
    # Note: Avoiding rotation/shearing transforms that can distort worm smoothness
    # Using only flips and minimal color adjustments to preserve image quality
    # Note: ToPILImage() converts numpy array to PIL Image, then all subsequent
    # transforms work with PIL Images and return PIL Images
    augmentation_transforms = [
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((window_size, window_size), interpolation=Image.LANCZOS)  # High-quality resize
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((window_size, window_size), interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(1.0)  # Always flip
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((window_size, window_size), interpolation=Image.LANCZOS),
            transforms.RandomVerticalFlip(1.0)  # Always flip
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((window_size, window_size), interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(1.0),
            transforms.RandomVerticalFlip(1.0)  # Both flips
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((window_size, window_size), interpolation=Image.LANCZOS),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)  # Reduced to minimize blurring
        ])
    ]
    
    # Base transform (no augmentation, just resize with high-quality interpolation)
    base_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((window_size, window_size), interpolation=Image.LANCZOS)  # High-quality resize
    ])
    
    # Map folders to labels
    # True Positive → worm (1)
    # True Negative → background (0)
    # False Positive → background (0) - incorrectly identified as worm, but actually background
    # False Negative → worm (1) - incorrectly identified as background, but actually worm
    
    folder_mapping = {
        "True Positive": (worm_output_dir, 1),
        "True Negative": (background_output_dir, 0),
        "False Positive": (background_output_dir, 0),
        "False Negative": (worm_output_dir, 1)
    }
    
    stats = {
        'total_samples': 0,
        'worm_samples': 0,
        'background_samples': 0,
        'augmentations_per_image': augmentations_per_image,
        'original_files': 0
    }
    
    # Process each folder
    for folder_name, (output_folder, label) in folder_mapping.items():
        source_folder = cropped_path / folder_name
        
        if not source_folder.exists():
            continue
        
        print(f"\n📁 Processing {folder_name} folder...")
        
        # Get all image files
        image_files = [f for f in source_folder.iterdir() 
                      if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']]
        
        print(f"   Found {len(image_files)} images")
        
        for img_idx, img_path in enumerate(image_files):
            if (img_idx + 1) % 50 == 0:
                print(f"   Processing {img_idx + 1}/{len(image_files)}...")
            
            try:
                # Load image
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"⚠️  Warning: Could not load {img_path}, skipping")
                    continue
                
                # Save original (resized but not augmented)
                base_name = img_path.stem
                original_output = output_folder / f"{base_name}_orig.png"
                img_resized = base_transform(img)
                img_resized.save(original_output)
                stats['total_samples'] += 1
                stats['original_files'] += 1
                if label == 1:
                    stats['worm_samples'] += 1
                else:
                    stats['background_samples'] += 1
                
                # Create augmented versions
                for aug_idx in range(min(augmentations_per_image, len(augmentation_transforms))):
                    try:
                        # Apply augmentation
                        aug_transform = augmentation_transforms[aug_idx]
                        img_aug = aug_transform(img)
                        
                        # Save augmented image
                        aug_output = output_folder / f"{base_name}_aug{aug_idx+1}.png"
                        img_aug.save(aug_output)
                        
                        stats['total_samples'] += 1
                        if label == 1:
                            stats['worm_samples'] += 1
                        else:
                            stats['background_samples'] += 1
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to create augmentation {aug_idx+1} for {img_path}: {e}")
                        continue
            except Exception as e:
                print(f"⚠️  Warning: Failed to process {img_path}: {e}")
                continue
        
        print(f"   ✅ Processed {len(image_files)} images from {folder_name}")
    
    return stats


def main():
    """Main execution function"""
    print("🪱 Worm Classifier Dataset Generator")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Ask user which method to use (only in GUI mode)
    if not args.dataset_dir:
        # GUI mode - ask which method
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        choice = messagebox.askyesno(
            "Select Source Data Method",
            "How would you like to generate the dataset?\n\n"
            "YES: Extract crops from larger images with masks\n"
            "(Requires DatasetImages and DatasetMasks folders)\n\n"
            "NO: Use already cropped files from TP/TN/FP/FN folders\n"
            "(Requires True Positive, True Negative, False Positive, False Negative folders)",
            parent=root
        )
        root.destroy()
        
        use_extraction_method = choice
    else:
        # Command line mode - default to extraction method
        use_extraction_method = True
    
    # Determine dataset and output directories
    if args.dataset_dir:
        # Command line mode
        dataset_dir = args.dataset_dir
        output_dir = args.output_dir if args.output_dir else dataset_dir
        print(f"📁 Command line mode:")
        print(f"   Dataset directory: {dataset_dir}")
        print(f"   Output directory: {output_dir}")
        print(f"   Method: {'Extract from images with masks' if use_extraction_method else 'Use cropped TP/TN/FP/FN files'}")
    else:
        # GUI mode
        print(f"📁 GUI mode - selecting folder...")
        if use_extraction_method:
            dataset_dir = select_dataset_folder()
        else:
            dataset_dir = select_cropped_folder()
        output_dir = dataset_dir  # Use same folder for output
        print(f"   Selected folder: {dataset_dir}")
        print(f"   Output will be saved to: {output_dir}")
        print(f"   Method: {'Extract from images with masks' if use_extraction_method else 'Use cropped TP/TN/FP/FN files'}")
    
    # Process based on selected method
    if use_extraction_method:
        # Original extraction method
        # Validate dataset structure
        try:
            images_dir, masks_dir, image_files, mask_files = validate_dataset_structure(dataset_dir)
        except ValueError as e:
            print(f"❌ Dataset validation failed: {e}")
            sys.exit(1)
    else:
        # New method: use cropped files from TP/TN/FP/FN folders
        # Validate cropped folder structure
        try:
            validate_cropped_structure(dataset_dir)
        except ValueError as e:
            print(f"❌ Folder validation failed: {e}")
            sys.exit(1)
    
    if use_extraction_method:
        # Original extraction method
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
                'method': 'extraction_from_images',
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
            
        except Exception as e:
            print(f"\n❌ Dataset generation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    else:
        # New method: use cropped files from TP/TN/FP/FN folders
        print(f"\n📊 Dataset Generation Plan:")
        print(f"   Window size: {args.window_size}x{args.window_size}")
        print(f"   Method: Using cropped files with augmentations")
        print(f"   Random seed: {args.random_seed}")
        
        if args.dry_run:
            print(f"\n🔍 DRY RUN MODE - No files will be generated")
            print(f"   Output directory: {output_dir}")
            print(f"   Would create: {output_dir}/Classifier/Worm/")
            print(f"   Would create: {output_dir}/Classifier/Background/")
            return
        
        # Generate dataset from cropped files
        print(f"\n🚀 Starting dataset generation from cropped files...")
        start_time = datetime.now()
        
        try:
            stats = generate_from_cropped_files(
                cropped_dir=dataset_dir,
                output_dir=output_dir,
                window_size=args.window_size,
                random_seed=args.random_seed
            )
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n🎉 Dataset generation completed successfully!")
            print(f"   Duration: {duration}")
            print(f"   Total samples: {stats['total_samples']}")
            print(f"   Worm samples: {stats['worm_samples']}")
            print(f"   Background samples: {stats['background_samples']}")
            print(f"   Augmentations per image: {stats['augmentations_per_image']}")
            
            # Save summary report
            summary_file = os.path.join(output_dir, "generation_summary.json")
            summary = {
                'method': 'cropped_files_with_augmentation',
                'command_line_args': vars(args),
                'statistics': stats,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration.total_seconds()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"   Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"\n❌ Dataset generation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    # Show next steps
    print(f"\n📋 Next Steps:")
    print(f"   1. Review generated crops in: {output_dir}/Classifier/")
    print(f"   2. Train classifier using: train_worm_classifier.py")
    print(f"   3. Or use the classifier directly in your analysis pipeline")


if __name__ == "__main__":
    main() 