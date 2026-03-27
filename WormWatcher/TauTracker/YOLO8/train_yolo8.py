"""
Training Script for YOLOv8 Segmentation on Worm Dataset
Supports VGG JSON format annotations (converts to YOLO format)

Usage:
    python train_yolo8.py [config.json]
    
    A configuration file is required. If no config file is provided, a file dialog will open to select one.
    
    Example config.json structure:
    {
      // Dataset paths
      // Note: Supports both VGG JSON (.json) and CVAT XML (.xml) annotation formats
      "train_json": "I:\\Tau worm mask RCNN\\Model 4c - combinaed 4a+4b\\Original\\merged_labels_20251207_234814.json",  // Path to training annotations (VGG JSON or CVAT XML)
      "train_images": "I:\\Tau worm mask RCNN\\Model 4c - combinaed 4a+4b\\Original\\DatasetImages",          // Path to training images directory
      "val_json": "E:\\Nikita JHU Plates\\Validation only data\\Data for Labeling\\merged_labels_20251109_221604.json",    // Path to validation annotations (VGG JSON or CVAT XML, null to auto-split)
      "val_images": "E:\\Nikita JHU Plates\\Validation only data\\Data for Labeling",                              // Path to validation images (null to auto-split)
      "yolo_dataset": null,                           // Path to existing YOLO dataset (null to convert from VGG)
      
      // Model settings
      "model_size": "n",                               // Model size: n, s, m, l, x
      "task": "detect",                                // Task mode: "seg" (segmentation) or "detect" (bounding boxes)
      "pretrained": true,                              // Use pretrained COCO weights
      "resume": null,                                  // Path to checkpoint to resume from (null for new training)
      
      // Training hyperparameters
      "epochs": 1000,                                  // Number of training epochs (random augmentation makes them all different)
      "batch_size": 16,                                // Batch size
      "imgsz": 512,                                    // Image size for training
      "learning_rate": 0.01,                           // Initial learning rate
      "device": null,                                  // Device: null=auto, "cuda", "cpu", or "0" for specific GPU
      "workers": 3,                                    // Number of data loading workers (0-8, lower if memory issues)
      "skip_final_val": false,                         // Skip final validation if memory issues (metrics already computed during training)
      
      // Data augmentation
      "augment": true,                                 // Enable YOLO's built-in augmentation
      "augmentation_per_image": 10,                     // Number of augmented versions to create per original image during dataset conversion (0 = no augmentation)
      
      // Output settings
      // output_dir: null = auto-set to train_images/_YOLO8_Training/project/name
      // Set to a specific path if you want custom location (must be under train_images or will warn)
      "output_dir": null,                             // null =
      "project": "worm_segmentation",                  // Project name (subfolder name)
      "name": "yolov8_seg"                            // Experiment name (subfolder name)
    }
    
    Note: JSON doesn't support comments, so remove the // comments when creating your actual config file.
    See train_yolo8_config.json.example for a valid JSON example.
"""

import os
import sys
import json
import argparse
import torch
from pathlib import Path
from typing import Dict, Optional

# ============================================================================
# Configuration is loaded from JSON file (required)
# See docstring above for example config structure with comments
# ============================================================================



# ============================================================================
# Error Handling Helper Functions
# ============================================================================

def _handle_numpy_import_error():
    """Handle NumPy import errors with helpful messages"""
    print("=" * 80)
    print("❌ NumPy import error detected!")
    print("=" * 80)
    print("This is usually caused by a corrupted or incompatible NumPy installation.")
    print("\nTo fix, run in your terminal:")
    print("  pip uninstall numpy -y")
    print("  pip install numpy")
    print("\nOr if that doesn't work:")
    print("  pip install --upgrade --force-reinstall numpy")
    print("=" * 80)


def _handle_numpy_compatibility_error():
    """Handle NumPy compatibility errors with Ultralytics"""
    print("=" * 80)
    print("❌ NumPy compatibility error with Ultralytics!")
    print("=" * 80)
    print("This is usually caused by a NumPy version mismatch.")
    print("\nTo fix, try:")
    print("  1. Reinstall NumPy:")
    print("     pip uninstall numpy -y")
    print("     pip install numpy")
    print("\n  2. If that doesn't work, try a specific compatible version:")
    print("     pip install numpy==1.24.3")
    print("\n  3. Or upgrade everything:")
    print("     pip install --upgrade numpy ultralytics")
    print("=" * 80)


def _get_device(requested_device):
    """Get device with error handling for memory issues"""
    if requested_device is None:
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except OSError as e:
            if "paging file" in str(e).lower() or "1455" in str(e):
                _handle_cuda_memory_error()
                return "cpu"
            else:
                raise
    else:
        return requested_device


def _handle_cuda_memory_error():
    """Handle CUDA loading failures due to memory issues"""
    print("\n" + "!" * 80)
    print("⚠️  WARNING: CUDA loading failed due to insufficient virtual memory!")
    print("   Falling back to CPU mode.")
    print("   To fix this permanently, increase Windows paging file size:")
    print("   1. Control Panel > System > Advanced system settings")
    print("   2. Performance > Settings > Advanced > Virtual memory")
    print("   3. Set paging file to at least 16GB (or System managed)")
    print("!" * 80 + "\n")


def _handle_validation_memory_error(save_dir, dataset_path):
    """Handle memory errors during final validation"""
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    print(f"\n⚠️  Final validation skipped due to memory constraints")
    print(f"   Training completed successfully - you can validate later using:")
    print(f"   python -m YOLO8.evaluation_yolo8 {best_model_path} {dataset_path}")
    print(f"   or programmatically:")
    print(f"   from YOLO8.evaluation_yolo8 import evaluate_yolo8_model")
    print(f"   evaluate_yolo8_model('{best_model_path}', '{dataset_path}', workers=0)")

# ============================================================================
# Import Dependencies with Error Handling
# ============================================================================

try:
    import numpy as np
except ImportError:
    _handle_numpy_import_error()
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError as e:
    if "numpy.core.multiarray" in str(e) or "numpy" in str(e).lower():
        _handle_numpy_compatibility_error()
        sys.exit(1)
    else:
        raise

try:
    from .dataset_converter import convert_vgg_to_yolo, convert_cvat_xml_to_yolo, create_yolo_dataset_config
except ImportError:
    from dataset_converter import convert_vgg_to_yolo, convert_cvat_xml_to_yolo, create_yolo_dataset_config


def get_script_directory():
    """Get the directory where this script is located"""
    return os.path.dirname(os.path.abspath(__file__))


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
    
    # Convert null/None strings to None for optional fields
    for key in ['val_json', 'val_images', 'yolo_dataset', 'resume', 'device', 'output_dir']:
        if key in config and (config[key] == 'null' or config[key] == ''):
            config[key] = None
    
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
        title="Select Training Configuration JSON File",
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


def ensure_output_in_data_directory(train_images_path, output_dir=None, project="worm_segmentation", name="yolov8_seg"):
    """
    Ensure output directory is a subfolder of train_images, not in code directory.
    
    Args:
        train_images_path: Path to training images directory
        output_dir: Desired output directory (None to auto-generate)
        project: Project name
        name: Experiment name
    
    Returns:
        str: Safe output directory path
    """
    script_dir = get_script_directory()
    train_images_abs = os.path.abspath(train_images_path)
    
    # If output_dir is None or relative, create it under train_images
    if output_dir is None or not os.path.isabs(output_dir):
        # Create output directory as subfolder of train_images
        output_dir = os.path.join(train_images_abs, "_YOLO8_Training", project, name)
    else:
        output_dir = os.path.abspath(output_dir)
    
    # Helper function to check if path1 is a parent of path2 (handles cross-drive)
    def is_parent_path(parent, child):
        """Check if parent is a parent directory of child, handling cross-drive paths"""
        try:
            # Try commonpath first (works for same drive)
            common = os.path.commonpath([parent, child])
            return common == parent or child.startswith(parent + os.sep)
        except ValueError:
            # Different drives - check if they're the same drive
            parent_drive = os.path.splitdrive(parent)[0]
            child_drive = os.path.splitdrive(child)[0]
            if parent_drive != child_drive:
                return False  # Different drives, can't be parent
            # Same drive but commonpath failed - use string comparison
            return child.startswith(parent + os.sep)
    
    # Sanity check: Ensure output is NOT in code directory
    if is_parent_path(script_dir, output_dir):
        raise ValueError(
            f"❌ ERROR: Output directory cannot be in code directory!\n"
            f"   Script directory: {script_dir}\n"
            f"   Output directory: {output_dir}\n"
            f"   Please set output_dir to be a subfolder of your data directory."
        )
    
    # Sanity check: Ensure output IS in train_images directory or a subdirectory
    if not is_parent_path(train_images_abs, output_dir):
        # Check if they're on different drives
        train_drive = os.path.splitdrive(train_images_abs)[0]
        output_drive = os.path.splitdrive(output_dir)[0]
        if train_drive != output_drive:
            print(f"⚠️  WARNING: Output directory is on a different drive than train_images")
            print(f"   Train images: {train_images_abs}")
            print(f"   Output dir: {output_dir}")
        else:
            print(f"⚠️  WARNING: Output directory is not under train_images directory")
            print(f"   Train images: {train_images_abs}")
            print(f"   Output dir: {output_dir}")
            print(f"   This is allowed but not recommended.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"✅ Output directory: {output_dir}")
    return output_dir


def main(config=None, config_file=None):
    # Parse command-line arguments first to get config_file if provided
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 segmentation on worm dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use GUI to select config file
  python train_yolo8.py
  
  # Specify config file directly
  python train_yolo8.py config.json
  
  # Override config file with command-line arguments
  python train_yolo8.py config.json --epochs 200 --batch_size 32
        """
    )
    
    parser.add_argument(
        'config_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to configuration JSON file (optional, will open file dialog if not provided)'
    )
    
    # All other arguments will override config file values
    parser.add_argument("--train_json", type=str, default=None,
                       help="Path to training annotation file (VGG JSON or CVAT XML)")
    parser.add_argument("--train_images", type=str, default=None,
                       help="Path to training images directory")
    parser.add_argument("--val_json", type=str, default=None,
                       help="Path to validation annotation file (VGG JSON or CVAT XML, optional)")
    parser.add_argument("--val_images", type=str, default=None,
                       help="Path to validation images directory (optional)")
    parser.add_argument("--yolo_dataset", type=str, default=None,
                       help="Path to existing YOLO format dataset (skip conversion if provided)")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default=None,
                       choices=["n", "s", "m", "l", "x"],
                       help="Model size: n (nano), s (small), m (medium), l (large), x (xlarge)")
    parser.add_argument("--task", type=str, default=None,
                       choices=["seg", "detect"],
                       help="Task mode: 'seg' for segmentation (masks), 'detect' for detection (bounding boxes)")
    parser.add_argument("--pretrained", action="store_true", default=None,
                       help="Use pretrained COCO weights")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                       help="Don't use pretrained COCO weights")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None,
                       help="Image size for training")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Initial learning rate")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda, cpu, or specific GPU like '0')")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of data loading workers (reduce if memory issues)")
    parser.add_argument("--skip_final_val", action="store_true", default=None,
                       help="Skip final validation to avoid memory issues (metrics already computed during training)")
    
    # Data augmentation
    parser.add_argument("--augment", action="store_true", default=None,
                       help="Enable YOLO's built-in augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                       help="Disable YOLO's built-in augmentation")
    parser.add_argument("--augmentation_per_image", type=int, default=None,
                       help="Number of augmented versions to create per original image during dataset conversion (0 = no augmentation)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for training artifacts")
    parser.add_argument("--project", type=str, default=None,
                       help="Project name")
    parser.add_argument("--name", type=str, default=None,
                       help="Experiment name")
    
    args = parser.parse_args()
    
    # Load config from file (required)
    config_file_path = None
    if config_file:
        # Config file provided via function parameter (programmatic use)
        config_file_path = config_file
    elif args.config_file:
        # Config file provided via command line
        config_file_path = args.config_file
    else:
        # No config file - use GUI to select
        print("=" * 80)
        print("No configuration file provided. Opening file dialog...")
        print("=" * 80)
        config_file_path = select_config_file_gui()
        if not config_file_path:
            print("\n❌ No configuration file selected. A config file is required.")
            print("   Please create a config.json file with your settings.")
            print("   See train_yolo8_config.json.example for an example.")
            return 1
    
    # Load config from file (required)
    if not config_file_path:
        print("❌ Error: No configuration file path. A config file is required.")
        return 1
    
    try:
        config = load_config(config_file_path)
        print(f"✅ Loaded configuration from: {config_file_path}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in configuration file: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return 1
    
    # If config was provided as parameter, use it (for programmatic use)
    if config is None:
        print("❌ Error: No configuration loaded. A config file is required.")
        return 1
    
    # Override config with command-line arguments (if provided)
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key != 'config_file' and value is not None:
            config[key] = value
    
    # Use config dict directly
    args = type('Args', (), config)()
    
    # Set device with error handling for memory issues
    device = _get_device(args.device)
    
    # If using CPU due to memory issues, reduce workers
    if device == "cpu" and args.workers > 2:
        print(f"⚠️  Reducing workers from {args.workers} to 2 for CPU mode")
        args.workers = 2
    
    # Get task mode (default to seg for backward compatibility)
    task_mode = getattr(args, 'task', 'seg')
    task_name = "Segmentation" if task_mode == "seg" else "Detection"
    
    print("\n" + "=" * 80)
    print(f"YOLOv8 {task_name} Training")
    print("=" * 80)
    print(f"Model size: YOLOv8-{args.model_size.upper()}")
    print(f"Task mode: {task_mode} ({task_name})")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.imgsz}")
    print("=" * 80 + "\n")
    
    # Validate required paths
    if args.yolo_dataset is None and (args.train_json == "path/to/train_annotations.json" or 
                                      args.train_images == "path/to/train_images"):
        print("\n" + "!" * 80)
        print("⚠️  WARNING: Please update your config.json file!")
        print("   Set 'train_json' and 'train_images' to your actual paths.")
        print("!" * 80 + "\n")
        return
    
    # Ensure train_images path exists
    if not os.path.exists(args.train_images):
        raise FileNotFoundError(f"Training images directory not found: {args.train_images}")
    
    # Set output directory to be under train_images (with sanity checks)
    print("\n" + "=" * 80)
    print("Setting up output directory...")
    print("=" * 80)
    args.output_dir = ensure_output_in_data_directory(
        args.train_images, 
        args.output_dir, 
        args.project, 
        args.name
    )
    print("=" * 80 + "\n")
    
    # Convert dataset if needed
    if args.yolo_dataset is None:
        # Detect annotation format (VGG JSON or CVAT XML)
        train_annotation_path = args.train_json
        train_ext = os.path.splitext(train_annotation_path.lower())[1] if train_annotation_path else None
        
        if train_ext == '.xml':
            print("Detected CVAT XML format...")
            if task_mode == "seg":
                print("⚠️  NOTE: CVAT XML contains bounding boxes, which will be converted to rectangular polygons.")
                print("   This means masks will be rectangular approximations of worm shapes.")
            converter_func = convert_cvat_xml_to_yolo
        else:
            print("Detected VGG JSON format...")
            converter_func = convert_vgg_to_yolo
        
        output_dataset_dir = os.path.join(args.output_dir, "dataset")
        
        # Check if dataset directory already exists
        if os.path.exists(output_dataset_dir):
            dataset_yaml_path = os.path.join(output_dataset_dir, "dataset.yaml")
            if os.path.exists(dataset_yaml_path):
                raise ValueError(
                    f"❌ ERROR: YOLO dataset directory already exists at:\n"
                    f"   {output_dataset_dir}\n\n"
                    f"   To use the existing dataset, set 'yolo_dataset' in your config to:\n"
                    f"   \"{dataset_yaml_path}\"\n\n"
                    f"   Or to recreate the dataset, delete the directory first:\n"
                    f"   {output_dataset_dir}"
                )
            else:
                # Directory exists but no dataset.yaml - warn but allow (might be partial/incomplete)
                print(f"⚠️  WARNING: Directory exists but no dataset.yaml found: {output_dataset_dir}")
                print("   Conversion will proceed, which may overwrite existing files.")
        
        print(f"Converting annotations to YOLO {task_mode} format...")
        os.makedirs(output_dataset_dir, exist_ok=True)
        
        # Get augmentation_per_image from config (default 0 = no augmentation)
        augmentation_per_image = getattr(args, 'augmentation_per_image', 0)
        
        # Convert training data
        train_output = os.path.join(output_dataset_dir, "train")
        converter_func(args.train_json, args.train_images, train_output, task=task_mode, augmentation_per_image=augmentation_per_image)
        
        # Convert validation data if provided (no augmentation for validation)
        if args.val_json and args.val_images:
            val_output = os.path.join(output_dataset_dir, "val")
            # Detect validation format
            val_ext = os.path.splitext(args.val_json.lower())[1] if args.val_json else None
            val_converter = convert_cvat_xml_to_yolo if val_ext == '.xml' else convert_vgg_to_yolo
            val_converter(args.val_json, args.val_images, val_output, task=task_mode, augmentation_per_image=0)
            
            # Create dataset config
            dataset_yaml = create_yolo_dataset_config(
                output_dataset_dir,
                train_split=1.0,  # Already split
                val_split=0.0
            )
        else:
            # Split training data into train/val
            dataset_yaml = create_yolo_dataset_config(
                train_output,
                train_split=0.8,
                val_split=0.2
            )
        
        dataset_path = dataset_yaml
    else:
        # Use existing YOLO dataset
        dataset_path = args.yolo_dataset
        if not os.path.exists(dataset_path):
            # Try to find dataset.yaml
            dataset_path = os.path.join(args.yolo_dataset, "dataset.yaml")
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset not found: {args.yolo_dataset}")
    
    print(f"\n✅ Using dataset: {dataset_path}\n")
    
    # Convert dataset_path to absolute path before changing directories
    dataset_path = os.path.abspath(dataset_path)
    print(f"📁 Dataset path (absolute): {dataset_path}")
    
    # Create or load model
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        # Co auto-set to train_images/_YOLO8_Training/project/namenvert resume path to absolute if it's relative
        resume_path = args.resume if os.path.isabs(args.resume) else os.path.abspath(args.resume)
        model = YOLO(resume_path)
    else:
        # Create new model - use detection or segmentation model based on task
        if task_mode == "detect":
            model_name = f"yolov8{args.model_size}.pt"  # Detection model (no -seg suffix)
            print(f"Creating detection model: {model_name}")
        else:
            model_name = f"yolov8{args.model_size}-seg.pt"  # Segmentation model
            print(f"Creating segmentation model: {model_name}")
        model = YOLO(model_name)
    
    # Training parameters
    # IMPORTANT: YOLO uses 'project' and 'name' to create: project/name/ directory
    # To ensure all outputs go to output_dir, we'll:
    # 1. Change working directory to output_dir (YOLO uses relative paths by default)
    # 2. Use project and name as relative paths
    # 3. Store original working directory to restore later
    
    # Store original working directory
    original_cwd = os.getcwd()
    
    # Change to output directory so YOLO creates all files there
    os.chdir(args.output_dir)
    print(f"📁 Changed working directory to: {args.output_dir}")
    
    # Use relative paths for project (relative to output_dir)
    # This ensures YOLO creates project/name/ under output_dir
    project_name = args.project  # e.g., "worm_segmentation"
    experiment_name = args.name  # e.g., "yolov8_seg"
    
    train_params = {
        'data': dataset_path,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': device,
        'workers': args.workers,  # Number of data loading workers
        'project': project_name,  # Relative to output_dir (current working directory)
        'name': experiment_name,
        'lr0': args.learning_rate,
        'augment': getattr(args, 'augment', True),  # YOLO's built-in augmentation
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs (intermediate models)
        'val': True,  # Validate during training
    }
    
    # Additional settings for memory-constrained systems
    if device == "cpu" or args.workers == 0:
        # Disable multiprocessing for CPU or if workers=0
        train_params['workers'] = 0
        print("ℹ️  Using single-threaded data loading (workers=0)")
    
    # Calculate expected save directory (relative to output_dir since we changed CWD)
    expected_save_dir = os.path.join(project_name, experiment_name)
    expected_weights_dir = os.path.join(expected_save_dir, 'weights')
    
    print(f"✅ Intermediate checkpoints will be saved every {train_params['save_period']} epochs")
    print(f"   Expected save directory: {expected_save_dir} (relative to {args.output_dir})")
    print(f"   Expected weights directory: {expected_weights_dir}")
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80 + "\n")
    try:
        results = model.train(**train_params)
    except Exception as e:
        # Restore working directory even if training fails
        os.chdir(original_cwd)
        raise
    
    # Restore original working directory
    os.chdir(original_cwd)
    print(f"📁 Restored working directory to: {original_cwd}")
    
    print("\n" + "=" * 80)
    print("✅ Training completed!")
    print("=" * 80)
    print(f"📁 Configured output directory: {args.output_dir}")
    print(f"📦 ACTUAL training run directory: {results.save_dir}")
    print(f"📦 Full path: {os.path.abspath(results.save_dir)}")
    
    # Show where intermediate checkpoints are saved
    weights_dir = os.path.join(results.save_dir, "weights")
    if os.path.exists(weights_dir):
        checkpoint_files = [f for f in os.listdir(weights_dir) if f.endswith('.pt')]
        if checkpoint_files:
            print(f"\n📦 Found {len(checkpoint_files)} model checkpoint(s) in:")
            print(f"   {weights_dir}")
            print(f"\n   Checkpoint files:")
            for ckpt in sorted(checkpoint_files):
                ckpt_path = os.path.join(weights_dir, ckpt)
                ckpt_size = os.path.getsize(ckpt_path) / (1024 * 1024)  # MB
                print(f"   - {ckpt} ({ckpt_size:.1f} MB)")
        else:
            print(f"\n⚠️  No checkpoint files found in: {weights_dir}")
    else:
        print(f"\n⚠️  Weights directory not found: {weights_dir}")
        print(f"   Expected location: {weights_dir}")
        # Try to find weights directory
        runs_dir = os.path.join(args.output_dir, 'runs')
        if os.path.exists(runs_dir):
            print(f"\n   Searching in runs directory: {runs_dir}")
            for root, dirs, files in os.walk(runs_dir):
                if 'weights' in dirs:
                    found_weights = os.path.join(root, 'weights')
                    found_ckpts = [f for f in os.listdir(found_weights) if f.endswith('.pt')]
                    if found_ckpts:
                        print(f"   ✅ Found weights in: {found_weights}")
                        print(f"      Checkpoints: {', '.join(sorted(found_ckpts))}")
    
    print(f"\n📊 Results: {results}")
    print("=" * 80 + "\n")
    
    # Validate final model (with error handling for memory issues)
    if args.skip_final_val:
        print("\n⚠️  Final validation skipped (skip_final_val=True)")
        print("   Validation metrics were already computed during training.")
        print("   To validate later, use:")
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        print(f"   python -m YOLO8.evaluation_yolo8 {best_model_path} {dataset_path}")
        print(f"   or programmatically:")
        print(f"   from YOLO8.evaluation_yolo8 import evaluate_yolo8_model")
        print(f"   evaluate_yolo8_model('{best_model_path}', '{dataset_path}')")
    else:
        print("Running final validation...")
        try:
            # Use same worker settings as training to avoid multiprocessing issues
            # IMPORTANT: Include project and name to ensure validation outputs go to same directory
            val_params = {
                'data': dataset_path,
                'imgsz': args.imgsz,
                'device': device,
                'workers': 0 if device == "cpu" or args.workers == 0 else min(args.workers, 2),  # Reduce workers for validation
                'project': project_name,  # Use same project as training
                'name': experiment_name,   # Use same name as training
            }
            metrics = model.val(**val_params)
            print(f"\n✅ Final validation metrics:")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            if task_mode == "seg" and hasattr(metrics, 'seg'):
                print(f"  mAP50-seg: {metrics.seg.map50:.4f}")
                print(f"  mAP50-95-seg: {metrics.seg.map:.4f}")
        except (OSError, MemoryError) as e:
            if "paging file" in str(e).lower() or "1455" in str(e) or "memory" in str(e).lower():
                _handle_validation_memory_error(results.save_dir, dataset_path)
                print(f"\n   You can also run evaluation separately using:")
                best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
                print(f"   python -m YOLO8.evaluation_yolo8 {best_model_path} {dataset_path}")
            else:
                raise


if __name__ == "__main__":
    main()

