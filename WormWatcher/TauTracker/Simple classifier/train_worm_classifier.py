#!/usr/bin/env python3
"""
Train Worm Classifier

This script trains a worm classifier using a previously generated dataset from
the generate_worm_classifier_dataset.py script.

Usage:
    python train_worm_classifier.py

Author: AI Assistant
Date: 2024
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
from pathlib import Path

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

import tkinter as tk
from tkinter import messagebox, filedialog


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train worm classifier using generated dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # GUI mode (default) - will prompt for classifier folder selection
    python train_worm_classifier.py
    
    # Command line mode with custom parameters
    python train_worm_classifier.py \\
        --classifier_dir ./classifier_data \\
        --window_size 40 \\
        --batch_size 32 \\
        --num_epochs 100 \\
        --learning_rate 0.001
    
    # Quick training with default parameters
    python train_worm_classifier.py \\
        --classifier_dir ./classifier_data \\
        --quick_mode
        """
    )
    
    # Optional arguments (GUI mode is default)
    parser.add_argument(
        '--classifier_dir', 
        type=str, 
        help='Directory containing Classifier/Worm and Classifier/Background folders (GUI mode if not specified)'
    )
    
    # Training parameters
    parser.add_argument(
        '--window_size', 
        type=int, 
        default=40,
        help='Window size for input images (default: 40)'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size for training (default: 32)'
    )
    
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--dropout_rate', 
        type=float, 
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    
    parser.add_argument(
        '--early_stopping_patience', 
        type=int, 
        default=10,
        help='Early stopping patience (default: 10)'
    )
    
    parser.add_argument(
        '--confidence_threshold', 
        type=float, 
        default=0.5,
        help='Classification confidence threshold (default: 0.5)'
    )
    
    # Data split parameters
    parser.add_argument(
        '--train_split', 
        type=float, 
        default=0.8,
        help='Fraction of data for training (default: 0.8)'
    )
    
    parser.add_argument(
        '--val_split', 
        type=float, 
        default=0.1,
        help='Fraction of data for validation (default: 0.1)'
    )
    
    # Training modes
    parser.add_argument(
        '--quick_mode', 
        action='store_true',
        help='Quick training mode with reduced epochs and batch size'
    )
    
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--dry_run', 
        action='store_true',
        help='Show what would be done without actually training'
    )
    
    parser.add_argument(
        '--transfer_from', 
        type=str,
        help='Path to existing model file (.pth) to load and continue training (transfer learning)'
    )
    
    return parser.parse_args()


def select_classifier_folder():
    """Select classifier folder using GUI with cache support"""
    script_name = os.path.basename(__file__)
    
    # Try to load cached folder path
    cached_paths = file_cache.load_file_paths(script_name)
    
    if cached_paths and 'classifier_folder' in cached_paths:
        # Show cache option to user
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        cached_folder = cached_paths['classifier_folder']
        folder_name = os.path.basename(cached_folder)
        
        use_cache = messagebox.askyesno(
            "Use Cached Folder?",
            f"Found previously selected classifier folder:\n\n{folder_name}\n\nWould you like to use this folder again?",
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
        title="Select Classifier Folder (containing Classifier/Worm and Classifier/Background)"
    )
    
    root.destroy()
    
    if not folder_path:
        print("❌ No folder selected. Exiting.")
        sys.exit(1)
    
    # Save to cache
    file_cache.save_file_paths(script_name, [folder_path], ['classifier_folder'])
    
    return folder_path


def load_config_model_path():
    """Load secondary classifier path from AnalyzePositionOnSinglePlate.json"""
    config_file = os.path.join(parent_dir, "AnalyzePositionOnSinglePlate.json")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            if 'secondary_classifier' in config and 'secondary_classifier_path' in config['secondary_classifier']:
                model_path = config['secondary_classifier']['secondary_classifier_path']
                if os.path.exists(model_path):
                    return model_path
        except Exception as e:
            print(f"⚠️  Warning: Could not load config from {config_file}: {e}")
    
    return None


def select_transfer_model():
    """Select existing model file for transfer learning using GUI with cache support"""
    script_name = os.path.basename(__file__)
    
    # First, try to load from config file
    config_path = load_config_model_path()
    
    # Try to load cached file path
    cached_paths = file_cache.load_file_paths(script_name)
    
    # Determine which path to suggest (priority: config > cache)
    suggested_path = config_path if config_path else (cached_paths.get('transfer_model') if cached_paths and 'transfer_model' in cached_paths else None)
    
    if suggested_path and os.path.exists(suggested_path):
        # Show option to user
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_name = os.path.basename(suggested_path)
        source = "config file (AnalyzePositionOnSinglePlate.json)" if suggested_path == config_path else "cache"
        
        use_suggested = messagebox.askyesno(
            f"Use Model from {source.title()}?",
            f"Found model path:\n\n{file_name}\n\n{suggested_path}\n\nWould you like to use this model?",
            parent=root
        )
        
        root.destroy()
        
        if use_suggested:
            # Save to cache for future use
            file_cache.save_file_paths(script_name, [suggested_path], ['transfer_model'])
            return suggested_path
    
    # Select new file
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    # Set initial directory if we have a config path
    initial_dir = None
    if config_path:
        initial_dir = os.path.dirname(config_path)
    
    file_path = filedialog.askopenfilename(
        title="Select Existing Model File for Transfer Learning (.pth)",
        initialdir=initial_dir,
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
    )
    
    root.destroy()
    
    if not file_path:
        return None  # User cancelled, return None to indicate no transfer learning
    
    # Save to cache
    file_cache.save_file_paths(script_name, [file_path], ['transfer_model'])
    
    return file_path


def validate_classifier_folder(classifier_dir):
    """Validate that the classifier directory has the expected structure"""
    worm_dir = os.path.join(classifier_dir, "Classifier", "Worm")
    background_dir = os.path.join(classifier_dir, "Classifier", "Background")
    
    # Check if directories exist
    if not os.path.exists(worm_dir):
        raise ValueError(f"Classifier/Worm directory not found: {worm_dir}")
    if not os.path.exists(background_dir):
        raise ValueError(f"Classifier/Background directory not found: {background_dir}")
    
    # Check if directories contain files
    worm_files = [f for f in os.listdir(worm_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    bg_files = [f for f in os.listdir(background_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    if len(worm_files) == 0:
        raise ValueError(f"No worm image files found in {worm_dir}")
    if len(bg_files) == 0:
        raise ValueError(f"No background image files found in {background_dir}")
    
    print(f"✅ Classifier folder validation passed:")
    print(f"   Worm samples: {len(worm_files)} files in {worm_dir}")
    print(f"   Background samples: {len(bg_files)} files in {background_dir}")
    
    return worm_dir, background_dir, worm_files, bg_files


def estimate_training_time(num_samples, batch_size, num_epochs, window_size):
    """Estimate training time based on dataset size and parameters"""
    # Rough estimate: 0.1 seconds per batch per epoch
    batches_per_epoch = max(1, num_samples // batch_size)
    total_batches = batches_per_epoch * num_epochs
    estimated_seconds = total_batches * 0.1
    
    # Adjust for window size (larger images take longer)
    size_factor = (window_size / 64) ** 2
    estimated_seconds *= size_factor
    
    return estimated_seconds


def main():
    """Main execution function"""
    print("🪱 Worm Classifier Trainer")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Determine classifier directory and transfer model
    transfer_model_path = None
    
    if args.classifier_dir:
        # Command line mode
        classifier_dir = args.classifier_dir
        transfer_model_path = args.transfer_from
        print(f"📁 Command line mode:")
        print(f"   Classifier directory: {classifier_dir}")
        if transfer_model_path:
            print(f"   Transfer learning from: {transfer_model_path}")
    else:
        # GUI mode
        print(f"📁 GUI mode - selecting classifier folder...")
        classifier_dir = select_classifier_folder()
        print(f"   Selected folder: {classifier_dir}")
        
        # Ask if user wants transfer learning
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        use_transfer = messagebox.askyesno(
            "Transfer Learning?",
            "Would you like to continue training from an existing model?\n\n"
            "YES: Select an existing model to continue training\n"
            "NO: Train a new model from scratch",
            parent=root
        )
        root.destroy()
        
        if use_transfer:
            print(f"\n📥 Selecting existing model for transfer learning...")
            transfer_model_path = select_transfer_model()
            if transfer_model_path:
                print(f"   Selected model: {transfer_model_path}")
            else:
                print(f"   No model selected, training from scratch")
                transfer_model_path = None
    
    # Validate classifier folder structure
    try:
        worm_dir, background_dir, worm_files, bg_files = validate_classifier_folder(classifier_dir)
    except ValueError as e:
        print(f"❌ Classifier folder validation failed: {e}")
        sys.exit(1)
    
    # Apply quick mode settings if requested
    if args.quick_mode:
        print(f"⚡ Quick mode enabled - reducing training parameters")
        args.num_epochs = min(args.num_epochs, 20)
        args.batch_size = min(args.batch_size, 16)
        args.early_stopping_patience = min(args.early_stopping_patience, 5)
    
    # Calculate total samples
    total_samples = len(worm_files) + len(bg_files)
    estimated_time = estimate_training_time(total_samples, args.batch_size, args.num_epochs, args.window_size)
    
    print(f"\n📊 Training Configuration:")
    print(f"   Window size: {args.window_size}x{args.window_size}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Dropout rate: {args.dropout_rate}")
    print(f"   Number of epochs: {args.num_epochs}")
    print(f"   Early stopping patience: {args.early_stopping_patience}")
    print(f"   Confidence threshold: {args.confidence_threshold}")
    print(f"   Train/Val split: {args.train_split:.1%}/{args.val_split:.1%}")
    print(f"   Total samples: {total_samples}")
    print(f"   Estimated training time: {estimated_time/60:.1f} minutes")
    
    if args.dry_run:
        print(f"\n🔍 DRY RUN MODE - No training will be performed")
        print(f"   Would train model with above configuration")
        return
    
    # Set up Models directory in data folder for all intermediate and final files
    models_dir = os.path.join(classifier_dir, "Classifier", "Models")
    os.makedirs(models_dir, exist_ok=True)
    print(f"📁 All model files will be saved to: {models_dir}")
    
    # Configure classifier
    config = get_default_config(window_size=args.window_size)
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'dropout_rate': args.dropout_rate,
        'early_stopping_patience': args.early_stopping_patience,
        'confidence_threshold': args.confidence_threshold,
        'model_save_path': models_dir  # Set to data folder Models directory
    })
    
    # Initialize classifier
    print(f"\n🔧 Initializing WormClassifier...")
    classifier = WormClassifier(config)
    
    # Load existing model for transfer learning if specified
    if transfer_model_path:
        if not os.path.exists(transfer_model_path):
            print(f"❌ Error: Transfer model file not found: {transfer_model_path}")
            sys.exit(1)
        print(f"\n📥 Loading existing model for transfer learning: {transfer_model_path}")
        classifier.load_model(transfer_model_path)
        print(f"   ✅ Model loaded. Continuing training with new data...")
        # Set model back to training mode (load_model sets it to eval mode)
        classifier.model.train()
    
    # Train the model
    print(f"\n🚀 Starting training...")
    start_time = datetime.now()
    
    try:
        results = classifier.train_from_generated_dataset(
            classifier_dir=classifier_dir,
            train_split=args.train_split,
            val_split=args.val_split
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Duration: {duration}")
        print(f"   Model saved to: {results['model_path']}")
        
        # Display test metrics
        test_metrics = results['test_metrics']
        print(f"\n📈 Test Set Performance:")
        print(f"   Accuracy: {test_metrics['accuracy']:.3f}")
        print(f"   Precision: {test_metrics['precision']:.3f}")
        print(f"   Recall: {test_metrics['recall']:.3f}")
        print(f"   F1-Score: {test_metrics['f1_score']:.3f}")
        
        # Save training summary in Models folder
        summary_file = os.path.join(models_dir, "training_summary.json")
        summary = {
            'command_line_args': vars(args),
            'training_config': config,
            'results': results,
            'data_splits': results['data_splits'],
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds()
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   Training summary saved to: {summary_file}")
        
        # Show next steps
        print(f"\n📋 Next Steps:")
        print(f"   1. Model is ready for inference: {results['model_path']}")
        print(f"   2. Use the classifier in your analysis pipeline")
        print(f"   3. Review training plots and metrics in: {summary_file}")
        print(f"   4. Models are saved in: {os.path.join(classifier_dir, 'Classifier', 'Models')}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 