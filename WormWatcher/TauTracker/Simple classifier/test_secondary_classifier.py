#!/usr/bin/env python3
"""
Test Secondary Classifier

This script evaluates a trained worm classifier on test data and plots
sensitivity, specificity, and accuracy as a function of confidence threshold.

The test data folder can contain either:
- worm/ and background/ subfolders
- True Positive/, True Negative/, False Positive/, False Negative/ subfolders

Usage:
    python test_secondary_classifier.py

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import WormClassifier
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from WormClassifier import WormClassifier, get_default_config
except ImportError as e:
    print(f"❌ Error importing WormClassifier: {e}")
    print("Make sure WormClassifier.py is in the parent directory.")
    sys.exit(1)

try:
    from Scripts.file_cache import file_cache
except ImportError as e:
    print(f"❌ Error importing file_cache: {e}")
    print("Make sure Scripts/file_cache.py exists.")
    sys.exit(1)

import tkinter as tk
from tkinter import messagebox, filedialog


def select_test_data_folder():
    """Select test data folder using GUI with cache support"""
    script_name = os.path.basename(__file__)
    
    # Try to load cached folder path
    cached_paths = file_cache.load_file_paths(script_name)
    
    if cached_paths and 'test_data_folder' in cached_paths:
        # Show cache option to user
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        cached_folder = cached_paths['test_data_folder']
        folder_name = os.path.basename(cached_folder)
        
        use_cache = messagebox.askyesno(
            "Use Cached Folder?",
            f"Found previously selected test data folder:\n\n{folder_name}\n\nWould you like to use this folder again?",
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
        title="Select Test Data Folder (containing worm/background OR TP/TN/FP/FN subfolders)"
    )
    
    root.destroy()
    
    if not folder_path:
        print("❌ No folder selected. Exiting.")
        sys.exit(1)
    
    # Save to cache
    file_cache.save_file_paths(script_name, [folder_path], ['test_data_folder'])
    
    return folder_path


def load_config_path():
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


def select_model_file():
    """Select trained model file using GUI with cache support, checking config first"""
    script_name = os.path.basename(__file__)
    
    # First, try to load from config file
    config_path = load_config_path()
    
    # Try to load cached file path
    cached_paths = file_cache.load_file_paths(script_name)
    
    # Determine which path to suggest (priority: config > cache)
    suggested_path = config_path if config_path else (cached_paths.get('model_file') if cached_paths and 'model_file' in cached_paths else None)
    
    if suggested_path and os.path.exists(suggested_path):
        # Show option to user
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        file_name = os.path.basename(suggested_path)
        source = "config file" if suggested_path == config_path else "cache"
        
        use_suggested = messagebox.askyesno(
            f"Use Model from {source.title()}?",
            f"Found model path:\n\n{file_name}\n\n{suggested_path}\n\nWould you like to use this model?",
            parent=root
        )
        
        root.destroy()
        
        if use_suggested:
            # Save to cache for future use
            file_cache.save_file_paths(script_name, [suggested_path], ['model_file'])
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
        title="Select Trained Model File (.pth)",
        initialdir=initial_dir,
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
    )
    
    root.destroy()
    
    if not file_path:
        print("❌ No model file selected. Exiting.")
        sys.exit(1)
    
    # Save to cache
    file_cache.save_file_paths(script_name, [file_path], ['model_file'])
    
    return file_path


def load_test_data(data_folder):
    """
    Load test data from folder
    
    Supports two folder structures:
    1. worm/ and background/ subfolders
    2. True Positive/, True Negative/, False Positive/, False Negative/ subfolders
    
    Returns:
        tuple: (image_paths, true_labels) where labels are 1=worm, 0=background
    """
    data_folder = Path(data_folder)
    
    # Check for worm/background structure
    worm_dir = data_folder / "worm"
    background_dir = data_folder / "background"
    
    # Check for TP/TN/FP/FN structure
    tp_dir = data_folder / "True Positive"
    tn_dir = data_folder / "True Negative"
    fp_dir = data_folder / "False Positive"
    fn_dir = data_folder / "False Negative"
    
    image_paths = []
    true_labels = []
    
    # Try worm/background structure first
    if worm_dir.exists() and background_dir.exists():
        print(f"📁 Found worm/background folder structure")
        
        # Load worm images (label = 1)
        worm_files = [f for f in worm_dir.iterdir() 
                      if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']]
        for f in worm_files:
            image_paths.append(str(f))
            true_labels.append(1)
        
        # Load background images (label = 0)
        bg_files = [f for f in background_dir.iterdir() 
                   if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']]
        for f in bg_files:
            image_paths.append(str(f))
            true_labels.append(0)
        
        print(f"   Worm samples: {len(worm_files)}")
        print(f"   Background samples: {len(bg_files)}")
        
    # Try TP/TN/FP/FN structure
    elif (tp_dir.exists() or tn_dir.exists() or fp_dir.exists() or fn_dir.exists()):
        print(f"📁 Found TP/TN/FP/FN folder structure")
        
        # True Positive: actual=worm (1)
        if tp_dir.exists():
            tp_files = [f for f in tp_dir.iterdir() 
                       if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']]
            for f in tp_files:
                image_paths.append(str(f))
                true_labels.append(1)
            print(f"   True Positive samples: {len(tp_files)}")
        
        # True Negative: actual=background (0)
        if tn_dir.exists():
            tn_files = [f for f in tn_dir.iterdir() 
                       if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']]
            for f in tn_files:
                image_paths.append(str(f))
                true_labels.append(0)
            print(f"   True Negative samples: {len(tn_files)}")
        
        # False Positive: actual=background (0) - incorrectly identified as worm
        if fp_dir.exists():
            fp_files = [f for f in fp_dir.iterdir() 
                       if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']]
            for f in fp_files:
                image_paths.append(str(f))
                true_labels.append(0)
            print(f"   False Positive samples: {len(fp_files)}")
        
        # False Negative: actual=worm (1) - incorrectly identified as background
        if fn_dir.exists():
            fn_files = [f for f in fn_dir.iterdir() 
                       if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.tif']]
            for f in fn_files:
                image_paths.append(str(f))
                true_labels.append(1)
            print(f"   False Negative samples: {len(fn_files)}")
    
    else:
        raise ValueError(
            f"Test data folder must contain either:\n"
            f"  - worm/ and background/ subfolders, OR\n"
            f"  - True Positive/, True Negative/, False Positive/, False Negative/ subfolders\n"
            f"Found folder: {data_folder}"
        )
    
    if len(image_paths) == 0:
        raise ValueError(f"No image files found in {data_folder}")
    
    # Limit to 1000 samples per class if dataset is large (randomly selected)
    max_samples_per_class = 1000
    worm_indices = [i for i, label in enumerate(true_labels) if label == 1]
    background_indices = [i for i, label in enumerate(true_labels) if label == 0]
    
    if len(worm_indices) > max_samples_per_class or len(background_indices) > max_samples_per_class:
        print(f"\n📊 Dataset is large, limiting to {max_samples_per_class} samples per class...")
        
        # Randomly sample indices
        np.random.seed(42)  # For reproducibility
        
        if len(worm_indices) > max_samples_per_class:
            selected_worm_indices = np.random.choice(worm_indices, max_samples_per_class, replace=False)
            print(f"   Randomly selected {max_samples_per_class} from {len(worm_indices)} worm samples")
        else:
            selected_worm_indices = worm_indices
        
        if len(background_indices) > max_samples_per_class:
            selected_background_indices = np.random.choice(background_indices, max_samples_per_class, replace=False)
            print(f"   Randomly selected {max_samples_per_class} from {len(background_indices)} background samples")
        else:
            selected_background_indices = background_indices
        
        # Combine selected indices
        selected_indices = np.concatenate([selected_worm_indices, selected_background_indices])
        
        # Filter image_paths and true_labels
        image_paths = [image_paths[i] for i in selected_indices]
        true_labels = [true_labels[i] for i in selected_indices]
    
    print(f"✅ Total test samples: {len(image_paths)}")
    print(f"   Worm samples (label=1): {sum(true_labels)}")
    print(f"   Background samples (label=0): {len(true_labels) - sum(true_labels)}")
    
    return image_paths, true_labels


def evaluate_at_threshold(classifier, image_paths, true_labels, threshold):
    """
    Evaluate classifier at a specific confidence threshold
    
    Returns:
        dict: Metrics including TP, TN, FP, FN, sensitivity, specificity, accuracy
    """
    # Load and predict on all images (get probabilities)
    probabilities = []
    valid_indices = []
    
    for i, img_path in enumerate(image_paths):
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️  Warning: Could not load {img_path}, skipping")
            continue
        
        # Predict (get probability, not binary prediction)
        result = classifier.predict([img])
        probabilities.append(result['probabilities'][0])
        valid_indices.append(i)
    
    # Filter true_labels to match valid images
    true_labels_filtered = [true_labels[i] for i in valid_indices]
    
    # Apply threshold to get binary predictions
    predictions = [1 if p >= threshold else 0 for p in probabilities]
    
    # Calculate confusion matrix
    true_labels_array = np.array(true_labels_filtered)
    predictions_array = np.array(predictions)
    
    TP = np.sum((true_labels_array == 1) & (predictions_array == 1))
    TN = np.sum((true_labels_array == 0) & (predictions_array == 0))
    FP = np.sum((true_labels_array == 0) & (predictions_array == 1))
    FN = np.sum((true_labels_array == 1) & (predictions_array == 0))
    
    # Calculate metrics
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Recall
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy
    }


def main():
    """Main execution function"""
    print("🧪 Secondary Classifier Tester")
    print("=" * 50)
    
    # Select test data folder
    print(f"\n📁 Selecting test data folder...")
    data_folder = select_test_data_folder()
    print(f"   Selected: {data_folder}")
    
    # Load test data
    print(f"\n📚 Loading test data...")
    try:
        image_paths, true_labels = load_test_data(data_folder)
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        sys.exit(1)
    
    # Try to load window size from config file first
    config_file = os.path.join(parent_dir, "AnalyzePositionOnSinglePlate.json")
    window_size_from_config = None
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            if 'secondary_classifier' in config and 'secondary_classifier_window_size' in config['secondary_classifier']:
                window_size_from_config = config['secondary_classifier']['secondary_classifier_window_size']
        except Exception as e:
            print(f"⚠️  Warning: Could not load window_size from config: {e}")
    
    # Select model file
    print(f"\n🤖 Selecting trained model...")
    model_path = select_model_file()
    print(f"   Selected: {model_path}")
    
    # Load model to get window size
    print(f"\n🔧 Loading model configuration...")
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'config' not in checkpoint:
            # Try to use window_size from JSON config if available
            if window_size_from_config is not None:
                print(f"   Using window_size from config file: {window_size_from_config}")
                window_size = window_size_from_config
            else:
                raise ValueError("Model file missing 'config' - cannot determine window_size")
        else:
            window_size = checkpoint['config']['window_size']
        print(f"   Window size: {window_size}x{window_size}")
    except Exception as e:
        print(f"❌ Error loading model config: {e}")
        sys.exit(1)
    
    # Initialize classifier
    print(f"\n🔧 Initializing classifier...")
    config = get_default_config(window_size=window_size)
    classifier = WormClassifier(config)
    classifier.load_model(model_path)
    
    # Define thresholds to test
    thresholds = [0.25, 0.5, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 0.999, 1.000]
    
    print(f"\n📊 Evaluating at {len(thresholds)} confidence thresholds...")
    print(f"   Thresholds: {thresholds}")
    
    # Evaluate at each threshold
    results = []
    for i, threshold in enumerate(thresholds):
        print(f"   Evaluating threshold {i+1}/{len(thresholds)}: {threshold:.3f}...", end='', flush=True)
        metrics = evaluate_at_threshold(classifier, image_paths, true_labels, threshold)
        results.append({
            'threshold': threshold,
            **metrics
        })
        print(f" Sensitivity={metrics['sensitivity']:.3f}, Specificity={metrics['specificity']:.3f}, Accuracy={metrics['accuracy']:.3f}")
    
    # Extract data for plotting
    thresholds_plot = [r['threshold'] for r in results]
    sensitivity_plot = [r['sensitivity'] for r in results]
    specificity_plot = [r['specificity'] for r in results]
    accuracy_plot = [r['accuracy'] for r in results]
    
    # Create plot
    print(f"\n📈 Generating plot...")
    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds_plot, sensitivity_plot, 'o-', label='Sensitivity (Recall)', linewidth=2, markersize=8)
    plt.plot(thresholds_plot, specificity_plot, 's-', label='Specificity', linewidth=2, markersize=8)
    plt.plot(thresholds_plot, accuracy_plot, '^-', label='Accuracy', linewidth=2, markersize=8)
    
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Classifier Performance vs Confidence Threshold', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.2, 1.05])
    plt.ylim([0, 1.05])
    
    # Add value labels on points
    for i, (th, sens, spec, acc) in enumerate(zip(thresholds_plot, sensitivity_plot, specificity_plot, accuracy_plot)):
        if i % 2 == 0:  # Label every other point to avoid clutter
            plt.annotate(f'{sens:.2f}', (th, sens), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            plt.annotate(f'{spec:.2f}', (th, spec), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)
            plt.annotate(f'{acc:.2f}', (th, acc), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(data_folder).parent
    plot_path = output_dir / "classifier_threshold_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   Plot saved to: {plot_path}")
    
    # Show plot
    plt.show()
    
    # Print summary table
    print(f"\n📋 Summary Table:")
    print(f"{'Threshold':<12} {'Sensitivity':<12} {'Specificity':<12} {'Accuracy':<12} {'TP':<6} {'TN':<6} {'FP':<6} {'FN':<6}")
    print("-" * 80)
    for r in results:
        print(f"{r['threshold']:<12.3f} {r['sensitivity']:<12.3f} {r['specificity']:<12.3f} {r['accuracy']:<12.3f} "
              f"{r['TP']:<6} {r['TN']:<6} {r['FP']:<6} {r['FN']:<6}")
    
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
