"""
YOLOv8 Model Evaluation Module
Standalone evaluation tool for YOLOv8 models on datasets.

Can be used independently of the training routine to evaluate trained models
on validation/test datasets and compute metrics including PR curves.

Usage:
    python -m YOLO8.evaluation_yolo8 model.pt dataset.yaml
    python -m YOLO8.evaluation_yolo8 model.pt dataset.yaml --imgsz 640 --device cuda
"""

# ============================================================================
# DEFAULT CONFIGURATION
# Modify these defaults if you want to run the script without arguments
# ============================================================================

DEFAULT_CONFIG = {
    'model_path': r"C:\WormWatcher\NeuralNets\yolo8_2025-12-31_detectmode.pt",          # Path to trained YOLOv8 model (.pt file) - REQUIRED if not provided via args
    'dataset_path': r"I:\Tau worm mask RCNN\Model 5 - YOLO8 Detect\_YOLO8_Training\worm_detection\yolov8_detect\dataset\train\dataset.yaml",        # Path to dataset YAML file or dataset directory - REQUIRED if not provided via args
    'imgsz': 512,                # Image size for evaluation
    'device': None,              # Device: None=auto, "cuda", "cpu", or specific GPU like "0"
    'workers': 0,                # Number of data loading workers (0 for single-threaded, reduces memory issues)
    'task_mode': "detect",           # Task mode: None=auto-detect, "seg" (segmentation), or "detect" (detection)
    'save_json': False,          # Save metrics to JSON file
    'save_plots': True,          # Save PR curves and other plots
    'output_dir': r"I:\Tau worm mask RCNN\Model 5 - YOLO8 Detect\_YOLO8_Training\worm_detection\yolov8_detect\dataset\train\evaluation",          # Output directory (None = same directory as model)
}


import os
import sys
import argparse
import json
import csv
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple

# Error handling for imports
try:
    import numpy as np
except ImportError:
    print("=" * 80)
    print("❌ NumPy import error detected!")
    print("=" * 80)
    print("This is usually caused by a corrupted or incompatible NumPy installation.")
    print("\nTo fix, run in your terminal:")
    print("  pip uninstall numpy -y")
    print("  pip install numpy")
    print("=" * 80)
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError as e:
    if "numpy.core.multiarray" in str(e) or "numpy" in str(e).lower():
        print("=" * 80)
        print("❌ NumPy compatibility error with Ultralytics!")
        print("=" * 80)
        print("This is usually caused by a NumPy version mismatch.")
        print("\nTo fix, try:")
        print("  pip install --upgrade numpy ultralytics")
        print("=" * 80)
        sys.exit(1)
    else:
        raise




def _create_enhanced_curves(metrics, output_dir: str, task_mode: str):
    """
    Create enhanced PR, Precision, and Recall curves with annotations.
    
    Args:
        metrics: YOLO validation metrics object
        output_dir: Directory to save enhanced plots
        task_mode: "seg" or "detect"
    """
    curves_created = False
    
    try:
        # YOLO stores curve data in metrics.curves which is a dict
        # Access the curve data structure
        if not hasattr(metrics, 'curves'):
            print("⚠️  Warning: Could not access curve data from metrics object")
            print("   Available attributes:", [attr for attr in dir(metrics) if not attr.startswith('_')])
            return False
        
        curves = metrics.curves
        
        if not isinstance(curves, dict):
            print(f"⚠️  Warning: metrics.curves is not a dict (type: {type(curves)})")
            return False
        
        # YOLO curves structure: curves is a dict with keys like 'box', 'seg'
        # Each contains px, py (precision), rx, ry (recall), and pr (PR curve data)
        
        # Process box curves (available for both detection and segmentation)
        if hasattr(metrics, 'box') and 'box' in curves:
            if _plot_enhanced_curves_from_data(curves['box'], output_dir, 'box'):
                curves_created = True
        
        # Process segmentation curves (only for segmentation models)
        if task_mode == "seg" and hasattr(metrics, 'seg') and 'seg' in curves:
            if _plot_enhanced_curves_from_data(curves['seg'], output_dir, 'seg'):
                curves_created = True
        
        return curves_created
            
    except Exception as e:
        print(f"⚠️  Warning: Could not create enhanced curves: {e}")
        import traceback
        traceback.print_exc()
        return False


def _plot_enhanced_curves_from_data(curve_data, output_dir: str, curve_type: str):
    """Create enhanced plots from YOLO curve data
    
    Returns:
        bool: True if plots were created successfully, False otherwise
    """
    
    try:
        # YOLO curve_data is typically a dict or object with attributes:
        # - px, py: Precision curve (x=confidence, y=precision)
        # - rx, ry: Recall curve (x=confidence, y=recall)  
        # - pr: PR curve (numpy array with shape [n, 2] where columns are [recall, precision])
        
        # Extract data based on structure
        if isinstance(curve_data, dict):
            px = curve_data.get('px', None)
            py = curve_data.get('py', None)
            rx = curve_data.get('rx', None)
            ry = curve_data.get('ry', None)
            pr_curve = curve_data.get('pr', None)
        elif hasattr(curve_data, 'px'):
            # Object with attributes
            px = getattr(curve_data, 'px', None)
            py = getattr(curve_data, 'py', None)
            rx = getattr(curve_data, 'rx', None)
            ry = getattr(curve_data, 'ry', None)
            pr_curve = getattr(curve_data, 'pr', None)
        else:
            print(f"⚠️  Warning: Unknown curve data format for {curve_type} (type: {type(curve_data)})")
            if hasattr(curve_data, '__dict__'):
                print(f"   Available attributes: {list(curve_data.__dict__.keys())}")
            return False
        
        # Convert to numpy arrays if needed
        if px is not None and not isinstance(px, np.ndarray):
            px = np.array(px)
        if py is not None and not isinstance(py, np.ndarray):
            py = np.array(py)
        if rx is not None and not isinstance(rx, np.ndarray):
            rx = np.array(rx)
        if ry is not None and not isinstance(ry, np.ndarray):
            ry = np.array(ry)
        
        # Check if we have any valid data to plot
        has_precision = px is not None and py is not None and len(px) > 0 and len(py) > 0
        has_recall = rx is not None and ry is not None and len(rx) > 0 and len(ry) > 0
        has_pr = pr_curve is not None
        
        if not (has_precision or has_recall or has_pr):
            print(f"⚠️  Warning: No valid curve data found for {curve_type}")
            print(f"   px/py: {has_precision}, rx/ry: {has_recall}, pr: {has_pr}")
            return False
        
        # Create plots directory
        plots_dir = os.path.join(output_dir, 'enhanced_curves')
        os.makedirs(plots_dir, exist_ok=True)
        
        plots_created = 0
        
        # Plot Precision vs Confidence if available
        if has_precision:
            _plot_precision_curve(px, py, plots_dir, curve_type)
            plots_created += 1
        
        # Plot Recall vs Confidence if available
        if has_recall:
            _plot_recall_curve(rx, ry, plots_dir, curve_type)
            plots_created += 1
        
        # Plot PR curve if available
        if has_pr:
            _plot_pr_curve(pr_curve, plots_dir, curve_type)
            plots_created += 1
        
        return plots_created > 0
            
    except Exception as e:
        print(f"⚠️  Warning: Could not plot {curve_type} curves: {e}")
        import traceback
        traceback.print_exc()
        return False


def _plot_precision_curve(px, py, plots_dir: str, curve_type: str):
    """Plot Precision vs Confidence with 0.95 precision line"""
    plt.figure(figsize=(10, 6))
    plt.plot(px, py, 'b-', linewidth=2, label='Precision')
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision vs Confidence ({curve_type})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.0])
    plt.xlim([0, 1.0])
    
    # Add horizontal line at 0.95 precision
    plt.axhline(y=0.95, color='r', linestyle='--', linewidth=1.5, label='P = 0.95', alpha=0.7)
    
    # Find intersection with 0.95 line
    if len(py) > 0:
        # Find closest point to 0.95
        idx_095 = np.argmin(np.abs(py - 0.95))
        if idx_095 < len(px):
            conf_095 = px[idx_095]
            prec_095 = py[idx_095]
            plt.plot(conf_095, prec_095, 'ro', markersize=8, zorder=5)
            plt.annotate(f'P={prec_095:.3f}\n@Conf={conf_095:.3f}',
                        xy=(conf_095, prec_095),
                        xytext=(conf_095 + 0.1, prec_095 + 0.05),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=10)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'precision_curve_{curve_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    csv_path = os.path.join(plots_dir, f'precision_curve_{curve_type}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Confidence', 'Precision'])
        for conf, prec in zip(px, py):
            writer.writerow([f'{conf:.6f}', f'{prec:.6f}'])


def _plot_recall_curve(rx, ry, plots_dir: str, curve_type: str):
    """Plot Recall vs Confidence with 0.95 recall line"""
    plt.figure(figsize=(10, 6))
    plt.plot(rx, ry, 'g-', linewidth=2, label='Recall')
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title(f'Recall vs Confidence ({curve_type})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.0])
    plt.xlim([0, 1.0])
    
    # Add horizontal line at 0.95 recall
    plt.axhline(y=0.95, color='r', linestyle='--', linewidth=1.5, label='R = 0.95', alpha=0.7)
    
    # Find intersection with 0.95 line
    if len(ry) > 0:
        # Find closest point to 0.95
        idx_095 = np.argmin(np.abs(ry - 0.95))
        if idx_095 < len(rx):
            conf_095 = rx[idx_095]
            recall_095 = ry[idx_095]
            plt.plot(conf_095, recall_095, 'ro', markersize=8, zorder=5)
            plt.annotate(f'R={recall_095:.3f}\n@Conf={conf_095:.3f}',
                        xy=(conf_095, recall_095),
                        xytext=(conf_095 + 0.1, recall_095 + 0.05),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        fontsize=10)
    
    plt.legend(fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'recall_curve_{curve_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    csv_path = os.path.join(plots_dir, f'recall_curve_{curve_type}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Confidence', 'Recall'])
        for conf, recall in zip(rx, ry):
            writer.writerow([f'{conf:.6f}', f'{recall:.6f}'])


def _plot_pr_curve(pr_curve, plots_dir: str, curve_type: str):
    """Plot PR curve with optimal point annotation (closest to 1,1)"""
    # Handle different PR curve formats
    recall = None
    precision = None
    
    if isinstance(pr_curve, np.ndarray):
        # YOLO PR curve is typically [n, 2] array where columns are [recall, precision]
        if pr_curve.ndim == 2 and pr_curve.shape[1] >= 2:
            recall = pr_curve[:, 0]  # First column is recall
            precision = pr_curve[:, 1]  # Second column is precision
        elif pr_curve.ndim == 1:
            print(f"⚠️  Warning: 1D PR curve array - may need different handling for {curve_type}")
            return
        else:
            print(f"⚠️  Warning: Unexpected PR curve array shape {pr_curve.shape} for {curve_type}")
            return
    elif isinstance(pr_curve, dict):
        recall = pr_curve.get('recall', None)
        precision = pr_curve.get('precision', None)
        if recall is None or precision is None:
            print(f"⚠️  Warning: Could not extract PR data from dict for {curve_type}")
            return
    elif hasattr(pr_curve, 'recall') and hasattr(pr_curve, 'precision'):
        # Object with attributes
        recall = getattr(pr_curve, 'recall', None)
        precision = getattr(pr_curve, 'precision', None)
    else:
        print(f"⚠️  Warning: Unknown PR curve format for {curve_type}: {type(pr_curve)}")
        return
    
    # Convert to numpy arrays if needed
    if recall is not None and not isinstance(recall, np.ndarray):
        recall = np.array(recall)
    if precision is not None and not isinstance(precision, np.ndarray):
        precision = np.array(precision)
    
    if recall is None or precision is None or len(recall) == 0 or len(precision) == 0:
        print(f"⚠️  Warning: Empty or invalid PR curve data for {curve_type}")
        return
    
    plt.figure(figsize=(10, 10))
    plt.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve ({curve_type})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.0])
    plt.xlim([0, 1.0])
    
    # Add vertical line at 0.95 recall
    plt.axvline(x=0.95, color='g', linestyle='--', linewidth=1.5, label='R = 0.95', alpha=0.7)
    
    # Add horizontal line at 0.95 precision
    plt.axhline(y=0.95, color='r', linestyle='--', linewidth=1.5, label='P = 0.95', alpha=0.7)
    
    # Find point closest to (1, 1) - optimal point
    if len(recall) > 0 and len(precision) > 0:
        # Calculate distance from each point to (1, 1)
        distances = np.sqrt((1 - recall)**2 + (1 - precision)**2)
        optimal_idx = np.argmin(distances)
        
        opt_recall = recall[optimal_idx]
        opt_precision = precision[optimal_idx]
        
        # Plot optimal point
        plt.plot(opt_recall, opt_precision, 'ko', markersize=12, zorder=5, label='Optimal Point')
        
        # Annotate optimal point
        plt.annotate(f'Optimal\nP={opt_precision:.3f}\nR={opt_recall:.3f}',
                    xy=(opt_recall, opt_precision),
                    xytext=(opt_recall - 0.15, opt_precision - 0.15),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2, alpha=0.7),
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='black', alpha=0.9),
                    fontsize=11, fontweight='bold',
                    ha='right')
    
    plt.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f'pr_curve_{curve_type}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data to CSV
    csv_path = os.path.join(plots_dir, f'pr_curve_{curve_type}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Recall', 'Precision'])
        for r, p in zip(recall, precision):
            writer.writerow([f'{r:.6f}', f'{p:.6f}'])
    
    # Also save optimal point info if it was calculated during plotting
    if len(recall) > 0 and len(precision) > 0:
        # Recalculate optimal point for CSV export (already calculated above but we need the index)
        distances = np.sqrt((1 - recall)**2 + (1 - precision)**2)
        optimal_idx = np.argmin(distances)
        opt_recall = recall[optimal_idx]
        opt_precision = precision[optimal_idx]
        
        optimal_csv_path = os.path.join(plots_dir, f'pr_curve_{curve_type}_optimal_point.csv')
        with open(optimal_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Optimal_Recall', f'{opt_recall:.6f}'])
            writer.writerow(['Optimal_Precision', f'{opt_precision:.6f}'])
            writer.writerow(['Distance_from_1_1', f'{distances[optimal_idx]:.6f}'])


def _get_device(requested_device):
    """Get device with error handling for memory issues"""
    if requested_device is None:
        try:
            return "cuda" if torch.cuda.is_available() else "cpu"
        except OSError as e:
            if "paging file" in str(e).lower() or "1455" in str(e):
                print("\n" + "!" * 80)
                print("⚠️  WARNING: CUDA loading failed due to insufficient virtual memory!")
                print("   Falling back to CPU mode.")
                print("!" * 80 + "\n")
                return "cpu"
            else:
                raise
    else:
        return requested_device


def evaluate_yolo8_model(
    model_path: str,
    dataset_path: str,
    imgsz: int = 640,
    device: Optional[str] = None,
    workers: int = 0,
    save_json: bool = False,
    save_plots: bool = True,
    output_dir: Optional[str] = None,
    task_mode: Optional[str] = None
) -> Dict:
    """
    Evaluate a YOLOv8 model on a dataset.
    
    Args:
        model_path: Path to the trained YOLOv8 model (.pt file)
        dataset_path: Path to the dataset YAML file (should contain 'val:' path pointing to validation set)
                     Note: Use the ROOT dataset.yaml file (not one inside train/ folder) which defines train/val/test splits
        imgsz: Image size for evaluation (default: 640)
        device: Device to use ("cuda", "cpu", or None for auto)
        workers: Number of data loading workers (0 for single-threaded, useful for memory-constrained systems)
        save_json: Whether to save metrics to JSON file
        save_plots: Whether to save PR curves and other plots
        output_dir: Output directory for results (None = same directory as model)
        task_mode: Task mode - "seg" (segmentation) or "detect" (detection), or None to auto-detect
    
    Note:
        YOLO's val() method automatically uses the 'val:' path specified in the dataset YAML file.
        Make sure your dataset_path points to a YAML file that includes a 'val:' entry.
        
        **Confidence Threshold for Precision/Recall:**
        The precision and recall metrics reported are computed at a default confidence threshold of 0.25.
        This means only detections with confidence >= 0.25 are considered when calculating these metrics.
        The mAP metrics (mAP50, mAP50-95) are computed across all confidence thresholds and are threshold-independent.
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Get device
    device = _get_device(device)
    
    # Determine task mode from model if not provided
    if task_mode is None:
        # Try to infer from model path or load model to check
        print("🔍 Auto-detecting task mode from model...")
        try:
            temp_model = YOLO(model_path)
            task_mode = temp_model.task
            print(f"   Detected task mode: {task_mode}")
        except:
            # Default to detection if can't determine
            task_mode = "detect"
            print(f"   Could not auto-detect, defaulting to: {task_mode}")
    
    if task_mode not in ["seg", "detect"]:
        raise ValueError(f"Invalid task_mode: {task_mode}. Must be 'seg' or 'detect'")
    
    task_name = "Segmentation" if task_mode == "seg" else "Detection"
    
    print("\n" + "=" * 80)
    print(f"YOLOv8 {task_name} Model Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Image size: {imgsz}")
    print(f"Device: {device}")
    print(f"Task mode: {task_mode}")
    print("=" * 80 + "\n")
    
    # Load model
    print("📦 Loading model...")
    model = YOLO(model_path)
    model.to(device)
    print("✅ Model loaded successfully")
    
    # Set up output directory
    if output_dir is None:
        # Default to same directory as model
        model_dir = os.path.dirname(os.path.abspath(model_path))
        output_dir = os.path.join(model_dir, "_evaluation_results")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Verify dataset YAML has val split (optional check)
    if dataset_path.endswith('.yaml') or dataset_path.endswith('.yml'):
        try:
            with open(dataset_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                if 'val' not in yaml_data:
                    print(f"⚠️  Warning: Dataset YAML file does not contain a 'val:' entry.")
                    print(f"   The evaluation will attempt to proceed, but may fail if no validation data is available.")
                else:
                    val_path = yaml_data.get('val', '')
                    print(f"📊 Validation split found in YAML: {val_path}")
        except Exception as e:
            # Non-critical - just continue if YAML parsing fails
            print(f"ℹ️  Could not verify YAML structure: {e}")
    
    # Run evaluation
    print("\n🔍 Running evaluation...")
    try:
        val_params = {
            'data': dataset_path,
            'imgsz': imgsz,
            'device': device,
            'workers': workers,
            'save_json': save_json,
            'plots': save_plots,
            'project': output_dir,
            'name': 'eval',
            'verbose': True,
        }
        
        metrics = model.val(**val_params)
        
        # Get the actual directory where YOLO saved the plots
        yolo_plots_dir = metrics.save_dir if hasattr(metrics, 'save_dir') else os.path.join(output_dir, 'eval')
        
        # Create enhanced curves with annotations if plots are enabled
        if save_plots:
            print("\n📊 Creating enhanced PR curves with annotations...")
            try:
                enhanced_curves_path = os.path.join(yolo_plots_dir, 'enhanced_curves')
                _create_enhanced_curves(metrics, yolo_plots_dir, task_mode)
                # Verify directory was actually created
                if os.path.exists(enhanced_curves_path):
                    print(f"✅ Enhanced curves (PNG + CSV) saved to: {enhanced_curves_path}")
                else:
                    print(f"⚠️  Warning: Enhanced curves directory was not created. Curve data may not be accessible.")
                    print(f"   Expected path: {enhanced_curves_path}")
            except Exception as e:
                print(f"⚠️  Warning: Could not create enhanced curves: {e}")
                import traceback
                traceback.print_exc()
        
        # Extract metrics
        results = {}
        
        # Try to get confidence threshold from metrics (YOLO default is 0.25)
        confidence_threshold = 0.25  # YOLOv8 default
        if hasattr(metrics, 'conf') and metrics.conf is not None:
            confidence_threshold = float(metrics.conf)
        elif hasattr(metrics, 'box') and hasattr(metrics.box, 'conf') and metrics.box.conf is not None:
            confidence_threshold = float(metrics.box.conf)
        
        # Box metrics (available for both detection and segmentation)
        if hasattr(metrics, 'box'):
            results['box'] = {
                'map50': float(metrics.box.map50),
                'map': float(metrics.box.map),  # mAP50-95
                'precision': float(metrics.box.p) if hasattr(metrics.box, 'p') else None,
                'recall': float(metrics.box.r) if hasattr(metrics.box, 'r') else None,
                'f1_score': float(metrics.box.f1) if hasattr(metrics.box, 'f1') else None,
                'confidence_threshold': confidence_threshold,  # Store threshold used
            }
        
        # Segmentation metrics (only for segmentation models)
        if task_mode == "seg" and hasattr(metrics, 'seg'):
            results['seg'] = {
                'map50': float(metrics.seg.map50),
                'map': float(metrics.seg.map),  # mAP50-95
                'precision': float(metrics.seg.p) if hasattr(metrics.seg, 'p') else None,
                'recall': float(metrics.seg.r) if hasattr(metrics.seg, 'r') else None,
                'f1_score': float(metrics.seg.f1) if hasattr(metrics.seg, 'f1') else None,
                'confidence_threshold': confidence_threshold,  # Store threshold used
            }
        
        # Additional metadata
        results['metadata'] = {
            'model_path': model_path,
            'dataset_path': dataset_path,
            'task_mode': task_mode,
            'image_size': imgsz,
            'device': device,
            'results_dir': metrics.save_dir if hasattr(metrics, 'save_dir') else output_dir,
            'confidence_threshold': confidence_threshold,  # Confidence threshold for P/R metrics
        }
        
        # Print results
        print("\n" + "=" * 80)
        print("✅ Evaluation Complete!")
        print("=" * 80)
        print(f"\n📊 Box Detection Metrics:")
        print(f"  mAP50: {results['box']['map50']:.4f}")
        print(f"  mAP50-95: {results['box']['map']:.4f}")
        if results['box']['precision'] is not None:
            conf_thresh = results['box'].get('confidence_threshold', confidence_threshold)
            print(f"  Precision: {results['box']['precision']:.4f} (at conf={conf_thresh:.2f})")
        if results['box']['recall'] is not None:
            conf_thresh = results['box'].get('confidence_threshold', confidence_threshold)
            print(f"  Recall: {results['box']['recall']:.4f} (at conf={conf_thresh:.2f})")
        if results['box']['f1_score'] is not None:
            print(f"  F1-Score: {results['box']['f1_score']:.4f}")
        
        if task_mode == "seg" and 'seg' in results:
            print(f"\n📊 Segmentation Metrics:")
            print(f"  mAP50-seg: {results['seg']['map50']:.4f}")
            print(f"  mAP50-95-seg: {results['seg']['map']:.4f}")
            if results['seg']['precision'] is not None:
                conf_thresh = results['seg'].get('confidence_threshold', confidence_threshold)
                print(f"  Precision-seg: {results['seg']['precision']:.4f} (at conf={conf_thresh:.2f})")
            if results['seg']['recall'] is not None:
                conf_thresh = results['seg'].get('confidence_threshold', confidence_threshold)
                print(f"  Recall-seg: {results['seg']['recall']:.4f} (at conf={conf_thresh:.2f})")
            if results['seg']['f1_score'] is not None:
                print(f"  F1-Score-seg: {results['seg']['f1_score']:.4f}")
        
        print(f"\n📁 Results saved to: {results['metadata']['results_dir']}")
        if save_plots:
            print("   (Includes PR curves and other evaluation plots)")
            enhanced_curves_dir = os.path.join(yolo_plots_dir, 'enhanced_curves')
            if os.path.exists(enhanced_curves_dir):
                print(f"   Enhanced curves with annotations saved to: {enhanced_curves_dir}")
        print("=" * 80 + "\n")
        
        # Save metrics to JSON if requested
        if save_json:
            json_path = os.path.join(output_dir, 'eval_metrics.json')
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📄 Metrics saved to: {json_path}")
        
        return results
        
    except (OSError, MemoryError) as e:
        if "paging file" in str(e).lower() or "1455" in str(e) or "memory" in str(e).lower():
            print("\n" + "!" * 80)
            print("⚠️  Evaluation failed due to memory constraints!")
            print("   Try reducing workers or image size:")
            print(f"   workers=0, imgsz={imgsz // 2}")
            print("!" * 80 + "\n")
            raise
        else:
            raise


def main():
    """Command-line interface for evaluation"""
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 model on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (using command-line args)
  python -m YOLO8.evaluation_yolo8 model.pt dataset.yaml
  
  # Using hardcoded defaults (if DEFAULT_CONFIG is set at top of script)
  python -m YOLO8.evaluation_yolo8
  
  # Override defaults with command-line args
  python -m YOLO8.evaluation_yolo8 --imgsz 512 --device cuda --save-json
  
  # Custom output directory
  python -m YOLO8.evaluation_yolo8 model.pt dataset.yaml --output-dir ./results
        """
    )
    
    # Use defaults from DEFAULT_CONFIG, but allow None for required args
    parser.add_argument(
        'model_path',
        type=str,
        nargs='?',
        default=DEFAULT_CONFIG.get('model_path'),
        help=f'Path to the trained YOLOv8 model (.pt file) (default: {DEFAULT_CONFIG.get("model_path")})'
    )
    
    parser.add_argument(
        'dataset_path',
        type=str,
        nargs='?',
        default=DEFAULT_CONFIG.get('dataset_path'),
        help=f'Path to the dataset YAML file or dataset directory (default: {DEFAULT_CONFIG.get("dataset_path")})'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=DEFAULT_CONFIG.get('imgsz', 640),
        help=f'Image size for evaluation (default: {DEFAULT_CONFIG.get("imgsz", 640)})'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=DEFAULT_CONFIG.get('device'),
        help=f'Device to use: "cuda", "cpu", or None for auto (default: {DEFAULT_CONFIG.get("device")})'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_CONFIG.get('workers', 0),
        help=f'Number of data loading workers (default: {DEFAULT_CONFIG.get("workers", 0)})'
    )
    
    parser.add_argument(
        '--task',
        type=str,
        default=DEFAULT_CONFIG.get('task_mode'),
        choices=["seg", "detect"],
        help=f'Task mode: "seg" (segmentation) or "detect" (detection). Auto-detected if not provided. (default: {DEFAULT_CONFIG.get("task_mode")})'
    )
    
    parser.add_argument(
        '--save-json',
        action='store_true',
        help=f'Save metrics to JSON file (default from DEFAULT_CONFIG: {DEFAULT_CONFIG.get("save_json", False)})'
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip saving plots (PR curves, etc.) - overrides DEFAULT_CONFIG'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_CONFIG.get('output_dir'),
        help=f'Output directory for results (default: {DEFAULT_CONFIG.get("output_dir")})'
    )
    
    args = parser.parse_args()
    
    # Validate required arguments
    if args.model_path is None:
        print("❌ Error: model_path is required")
        print("   Provide it as a command-line argument or set DEFAULT_CONFIG['model_path'] at the top of the script")
        return 1
    
    if args.dataset_path is None:
        print("❌ Error: dataset_path is required")
        print("   Provide it as a command-line argument or set DEFAULT_CONFIG['dataset_path'] at the top of the script")
        return 1
    
    try:
        # Apply defaults from DEFAULT_CONFIG for options that weren't explicitly set
        # For boolean flags, use DEFAULT_CONFIG if flag wasn't provided
        save_json = args.save_json if args.save_json else DEFAULT_CONFIG.get('save_json', False)
        save_plots = DEFAULT_CONFIG.get('save_plots', True) if not args.no_plots else False
        
        results = evaluate_yolo8_model(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            imgsz=args.imgsz,
            device=args.device,
            workers=args.workers,
            save_json=save_json,
            save_plots=save_plots,
            output_dir=args.output_dir,
            task_mode=args.task
        )
        
        # Exit with success code
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

