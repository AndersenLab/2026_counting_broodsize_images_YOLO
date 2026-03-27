"""
Example usage of YOLOv8 for worm segmentation
Demonstrates both training and inference workflows
"""

import cv2
import numpy as np
try:
    from .YOLO8_Segment import YOLO8_Segment, loadYOLO8
except ImportError:
    from YOLO8_Segment import YOLO8_Segment, loadYOLO8


def example_inference():
    """Example of using YOLOv8 for inference"""
    
    print("=" * 80)
    print("YOLOv8 Inference Example")
    print("=" * 80)
    
    # Load model
    model_path = "path/to/your/trained/model.pt"
    model = loadYOLO8(
        model_path=model_path,
        num_classes=2,  # background + worm
        use_cuda=True
    )
    
    # Load test image
    img_path = "path/to/test/image.png"
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Could not load image: {img_path}")
        return
    
    print(f"✅ Loaded image: {img_path}")
    print(f"   Image shape: {img.shape}")
    
    # Segmentation parameters (optimized for YOLOv8)
    seg_params = {
        "pix conf thresh": 0.5,      # Pixel confidence threshold
        "overlap thresh": 0.5,       # Overlap threshold for NMS
        "min object area": 80,       # Minimum object area in pixels
        "object conf thresh": 0.25,  # Object confidence threshold (lower for YOLO)
        "iou thresh": 0.45,          # IoU threshold for NMS
    }
    
    # Run segmentation
    print("\n🔍 Running segmentation...")
    masks, masks_thresh, weighted, weighted_touch, touch_count = YOLO8_Segment(
        model, img, seg_params
    )
    
    # Print results
    n_detections = masks_thresh.shape[3] if len(masks_thresh.shape) == 4 else 0
    print(f"\n✅ Segmentation complete!")
    print(f"   Detections: {n_detections}")
    print(f"   Touch counts: {touch_count}")
    
    # Visualize results
    visualize_results(img, masks_thresh, weighted)
    
    return masks, masks_thresh, weighted, weighted_touch, touch_count


def visualize_results(img, masks_thresh, weighted):
    """Visualize segmentation results"""
    
    import matplotlib.pyplot as plt
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Weighted mask overlay
    overlay = img.copy()
    weighted_colored = cv2.applyColorMap((weighted * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay, 0.6, weighted_colored, 0.4, 0)
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Weighted Mask Overlay")
    axes[1].axis('off')
    
    # Individual masks
    if masks_thresh.shape[3] > 0:
        all_masks = np.sum(masks_thresh, axis=3)
        axes[2].imshow(all_masks.squeeze(), cmap='hot')
        axes[2].set_title(f"All Masks ({masks_thresh.shape[3]} detections)")
    else:
        axes[2].imshow(np.zeros_like(img[:,:,0]), cmap='gray')
        axes[2].set_title("No Detections")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig("yolo8_results.png", dpi=150, bbox_inches='tight')
    print(f"\n💾 Visualization saved to: yolo8_results.png")
    plt.show()


def example_training_command():
    """Example command for training"""
    
    print("=" * 80)
    print("YOLOv8 Training Example Command")
    print("=" * 80)
    
    command = """
python -m YOLO8.train_yolo8 \\
    --train_json path/to/train_annotations.json \\
    --train_images path/to/train_images \\
    --val_json path/to/val_annotations.json \\
    --val_images path/to/val_images \\
    --model_size n \\
    --epochs 100 \\
    --batch_size 16 \\
    --imgsz 640 \\
    --learning_rate 0.01 \\
    --output_dir ./yolo8_output
"""
    
    print(command)
    print("\nThis command will:")
    print("  1. Convert VGG JSON annotations to YOLO format")
    print("  2. Train YOLOv8-nano segmentation model")
    print("  3. Save checkpoints and logs to ./yolo8_output")
    print("  4. Validate during training")


def example_integration_with_multiwellanysis():
    """Example of integrating YOLOv8 with MultiWellAnalysis"""
    
    print("=" * 80)
    print("Integration with MultiWellAnalysis")
    print("=" * 80)
    
    integration_code = """
# In MultiWellAnalysis.py, replace the imports:

# OLD:
# from RCNN_Segment import RCNN_Segment
# from RCNN_Helper import loadRCNN

# NEW:
from YOLO8.YOLO8_Segment import YOLO8_Segment as RCNN_Segment
from YOLO8.YOLO8_Helper import loadYOLO8 as loadRCNN

# The rest of the code remains unchanged!
# The interface is fully compatible.
"""
    
    print(integration_code)
    print("\n✅ No other changes needed - YOLOv8 is a drop-in replacement!")


def example_model_size_selection():
    """Example of selecting appropriate model size"""
    
    print("=" * 80)
    print("Model Size Selection Guide")
    print("=" * 80)
    
    guide = """
For worm segmentation, recommended model sizes:

1. YOLOv8-nano (n) - RECOMMENDED
   - Fastest inference
   - Good for small objects (worms)
   - Best for real-time applications
   - Use with: --model_size n

2. YOLOv8-small (s)
   - Balanced speed/accuracy
   - Better accuracy than nano
   - Still very fast
   - Use with: --model_size s

3. YOLOv8-medium (m) and larger
   - Higher accuracy
   - Slower inference
   - Use only if nano/small don't meet accuracy requirements
   - Use with: --model_size m (or l, x)

For high-resolution images (640px+), nano or small usually work best!
"""
    
    print(guide)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("YOLOv8 Usage Examples")
    print("=" * 80 + "\n")
    
    # Show training command example
    example_training_command()
    
    print("\n" + "-" * 80 + "\n")
    
    # Show integration example
    example_integration_with_multiwellanysis()
    
    print("\n" + "-" * 80 + "\n")
    
    # Show model size selection
    example_model_size_selection()
    
    print("\n" + "-" * 80 + "\n")
    
    # Uncomment to run inference example (requires trained model)
    # example_inference()
    
    print("\n" + "=" * 80)
    print("For more information, see README.md")
    print("=" * 80 + "\n")

