# YOLOv8 Segmentation for Worm Tracking

This folder contains a YOLOv8 segmentation implementation for instance segmentation of worms, designed to be a drop-in replacement for Mask R-CNN in the TauTracker codebase.

## Features

- **YOLOv8 Architecture**: Modern, fast, and easy-to-use segmentation model
- **Compatible Interface**: Drop-in replacement for Mask R-CNN (`RCNN_Segment` interface)
- **VGG JSON Support**: Training scripts support VGG Image Annotator (VIA) JSON format
- **Multiple Model Sizes**: Nano (n), Small (s), Medium (m), Large (l), XLarge (x)
- **Fast Inference**: Optimized for speed while maintaining good accuracy
- **Easy Training**: Minimal setup required, automatic dataset conversion

## Why YOLOv8?

### Advantages
- ✅ **Minimal training overhead** - Easy to set up and train
- ✅ **Good small-object performance** - Especially with YOLOv8-n/s and high-resolution input
- ✅ **Fast inference** - Much faster than Mask R-CNN or Mask2Former
- ✅ **Simple API** - Clean, intuitive interface

### Considerations
- ⚠️ **Polygon masks** - Segment masks are polygon-based, not as pixel-perfect as Mask2Former
- ⚠️ **High curvature objects** - Less robust on worm-shaped objects with high curvature compared to Mask2Former

## Installation

### Prerequisites

1. Install PyTorch (with CUDA if available):
```bash
pip install torch torchvision torchaudio
```

2. Install Ultralytics YOLOv8:
```bash
pip install ultralytics
```

3. Install additional dependencies:
```bash
pip install opencv-python numpy scipy
```

That's it! YOLOv8 is much easier to install than Detectron2.

## Usage

### Inference (Compatible with Existing Code)

The YOLOv8 implementation is designed to be a drop-in replacement for Mask R-CNN:

```python
from YOLO8.YOLO8_Segment import YOLO8_Segment, loadYOLO8

# Load model (same interface as loadRCNN)
model = loadYOLO8(
    model_path="path/to/model.pt",
    num_classes=2,  # background + worm
    use_cuda=True
)

# Segment worms (same interface as RCNN_Segment)
seg_params = {
    "pix conf thresh": 0.5,
    "overlap thresh": 0.5,
    "min object area": 80,
    "object conf thresh": 0.25,  # Lower default for YOLO
    "iou thresh": 0.45
}

masks, masks_thresh, weighted, weighted_touch, touch_count = YOLO8_Segment(
    model, img_crop, seg_params
)
```

### Integration with MultiWellAnalysis

To use YOLOv8 in `MultiWellAnalysis.py`, you can modify the imports:

```python
# Replace:
# from RCNN_Segment import RCNN_Segment
# from RCNN_Helper import loadRCNN

# With:
from YOLO8.YOLO8_Segment import YOLO8_Segment as RCNN_Segment
from YOLO8.YOLO8_Helper import loadYOLO8 as loadRCNN
```

### Training

Train a YOLOv8 model on your worm dataset:

```bash
python -m YOLO8.train_yolo8 \
    --train_json path/to/train_annotations.json \
    --train_images path/to/train_images \
    --val_json path/to/val_annotations.json \
    --val_images path/to/val_images \
    --model_size n \
    --epochs 100 \
    --batch_size 16 \
    --imgsz 640 \
    --output_dir ./yolo8_output
```

#### Training Arguments

- `--train_json`: Path to VGG JSON annotation file for training
- `--train_images`: Directory containing training images
- `--val_json`: (Optional) Path to VGG JSON annotation file for validation
- `--val_images`: (Optional) Directory containing validation images
- `--model_size`: Model size (`n`, `s`, `m`, `l`, `x`) - default: `n` (nano, fastest)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size (default: 16)
- `--imgsz`: Input image size (default: 640)
- `--learning_rate`: Initial learning rate (default: 0.01)
- `--device`: Device to use (`cuda`, `cpu`, or GPU ID like `0`)
- `--output_dir`: Output directory for training artifacts

### VGG JSON Format

The training script expects annotations in VGG Image Annotator (VIA) JSON format. Example structure:

```json
{
  "_via_img_metadata": {
    "image1.jpg123456": {
      "filename": "image1.jpg",
      "size": 123456,
      "regions": [
        {
          "shape_attributes": {
            "name": "polygon",
            "all_points_x": [100, 150, 150, 100],
            "all_points_y": [100, 100, 150, 150]
          },
          "region_attributes": {
            "name": "worm"
          }
        }
      ]
    }
  }
}
```

Supported annotation shapes:
- **Polygon**: `"name": "polygon"` with `all_points_x` and `all_points_y`
- **Rectangle**: `"name": "rect"` with `x`, `y`, `width`, `height`
- **Circle**: `"name": "circle"` with `cx`, `cy`, `r`

## Model Sizes

Choose the model size based on your needs:

| Size | Speed | Accuracy | Best For |
|------|-------|----------|----------|
| **n** (nano) | Fastest | Good | Real-time inference, small objects |
| **s** (small) | Fast | Better | Balanced performance |
| **m** (medium) | Medium | Good | Higher accuracy needs |
| **l** (large) | Slow | Better | Maximum accuracy |
| **x** (xlarge) | Slowest | Best | Research, best accuracy |

**Recommendation for worms**: Use `n` (nano) or `s` (small) with high-resolution input (640px or higher) for best small-object detection.

## Architecture Details

### YOLOv8 Advantages

1. **Speed**: Much faster inference than Mask R-CNN or Mask2Former
2. **Simplicity**: Easy to install and use, no complex dependencies
3. **Small Objects**: Good performance on small objects with appropriate model size
4. **Unified Framework**: Single model for detection and segmentation

### Output Format

The `YOLO8_Segment` function returns the same format as `RCNN_Segment`:

- `masks`: Full masks array `[H, W, 1, N]` where N is number of detections
- `masks_thresh`: Thresholded masks `[H, W, 1, N]`
- `weighted`: Weighted sum of all masks `[H, W]`
- `weighted_touch`: Weighted sum of touching masks `[H, W]`
- `touch_count`: List of touch counts for each detection

## Tips for Training

1. **Start with Nano Model**: Use `--model_size n` for faster training and inference
2. **High Resolution**: Use `--imgsz 640` or higher for better small object detection
3. **Data Augmentation**: Enabled by default - helps with generalization
4. **Batch Size**: Adjust based on GPU memory (16 is a good starting point)
5. **Learning Rate**: Default 0.01 works well, but adjust if loss doesn't decrease
6. **Epochs**: 100-200 epochs typically sufficient for fine-tuning

## File Structure

```
YOLO8/
├── __init__.py                 # Package initialization
├── YOLO8_Helper.py             # Model loading and configuration
├── YOLO8_Segment.py            # Segmentation function (compatible with RCNN_Segment)
├── dataset_converter.py        # VGG JSON to YOLO format converter
├── train_yolo8.py              # Training script
├── example_usage.py            # Usage examples
├── test_installation.py        # Installation verification
├── requirements.txt             # Dependencies
└── README.md                   # This file
```

## Comparison with Mask2Former

| Feature | YOLOv8 | Mask2Former |
|---------|--------|-------------|
| **Installation** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐ Complex (requires Detectron2) |
| **Speed** | ⭐⭐⭐⭐⭐ Very Fast | ⭐⭐⭐ Moderate |
| **Accuracy** | ⭐⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Small Objects** | ⭐⭐⭐⭐ Good (with right settings) | ⭐⭐⭐⭐⭐ Excellent |
| **Thin Objects** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent |
| **Training Time** | ⭐⭐⭐⭐⭐ Fast | ⭐⭐⭐ Moderate |
| **Ease of Use** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Moderate |

**Recommendation**: 
- Use **YOLOv8** if you need fast inference, easy setup, and good-enough accuracy
- Use **Mask2Former** if you need maximum accuracy, especially for thin/curved objects

## Troubleshooting

### Import Errors
If you get import errors for Ultralytics:
```bash
pip install --upgrade ultralytics
```

### CUDA Out of Memory
- Reduce batch size (`--batch_size 8` or `--batch_size 4`)
- Reduce image size (`--imgsz 512`)
- Use smaller model (`--model_size n`)

### Poor Detection on Small Objects
- Increase input image size (`--imgsz 640` or `--imgsz 800`)
- Use smaller model size (`--model_size n` or `--model_size s`)
- Lower confidence threshold in inference

### Model Not Loading
- Ensure model path is correct
- Check that model file is a valid YOLOv8 checkpoint (.pt file)
- Verify model was trained for segmentation (not detection)

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/)

## License

This implementation uses Ultralytics YOLOv8, which is licensed under AGPL-3.0.

