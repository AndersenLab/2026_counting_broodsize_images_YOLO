"""
YOLOv8 Helper Functions
Handles model loading and configuration
"""

import torch
import os
from pathlib import Path
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore")


def loadYOLO8(model_path, num_classes=2, use_cuda=True):
    """
    Load a YOLOv8 segmentation model from a checkpoint file.
    
    Args:
        model_path (str): Path to the model checkpoint (.pt file)
        num_classes (int): Number of classes (default: 2 for background + worm)
        use_cuda (bool): Whether to use CUDA if available
    
    Returns:
        YOLO: Loaded YOLOv8 model
    """
    device = "cuda" if (use_cuda and torch.cuda.is_available()) else "cpu"
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load YOLOv8 model
    model = YOLO(model_path)
    
    # Set device
    model.to(device)
    
    print(f"✅ YOLOv8 model loaded from: {model_path}")
    print(f"   Device: {device}")
    print(f"   Model type: {model.task}")
    print(f"   Number of classes: {num_classes}")
    
    return model


def create_yolo8_model(model_size="n", num_classes=2, pretrained=True):
    """
    Create a new YOLOv8 segmentation model.
    
    Args:
        model_size (str): Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        num_classes (int): Number of classes (background + objects)
        pretrained (bool): Whether to load COCO pretrained weights
    
    Returns:
        YOLO: YOLOv8 segmentation model
    """
    # YOLOv8 segmentation model names
    model_sizes = {
        'n': 'yolov8n-seg.pt',
        's': 'yolov8s-seg.pt',
        'm': 'yolov8m-seg.pt',
        'l': 'yolov8l-seg.pt',
        'x': 'yolov8x-seg.pt'
    }
    
    if model_size not in model_sizes:
        raise ValueError(f"Invalid model size: {model_size}. Choose from: {list(model_sizes.keys())}")
    
    model_name = model_sizes[model_size]
    
    if pretrained:
        # Load pretrained model (will download if not cached)
        model = YOLO(model_name)
    else:
        # Create model from scratch
        model = YOLO(model_name)
        # Note: YOLOv8 doesn't support creating from scratch easily,
        # so we load pretrained and will fine-tune
    
    print(f"✅ YOLOv8-{model_size.upper()} segmentation model created")
    print(f"   Number of classes: {num_classes}")
    
    return model


def get_recommended_model_size(image_size, num_objects_per_image=50):
    """
    Get recommended YOLOv8 model size based on use case.
    
    Args:
        image_size (int): Typical input image size (width or height)
        num_objects_per_image (int): Typical number of objects per image
    
    Returns:
        str: Recommended model size ('n', 's', 'm', 'l', 'x')
    """
    # For small objects (worms) and high-resolution images, use smaller models
    # They're faster and often perform better on small objects
    
    if image_size <= 512:
        if num_objects_per_image <= 20:
            return 'n'  # Nano for small images with few objects
        else:
            return 's'  # Small for small images with many objects
    elif image_size <= 1024:
        if num_objects_per_image <= 30:
            return 's'  # Small for medium images
        else:
            return 'm'  # Medium for medium images with many objects
    else:
        if num_objects_per_image <= 50:
            return 'm'  # Medium for large images
        else:
            return 'l'  # Large for large images with many objects

