"""
YOLOv8 Segmentation - Ultralytics YOLOv8 for worm tracking
Compatible with existing Mask R-CNN interface
"""

from .YOLO8_Segment import YOLO8_Segment
from .YOLO8_Helper import loadYOLO8, create_yolo8_model, get_recommended_model_size
from .data_augmentation import DataAugmentation

__all__ = ['YOLO8_Segment', 'loadYOLO8', 'create_yolo8_model', 'get_recommended_model_size', 'DataAugmentation']

