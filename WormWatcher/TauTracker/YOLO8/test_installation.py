"""
Test script to verify YOLOv8 installation
Run this to check if all dependencies are properly installed
"""

import sys

def test_imports():
    """Test if all required modules can be imported"""
    
    print("=" * 80)
    print("Testing YOLOv8 Installation")
    print("=" * 80)
    
    errors = []
    
    # Test basic dependencies
    print("\n1. Testing basic dependencies...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__}")
        print(f"      CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"      CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
        print(f"   ❌ PyTorch not found: {e}")
    
    try:
        import cv2
        print(f"   ✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"OpenCV: {e}")
        print(f"   ❌ OpenCV not found: {e}")
    
    try:
        import numpy as np
        print(f"   ✅ NumPy {np.__version__}")
    except ImportError as e:
        errors.append(f"NumPy: {e}")
        print(f"   ❌ NumPy not found: {e}")
    
    # Test Ultralytics YOLOv8
    print("\n2. Testing Ultralytics YOLOv8...")
    try:
        from ultralytics import YOLO
        print(f"   ✅ Ultralytics YOLO installed")
        
        # Try to create a model (this will download if needed)
        try:
            # Just test import, don't download
            print(f"   ✅ YOLO module available")
        except Exception as e:
            errors.append(f"YOLO model creation: {e}")
            print(f"   ⚠️  YOLO available but model test failed: {e}")
    except ImportError as e:
        errors.append(f"Ultralytics: {e}")
        print(f"   ❌ Ultralytics not found: {e}")
        print(f"      Install with: pip install ultralytics")
    
    # Test local modules
    print("\n3. Testing local YOLO8 modules...")
    try:
        try:
            from .YOLO8_Helper import loadYOLO8, create_yolo8_model
        except ImportError:
            from YOLO8_Helper import loadYOLO8, create_yolo8_model
        print(f"   ✅ YOLO8_Helper imported")
    except ImportError as e:
        errors.append(f"YOLO8_Helper: {e}")
        print(f"   ❌ YOLO8_Helper not found: {e}")
    
    try:
        try:
            from .YOLO8_Segment import YOLO8_Segment
        except ImportError:
            from YOLO8_Segment import YOLO8_Segment
        print(f"   ✅ YOLO8_Segment imported")
    except ImportError as e:
        errors.append(f"YOLO8_Segment: {e}")
        print(f"   ❌ YOLO8_Segment not found: {e}")
    
    try:
        try:
            from .dataset_converter import convert_vgg_to_yolo
        except ImportError:
            from dataset_converter import convert_vgg_to_yolo
        print(f"   ✅ dataset_converter imported")
    except ImportError as e:
        errors.append(f"dataset_converter: {e}")
        print(f"   ❌ dataset_converter not found: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    if len(errors) == 0:
        print("✅ All tests passed! YOLOv8 is ready to use.")
        print("=" * 80)
        return True
    else:
        print("❌ Some tests failed. Please fix the following issues:")
        print("=" * 80)
        for error in errors:
            print(f"   - {error}")
        print("\nSee README.md for installation instructions.")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

