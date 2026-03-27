"""
Test script for the enhanced Mask RCNN augmentations (Grayscale Optimized).
This script tests each augmentation function to ensure they work correctly with grayscale images.
"""

import cv2
import numpy as np
import os
import sys
from AugmentMaskRCNNTrainingData import (
    rotateImage, scaleImage, adjustContrast,
    addGaussianNoise, addSaltAndPepperNoise, elasticDeformation,
    addShadows, addMotionBlur, addIntensityVariation, addPerspectiveTransform
)

def create_test_image(size=(256, 256)):
    """Create a test grayscale image with a simple pattern"""
    image = np.zeros((size[0], size[1]), dtype=np.uint8)
    
    # Create a simple pattern (simulating a plate with some features)
    cv2.rectangle(image, (50, 50), (200, 150), 100, -1)
    cv2.circle(image, (128, 128), 30, 150, -1)
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    return image

def create_test_mask(size=(256, 256)):
    """Create a test grayscale mask with a simple shape"""
    mask = np.zeros((size[0], size[1]), dtype=np.uint8)
    
    # Create a simple mask (simulating a worm)
    cv2.ellipse(mask, (128, 128), (40, 10), 45, 0, 360, 255, -1)
    
    return mask

def test_augmentation(func, name, image, mask):
    """Test a single augmentation function"""
    print(f"Testing {name}...")
    
    try:
        # Apply augmentation
        aug_image, aug_mask, aug_list = func(image.copy(), mask.copy(), [])
        
        # Check that output has correct shape
        assert aug_image.shape == image.shape, f"{name}: Image shape mismatch"
        assert aug_mask.shape == mask.shape, f"{name}: Mask shape mismatch"
        
        # Check that augmentation list is not empty
        assert len(aug_list) > 0, f"{name}: No augmentation applied"
        
        # Check that values are in valid range
        assert np.min(aug_image) >= 0 and np.max(aug_image) <= 255, f"{name}: Image values out of range"
        assert np.min(aug_mask) >= 0 and np.max(aug_mask) <= 255, f"{name}: Mask values out of range"
        
        print(f"✓ {name} passed")
        return True
        
    except Exception as e:
        print(f"✗ {name} failed: {str(e)}")
        return False

def main():
    """Main test function"""
    print("Testing Enhanced Mask RCNN Augmentations (Grayscale)")
    print("=" * 60)
    
    # Create test image and mask
    test_img = create_test_image()
    test_mask = create_test_mask()
    
    # List of augmentations to test (grayscale-compatible only)
    augmentations = [
        (rotateImage, "Rotation"),
        (scaleImage, "Scaling"),
        (adjustContrast, "Contrast Adjustment"),
        (addGaussianNoise, "Gaussian Noise"),
        (addSaltAndPepperNoise, "Salt and Pepper Noise"),
        (elasticDeformation, "Elastic Deformation"),
        (addShadows, "Shadows"),
        (addMotionBlur, "Motion Blur"),
        (addIntensityVariation, "Intensity Variation"),
        (addPerspectiveTransform, "Perspective Transform")
    ]
    
    # Test each augmentation
    passed = 0
    total = len(augmentations)
    
    for func, name in augmentations:
        if test_augmentation(func, name, test_img, test_mask):
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} augmentations passed")
    
    if passed == total:
        print("✓ All augmentations working correctly!")
    else:
        print("✗ Some augmentations failed. Check the errors above.")
    
    # Optional: Save test results
    print("\nSaving test images...")
    os.makedirs("test_output", exist_ok=True)
    
    cv2.imwrite("test_output/original_image.png", test_img)
    cv2.imwrite("test_output/original_mask.png", test_mask)
    
    # Save one example of each augmentation
    for func, name in augmentations:
        try:
            aug_img, aug_mask, aug_list = func(test_img.copy(), test_mask.copy(), [])
            cv2.imwrite(f"test_output/{name.lower().replace(' ', '_')}_image.png", aug_img)
            cv2.imwrite(f"test_output/{name.lower().replace(' ', '_')}_mask.png", aug_mask)
        except Exception as e:
            print(f"Could not save {name}: {str(e)}")
    
    print("Test images saved to test_output/ directory")
    print("\nNote: All augmentations are optimized for grayscale images.")

if __name__ == "__main__":
    main() 