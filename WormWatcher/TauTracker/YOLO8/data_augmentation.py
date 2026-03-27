"""
Data Augmentation Class for YOLO Segmentation Training
Robust augmentation with proper mask handling
"""

import cv2
import numpy as np
import random
from typing import Tuple, List, Optional

__all__ = ['DataAugmentation']


class DataAugmentation:
    """
    Data augmentation class for image and mask pairs.
    Ensures masks preserve discrete values while images are smoothly augmented.
    """
    
    def __init__(self, 
                 apply_flips=True,
                 apply_blur=True,
                 apply_noise=True,
                 apply_brightness=True,
                 apply_rotation=True,
                 apply_scaling=True,
                 apply_translation=True,
                 apply_contrast=True,
                 apply_shear=True,
                 apply_elastic=False,  # Expensive, disabled by default
                 
                 # Parameter ranges
                 rotation_range=(-45, 45),
                 scale_range=(0.7, 1.3),
                 translation_percent=12,
                 contrast_range=(0.7, 1.3),
                 shear_range=(-0.1, 0.1),
                 noise_range=(5, 15),
                 brightness_range=(30, 70),
                 blur_kernel_range=(7, 14),
                 
                 # Probabilities
                 elastic_probability=0.3,
                 
                 # Validation
                 min_objects_per_mask=1,
                 min_pixels_per_object=10):
        """
        Initialize augmentation parameters.
        
        Args:
            apply_*: Boolean flags to enable/disable augmentation types
            *_range: Parameter ranges for augmentations
            elastic_probability: Probability of applying expensive elastic deformation
            min_objects_per_mask: Minimum objects required in mask
            min_pixels_per_object: Minimum pixels per object
        """
        self.apply_flips = apply_flips
        self.apply_blur = apply_blur
        self.apply_noise = apply_noise
        self.apply_brightness = apply_brightness
        self.apply_rotation = apply_rotation
        self.apply_scaling = apply_scaling
        self.apply_translation = apply_translation
        self.apply_contrast = apply_contrast
        self.apply_shear = apply_shear
        self.apply_elastic = apply_elastic
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_percent = translation_percent
        self.contrast_range = contrast_range
        self.shear_range = shear_range
        self.noise_range = noise_range
        self.brightness_range = brightness_range
        self.blur_kernel_range = blur_kernel_range
        
        self.elastic_probability = elastic_probability
        
        self.min_objects_per_mask = min_objects_per_mask
        self.min_pixels_per_object = min_pixels_per_object
    
    def validate_mask(self, mask: np.ndarray, debug: bool = False) -> bool:
        """
        Validate that mask contains viable objects.
        
        Args:
            mask: Mask array
            debug: Print debug information
        
        Returns:
            bool: True if valid
        """
        # Convert to grayscale if needed
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Get unique values (excluding background)
        unique_values = np.unique(mask)
        unique_values = unique_values[unique_values != 0]
        
        if len(unique_values) < self.min_objects_per_mask:
            if debug:
                print(f"❌ Not enough objects: {len(unique_values)} < {self.min_objects_per_mask}")
            return False
        
        # Check all objects have sufficient pixels
        for value in unique_values:
            object_pixels = np.sum(mask == value)
            if object_pixels < self.min_pixels_per_object:
                if debug:
                    print(f"❌ Object {value} too small: {object_pixels} < {self.min_pixels_per_object}")
                return False
        
        return True
    
    def flip(self, image: np.ndarray, mask: np.ndarray, horizontal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Flip image and mask horizontally or vertically"""
        flip_code = 1 if horizontal else 0
        return cv2.flip(image, flip_code), cv2.flip(mask, flip_code)
    
    def blur(self, image: np.ndarray, mask: np.ndarray, kernel_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Gaussian blur to image only (mask unchanged)"""
        if kernel_size is None:
            kernel_size = random.randrange(
                self.blur_kernel_range[0], 
                self.blur_kernel_range[1] + 1, 
                2  # Odd numbers only
            )
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred, mask  # Mask unchanged
    
    def add_noise(self, image: np.ndarray, mask: np.ndarray, noise_amount: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Add random noise to image only (mask unchanged)"""
        if noise_amount is None:
            noise_amount = random.randint(*self.noise_range)
        
        noise = np.random.normal(0, noise_amount, image.shape).astype(np.float32)
        img_float = image.astype(np.float32)
        img_noise = img_float + noise
        img_noise = np.clip(img_noise, 0, 255).astype(np.uint8)
        
        return img_noise, mask  # Mask unchanged
    
    def adjust_brightness(self, image: np.ndarray, mask: np.ndarray, brightness_change: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust brightness of image only (mask unchanged)"""
        if brightness_change is None:
            brightness_change = random.randint(*self.brightness_range)
        
        lighting = np.full(image.shape, brightness_change, dtype=np.uint8)
        if random.randint(0, 1) == 1:
            img_bright = cv2.add(image, lighting)
        else:
            img_bright = cv2.subtract(image, lighting)
        
        return img_bright, mask  # Mask unchanged
    
    def rotate(self, image: np.ndarray, mask: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate image and mask"""
        if angle is None:
            angle = random.uniform(*self.rotation_range)
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Image: linear interpolation
        rotated_img = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                    borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
        # Mask: nearest neighbor to preserve discrete values
        rotated_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
        
        return rotated_img, rotated_mask
    
    def scale(self, image: np.ndarray, mask: np.ndarray, scale_factor: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Scale image and mask"""
        if scale_factor is None:
            scale_factor = random.uniform(*self.scale_range)
        
        h, w = image.shape[:2]
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        
        # Resize
        scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Crop or pad to original size
        if scale_factor > 1.0:
            # Crop from center
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            scaled_img = scaled_img[start_y:start_y+h, start_x:start_x+w]
            scaled_mask = scaled_mask[start_y:start_y+h, start_x:start_x+w]
        else:
            # Pad to original size
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_y, h-new_h-pad_y, 
                                          pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
            scaled_mask = cv2.copyMakeBorder(scaled_mask, pad_y, h-new_h-pad_y, 
                                           pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
        
        return scaled_img, scaled_mask
    
    def translate(self, image: np.ndarray, mask: np.ndarray, 
                  tx: Optional[int] = None, ty: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Translate image and mask"""
        h, w = image.shape[:2]
        max_trans = int(min(w, h) * self.translation_percent / 100)
        
        if tx is None:
            tx = random.randint(-max_trans, max_trans)
        if ty is None:
            ty = random.randint(-max_trans, max_trans)
        
        translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        
        translated_img = cv2.warpAffine(image, translation_matrix, (w, h), 
                                        borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
        translated_mask = cv2.warpAffine(mask, translation_matrix, (w, h), 
                                        borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
        
        return translated_img, translated_mask
    
    def adjust_contrast(self, image: np.ndarray, mask: np.ndarray, 
                       contrast_factor: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust contrast of image only (mask unchanged)"""
        if contrast_factor is None:
            contrast_factor = random.uniform(*self.contrast_range)
        
        contrasted = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
        return contrasted, mask  # Mask unchanged
    
    def shear(self, image: np.ndarray, mask: np.ndarray, 
              shear_x: Optional[float] = None, shear_y: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply shear transformation to image and mask"""
        h, w = image.shape[:2]
        
        if shear_x is None:
            shear_x = random.uniform(*self.shear_range)
        if shear_y is None:
            shear_y = random.uniform(*self.shear_range)
        
        shear_matrix = np.array([[1, shear_x, 0], [shear_y, 1, 0]], dtype=np.float32)
        
        sheared_img = cv2.warpAffine(image, shear_matrix, (w, h), 
                                    borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_LINEAR)
        sheared_mask = cv2.warpAffine(mask, shear_matrix, (w, h), 
                                     borderMode=cv2.BORDER_REFLECT, flags=cv2.INTER_NEAREST)
        
        return sheared_img, sheared_mask
    
    def elastic_deformation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply elastic deformation to image and mask (expensive)"""
        h, w = image.shape[:2]
        
        # Reduced parameters for subtle distortion
        alpha = random.uniform(20, 60)
        sigma = random.uniform(6, 10)
        
        # Generate random displacement fields
        dx = np.random.randn(h, w) * alpha
        dy = np.random.randn(h, w) * alpha
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        deformed_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        deformed_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
        
        return deformed_img, deformed_mask
    
    def apply_random_augmentation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Apply a random augmentation to image and mask.
        
        Args:
            image: Input image
            mask: Input mask
        
        Returns:
            Tuple of (augmented_image, augmented_mask, augmentation_name)
        """
        # Collect available augmentations
        augmentations = []
        
        if self.apply_flips:
            augmentations.append(('flip_h', lambda img, msk: self.flip(img, msk, horizontal=True)))
            augmentations.append(('flip_v', lambda img, msk: self.flip(img, msk, horizontal=False)))
        
        if self.apply_blur:
            augmentations.append(('blur', lambda img, msk: self.blur(img, msk)))
        
        if self.apply_noise:
            augmentations.append(('noise', lambda img, msk: self.add_noise(img, msk)))
        
        if self.apply_brightness:
            augmentations.append(('brightness', lambda img, msk: self.adjust_brightness(img, msk)))
        
        if self.apply_rotation:
            augmentations.append(('rotate', lambda img, msk: self.rotate(img, msk)))
        
        if self.apply_scaling:
            augmentations.append(('scale', lambda img, msk: self.scale(img, msk)))
        
        if self.apply_translation:
            augmentations.append(('translate', lambda img, msk: self.translate(img, msk)))
        
        if self.apply_contrast:
            augmentations.append(('contrast', lambda img, msk: self.adjust_contrast(img, msk)))
        
        if self.apply_shear:
            augmentations.append(('shear', lambda img, msk: self.shear(img, msk)))
        
        if self.apply_elastic and random.random() < self.elastic_probability:
            augmentations.append(('elastic', lambda img, msk: self.elastic_deformation(img, msk)))
        
        if not augmentations:
            return image, mask, "none"
        
        # Select random augmentation
        aug_name, aug_func = random.choice(augmentations)
        aug_img, aug_mask = aug_func(image, mask)
        
        # Validate result
        if not self.validate_mask(aug_mask):
            # If augmentation broke the mask, return original
            return image, mask, f"{aug_name}_invalid"
        
        return aug_img, aug_mask, aug_name
    
    def apply_combination(self, image: np.ndarray, mask: np.ndarray, 
                         num_augmentations: int = 2) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Apply multiple random augmentations in sequence.
        
        Args:
            image: Input image
            mask: Input mask
            num_augmentations: Number of augmentations to apply
        
        Returns:
            Tuple of (augmented_image, augmented_mask, list_of_augmentation_names)
        """
        aug_img, aug_mask = image.copy(), mask.copy()
        aug_names = []
        
        for _ in range(num_augmentations):
            aug_img, aug_mask, aug_name = self.apply_random_augmentation(aug_img, aug_mask)
            if aug_name != "none":
                aug_names.append(aug_name)
        
        return aug_img, aug_mask, aug_names

