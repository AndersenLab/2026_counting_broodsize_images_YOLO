#!/usr/bin/env python3
"""
WormClassifier.py

PyTorch neural network classifier for filtering false positive worm detections.
Designed as a second-stage classifier after Mask R-CNN detection.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


class WormDataset(Dataset):
    """Dataset class for worm classification training data"""
    
    def __init__(self, image_paths, labels, window_size, augment=True):
        """
        Args:
            image_paths (list): List of paths to worm crop images
            labels (list): List of labels (0=false_positive, 1=real_worm)
            window_size (int): Target image size for training
            augment (bool): Whether to apply data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.window_size = window_size
        self.augment = augment
        
        # Basic transforms
        # Note: ToTensor() converts PIL Image to [0,1] range
        # For grayscale, normalize to [-1, 1] using mean=0.5, std=0.5
        # This is standard for grayscale images (better than ImageNet RGB stats)
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((window_size, window_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale: normalize [0,1] to [-1,1]
        ])
        
        # Augmentation transforms for training
        self.augment_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((window_size, window_size)),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale: normalize [0,1] to [-1,1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply transforms
        if self.augment:
            image = self.augment_transform(image)
        else:
            image = self.base_transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


class WormClassifierNet(nn.Module):
    """Lightweight CNN for worm classification"""
    
    def __init__(self, input_size, num_classes=2, dropout_rate=0.3):
        """
        Args:
            input_size (int): Input image size (assumed square) - REQUIRED
            num_classes (int): Number of output classes (2 for binary)
            dropout_rate (float): Dropout rate for regularization
        """
        if input_size is None:
            raise ValueError("input_size is required but not provided")
        super(WormClassifierNet, self).__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8x8
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 4x4
        
        # Calculate flattened size
        conv_output_size = 256 * (input_size // 16) * (input_size // 16)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers with batch norm and ReLU
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


class WormClassifier:
    """Main classifier class with training, saving, loading, and inference methods"""
    
    def __init__(self, config):
        """
        Initialize classifier with configuration
        
        Args:
            config (dict): Configuration dictionary with keys:
                - window_size (int): Input image size
                - learning_rate (float): Learning rate for training
                - batch_size (int): Batch size for training
                - num_epochs (int): Number of training epochs
                - dropout_rate (float): Dropout rate
                - device (str): 'cuda' or 'cpu' or 'auto'
                - model_save_path (str): Path to save trained models
                - early_stopping_patience (int): Early stopping patience
                - confidence_threshold (float): Classification threshold
        """
        self.config = config
        
        # Window size - CRITICAL: must be provided in config
        if 'window_size' not in config:
            raise ValueError("'window_size' is required in config but not provided")
        self.window_size = config['window_size']
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.num_epochs = config.get('num_epochs', 50)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.model_save_path = config.get('model_save_path', './models/')
        
        # Set device
        device_setting = config.get('device', 'auto')
        if device_setting == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_setting)
        
        print(f"WormClassifier initialized - Device: {self.device}, Window size: {self.window_size}")
        
        # Initialize network
        self.model = WormClassifierNet(
            input_size=self.window_size,
            num_classes=2,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train(self, train_image_paths, train_labels, val_image_paths=None, val_labels=None):
        """
        Train the classifier
        
        Args:
            train_image_paths (list): List of paths to training images
            train_labels (list): List of training labels (0=false_positive, 1=real_worm)
            val_image_paths (list): List of paths to validation images (optional)
            val_labels (list): List of validation labels (optional)
        
        Returns:
            dict: Training history with losses and accuracies
        """
        print(f"Starting training with {len(train_image_paths)} training samples")
        
        # Create datasets
        train_dataset = WormDataset(train_image_paths, train_labels, self.window_size, augment=True)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        
        val_loader = None
        if val_image_paths is not None and val_labels is not None:
            val_dataset = WormDataset(val_image_paths, val_labels, self.window_size, augment=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
            print(f"Using {len(val_image_paths)} validation samples")
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.save_model(f"best_model_epoch_{epoch+1}.pth")
                else:
                    patience_counter += 1
                
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            else:
                print(f"Epoch {epoch+1}/{self.num_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        print("Training completed!")
        return self.training_history
    
    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def save_model(self, filename=None):
        """
        Save model and configuration to disk
        
        Args:
            filename (str): Custom filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"worm_classifier_{timestamp}.pth"
        
        os.makedirs(self.model_save_path, exist_ok=True)
        filepath = os.path.join(self.model_save_path, filename)
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'model_architecture': {
                'input_size': self.window_size,
                'num_classes': 2,
                'dropout_rate': self.dropout_rate
            }
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to: {filepath}")
        
        # Also save config as JSON for easy reading
        config_path = filepath.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return filepath
    
    def load_model(self, filepath):
        """
        Load model and configuration from disk
        
        Args:
            filepath (str): Path to saved model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        print(f"Loading model from: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Update config if saved
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
            # Ensure window_size is loaded from saved config
            if 'window_size' in checkpoint['config']:
                self.window_size = checkpoint['config']['window_size']
            else:
                raise ValueError("Saved model config missing 'window_size'")
        else:
            raise ValueError("Saved model missing 'config' - cannot determine window_size")
        
        # Recreate model with correct architecture
        if 'model_architecture' in checkpoint:
            arch = checkpoint['model_architecture']
            self.model = WormClassifierNet(
                input_size=arch['input_size'],
                num_classes=arch['num_classes'],
                dropout_rate=arch['dropout_rate']
            ).to(self.device)
        
        # Load state dicts
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        self.model.eval()
        print(f"Model loaded successfully - Device: {self.device}, Window size: {self.window_size}")
    
    def predict(self, image_crops):
        """
        Perform inference on image crops
        
        Args:
            image_crops (list or np.array): List of image crops or single crop
                                          Each crop should be grayscale numpy array
        
        Returns:
            dict: Predictions with probabilities and binary classifications
                - probabilities (list): Probability of being a real worm [0-1]
                - predictions (list): Binary predictions (0=false_positive, 1=real_worm)
                - confidences (list): Confidence scores (max probability)
        """
        self.model.eval()
        
        # Handle single image input
        if isinstance(image_crops, np.ndarray) and len(image_crops.shape) == 2:
            image_crops = [image_crops]
        
        # Prepare transform
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.window_size, self.window_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale: normalize [0,1] to [-1,1]
        ])
        
        probabilities = []
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for crop in image_crops:
                # Preprocess
                if crop.dtype != np.uint8:
                    crop = (crop * 255).astype(np.uint8)
                
                tensor = transform(crop).unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs = self.model(tensor)
                probs = F.softmax(outputs, dim=1)
                
                # Extract results
                prob_real_worm = probs[0][1].item()  # Probability of class 1 (real worm)
                confidence = torch.max(probs, 1)[0].item()
                prediction = 1 if prob_real_worm > self.confidence_threshold else 0
                
                probabilities.append(prob_real_worm)
                predictions.append(prediction)
                confidences.append(confidence)
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'confidences': confidences
        }
    
    def evaluate(self, test_image_paths, test_labels):
        """
        Evaluate model performance on test set
        
        Args:
            test_image_paths (list): List of paths to test images
            test_labels (list): List of test labels
        
        Returns:
            dict: Evaluation metrics
        """
        print(f"Evaluating on {len(test_image_paths)} test samples...")
        
        test_dataset = WormDataset(test_image_paths, test_labels, self.window_size, augment=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                probs = F.softmax(outputs, dim=1)
                
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of real worm
                all_labels.extend(targets.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions),
            'recall': recall_score(all_labels, all_predictions),
            'f1_score': f1_score(all_labels, all_predictions),
            'roc_auc': roc_auc_score(all_labels, all_probabilities)
        }
        
        print("Evaluation Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if not self.training_history['train_loss']:
            print("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.training_history['train_loss'], label='Train Loss')
        if self.training_history['val_loss']:
            ax1.plot(self.training_history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.training_history['train_acc'], label='Train Acc')
        if self.training_history['val_acc']:
            ax2.plot(self.training_history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()

    def generate_training_dataset(self, dataset_parent_dir, output_dir, extraction_config=None):
        """
        Generate training dataset from DatasetImages and DatasetMasks folders
        
        Args:
            dataset_parent_dir (str): Parent directory containing DatasetImages and DatasetMasks
            output_dir (str): Output directory for extracted crops (will create Classifier/Worm and Classifier/Background)
            extraction_config (dict): Configuration for extraction process with keys:
                - samples_per_image (int): Number of crops to extract per image (default: 100)
                - min_worm_pixels (int): Minimum worm pixels to classify as worm (default: 50)
                - target_balance_ratio (float): Target ratio of worm:background samples (default: 1.0)
                - max_attempts_per_image (int): Max attempts to find valid crops per image (default: 500)
                - random_seed (int): Random seed for reproducibility (default: 42)
        
        Returns:
            dict: Statistics about the generated dataset
        """
        # Default extraction configuration
        if extraction_config is None:
            extraction_config = {}
        
        samples_per_image = extraction_config.get('samples_per_image', 100)
        min_worm_pixels = extraction_config.get('min_worm_pixels', 50)
        target_balance_ratio = extraction_config.get('target_balance_ratio', 1.0)
        max_attempts = extraction_config.get('max_attempts_per_image', 500)
        random_seed = extraction_config.get('random_seed', 42)
        
        print(f"🔄 Starting training dataset generation...")
        print(f"   Window size: {self.window_size}x{self.window_size}")
        print(f"   Samples per image: {samples_per_image}")
        print(f"   Min worm pixels for positive label: {min_worm_pixels}")
        print(f"   Target balance ratio (worm:background): {target_balance_ratio}")
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Setup directories
        images_dir = os.path.join(dataset_parent_dir, "DatasetImages")
        masks_dir = os.path.join(dataset_parent_dir, "DatasetMasks")
        
        if not os.path.exists(images_dir):
            raise ValueError(f"DatasetImages directory not found: {images_dir}")
        if not os.path.exists(masks_dir):
            raise ValueError(f"DatasetMasks directory not found: {masks_dir}")
        
        # Create output directories
        worm_output_dir = os.path.join(output_dir, "Classifier", "Worm")
        background_output_dir = os.path.join(output_dir, "Classifier", "Background")
        os.makedirs(worm_output_dir, exist_ok=True)
        os.makedirs(background_output_dir, exist_ok=True)
        
        # Find matching image and mask files
        image_files = sorted([f for f in os.listdir(images_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
        mask_files = sorted([f for f in os.listdir(masks_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))])
        
        print(f"📁 Found {len(image_files)} images and {len(mask_files)} masks")
        
        # Match image and mask files by name
        matched_pairs = []
        
        # First try index-based matching (fastest) - assumes files are in same order
        for i, img_file in enumerate(image_files):
            if i < len(mask_files):
                img_base = os.path.splitext(img_file)[0]
                mask_base = os.path.splitext(mask_files[i])[0]
                
                # Remove _mask from mask name and check if it matches image name
                mask_without_suffix = mask_base.replace('_mask', '')
                if mask_without_suffix == img_base:
                    matched_pairs.append((img_file, mask_files[i]))
                    continue
            
            # If index matching fails, fall back to search through all mask files
            img_base = os.path.splitext(img_file)[0]
            mask_candidates = []
            for mask_file in mask_files:
                mask_base = os.path.splitext(mask_file)[0]
                # Remove _mask from mask name and check if it matches image name
                mask_without_suffix = mask_base.replace('_mask', '')
                if mask_without_suffix == img_base:
                    mask_candidates.append(mask_file)
            
            if mask_candidates:
                matched_pairs.append((img_file, mask_candidates[0]))
            else:
                print(f"⚠️  No matching mask found for image: {img_file}")
        
        print(f"✅ Matched {len(matched_pairs)} image-mask pairs")
        
        if len(matched_pairs) == 0:
            raise ValueError("No matching image-mask pairs found!")
        
        # Statistics tracking
        stats = {
            'total_crops_extracted': 0,
            'worm_crops': 0,
            'background_crops': 0,
            'images_processed': 0,
            'extraction_failures': 0
        }
        
        worm_samples = []
        background_samples = []
        
        # Process each image-mask pair
        for img_idx, (img_file, mask_file) in enumerate(matched_pairs):
            print(f"🖼️  Processing {img_idx+1}/{len(matched_pairs)}: {img_file}")
            
            try:
                # Load image and mask
                img_path = os.path.join(images_dir, img_file)
                mask_path = os.path.join(masks_dir, mask_file)
                
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"❌ Failed to load image: {img_path}")
                    stats['extraction_failures'] += 1
                    continue
                    
                if mask is None:
                    print(f"❌ Failed to load mask: {mask_path}")
                    stats['extraction_failures'] += 1
                    continue
                
                # Ensure mask and image have same dimensions
                if image.shape != mask.shape:
                    print(f"⚠️  Image and mask size mismatch for {img_file}: {image.shape} vs {mask.shape}")
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # Extract crops from this image
                img_crops = self._extract_crops_from_image(
                    image, mask, img_file, samples_per_image, 
                    min_worm_pixels, max_attempts
                )
                
                worm_samples.extend(img_crops['worm'])
                background_samples.extend(img_crops['background'])
                
                stats['images_processed'] += 1
                
                print(f"   Extracted {len(img_crops['worm'])} worm and {len(img_crops['background'])} background crops")
                
            except Exception as e:
                print(f"❌ Error processing {img_file}: {e}")
                stats['extraction_failures'] += 1
                continue
        
        print(f"\n📊 Extraction complete! Total samples:")
        print(f"   Worm samples: {len(worm_samples)}")
        print(f"   Background samples: {len(background_samples)}")
        
        # Balance the dataset
        target_worm_count = min(len(worm_samples), len(background_samples))
        target_background_count = int(target_worm_count / target_balance_ratio)
        
        # Randomly sample to achieve balance
        if len(worm_samples) > target_worm_count:
            worm_indices = np.random.choice(len(worm_samples), target_worm_count, replace=False)
            worm_samples = [worm_samples[i] for i in worm_indices]
        if len(background_samples) > target_background_count:
            background_indices = np.random.choice(len(background_samples), target_background_count, replace=False)
            background_samples = [background_samples[i] for i in background_indices]
        
        print(f"\n⚖️  Balanced dataset:")
        print(f"   Final worm samples: {len(worm_samples)}")
        print(f"   Final background samples: {len(background_samples)}")
        
        # Save crops to disk
        print(f"\n💾 Saving crops to disk...")
        
        # Save worm crops
        for i, (crop, metadata) in enumerate(worm_samples):
            filename = f"worm_{metadata['source_img']}_{metadata['crop_id']:04d}.png"
            filepath = os.path.join(worm_output_dir, filename)
            cv2.imwrite(filepath, crop)
            
        # Save background crops  
        for i, (crop, metadata) in enumerate(background_samples):
            filename = f"bg_{metadata['source_img']}_{metadata['crop_id']:04d}.png"
            filepath = os.path.join(background_output_dir, filename)
            cv2.imwrite(filepath, crop)
        
        stats['total_crops_extracted'] = len(worm_samples) + len(background_samples)
        stats['worm_crops'] = len(worm_samples)
        stats['background_crops'] = len(background_samples)
        
        # Save dataset statistics
        stats_file = os.path.join(output_dir, "Classifier", "dataset_stats.json")
        with open(stats_file, 'w') as f:
            json.dump({
                'extraction_config': extraction_config,
                'dataset_statistics': stats,
                'source_directory': dataset_parent_dir,
                'output_directory': output_dir,
                'generation_timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"✅ Dataset generation complete!")
        print(f"   Worm crops saved to: {worm_output_dir}")
        print(f"   Background crops saved to: {background_output_dir}")
        print(f"   Statistics saved to: {stats_file}")
        
        return stats
    
    def _extract_crops_from_image(self, image, mask, source_filename, samples_per_image, min_worm_pixels, max_attempts):
        """
        Extract crops from a single image using corresponding mask for labeling
        
        Args:
            image (np.array): Grayscale image
            mask (np.array): Corresponding mask image
            source_filename (str): Source image filename for metadata
            samples_per_image (int): Target number of samples to extract
            min_worm_pixels (int): Minimum worm pixels for positive classification
            max_attempts (int): Maximum attempts to find valid crops
        
        Returns:
            dict: Extracted crops separated by class {'worm': [...], 'background': [...]}
        """
        h, w = image.shape
        half_window = self.window_size // 2
        
        # Ensure we can extract full windows
        if h < self.window_size or w < self.window_size:
            print(f"⚠️  Image {source_filename} too small ({h}x{w}) for window size {self.window_size}")
            return {'worm': [], 'background': []}
        
        worm_crops = []
        background_crops = []
        attempts = 0
        
        # Find worm objects in the mask for centered worm crops
        worm_objects = self._find_worm_objects(mask, min_worm_pixels)
        
        # Calculate target samples per class
        target_worm = samples_per_image // 2
        target_background = samples_per_image - target_worm
        
        # Extract worm crops centered on worm objects
        worm_attempts = 0
        while len(worm_crops) < target_worm and worm_attempts < max_attempts // 2:
            worm_attempts += 1
            
            if worm_objects:
                # Select a random worm object and center crop on it
                worm_obj = worm_objects[np.random.randint(len(worm_objects))]
                center_x, center_y = worm_obj['center']
                
                # Ensure the crop fits within image bounds
                if (half_window <= center_x < w - half_window and 
                    half_window <= center_y < h - half_window):
                    
                    # Extract crop centered on worm
                    crop = image[center_y - half_window:center_y + half_window,
                                center_x - half_window:center_x + half_window]
                    mask_crop = mask[center_y - half_window:center_y + half_window,
                                   center_x - half_window:center_x + half_window]
                    
                    # Verify crop size
                    if crop.shape == (self.window_size, self.window_size):
                        worm_pixel_count = np.sum(mask_crop > 0)
                        
                        # Create metadata
                        metadata = {
                            'source_img': os.path.splitext(source_filename)[0],
                            'crop_id': len(worm_crops),
                            'center_x': center_x,
                            'center_y': center_y,
                            'worm_pixels': worm_pixel_count,
                            'worm_object_id': worm_obj['id']
                        }
                        
                        worm_crops.append((crop, metadata))
            else:
                # No worm objects found, try random locations with worm pixels
                center_x = np.random.randint(half_window, w - half_window)
                center_y = np.random.randint(half_window, h - half_window)
                
                crop = image[center_y - half_window:center_y + half_window,
                            center_x - half_window:center_x + half_window]
                mask_crop = mask[center_y - half_window:center_y + half_window,
                               center_x - half_window:center_x + half_window]
                
                if crop.shape == (self.window_size, self.window_size):
                    worm_pixel_count = np.sum(mask_crop > 0)
                    
                    if worm_pixel_count >= min_worm_pixels:
                        metadata = {
                            'source_img': os.path.splitext(source_filename)[0],
                            'crop_id': len(worm_crops),
                            'center_x': center_x,
                            'center_y': center_y,
                            'worm_pixels': worm_pixel_count,
                            'worm_object_id': -1  # Random location
                        }
                        worm_crops.append((crop, metadata))
        
        # Extract background crops with no worm pixels
        background_attempts = 0
        while len(background_crops) < target_background and background_attempts < max_attempts // 2:
            background_attempts += 1
            
            # Random crop location
            center_x = np.random.randint(half_window, w - half_window)
            center_y = np.random.randint(half_window, h - half_window)
            
            # Extract crop and corresponding mask region
            crop = image[center_y - half_window:center_y + half_window,
                        center_x - half_window:center_x + half_window]
            mask_crop = mask[center_y - half_window:center_y + half_window,
                           center_x - half_window:center_x + half_window]
            
            # Verify crop size
            if crop.shape != (self.window_size, self.window_size):
                continue
            
            # Count worm pixels in mask (any pixel > 0 is worm)
            worm_pixel_count = np.sum(mask_crop > 0)
            
            # Only accept background crops with NO worm pixels
            if worm_pixel_count == 0:
                metadata = {
                    'source_img': os.path.splitext(source_filename)[0],
                    'crop_id': len(background_crops),
                    'center_x': center_x,
                    'center_y': center_y,
                    'worm_pixels': worm_pixel_count
                }
                background_crops.append((crop, metadata))
        
        return {'worm': worm_crops, 'background': background_crops}
    
    def _find_worm_objects(self, mask, min_worm_pixels):
        """
        Find worm objects in the mask using connected components analysis
        
        Args:
            mask (np.array): Binary mask where > 0 indicates worm pixels
            min_worm_pixels (int): Minimum pixels to consider a worm object
        
        Returns:
            list: List of worm objects with center coordinates and pixel count
        """
        import cv2
        
        # Create binary mask
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        worm_objects = []
        
        # Skip background (label 0)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_worm_pixels:
                # Get centroid
                center_x = int(centroids[i][0])
                center_y = int(centroids[i][1])
                
                worm_objects.append({
                    'id': i,
                    'center': (center_x, center_y),
                    'area': area,
                    'bbox': stats[i, :4]  # x, y, width, height
                })
        
        return worm_objects
    
    def train_from_generated_dataset(self, classifier_dir, train_split=0.8, val_split=0.1):
        """
        Train the classifier using a previously generated dataset
        
        Args:
            classifier_dir (str): Directory containing Classifier/Worm and Classifier/Background folders
            train_split (float): Fraction of data for training (default: 0.8)
            val_split (float): Fraction of data for validation (default: 0.1, remainder is test)
        
        Returns:
            dict: Training results and evaluation metrics
        """
        worm_dir = os.path.join(classifier_dir, "Classifier", "Worm")
        background_dir = os.path.join(classifier_dir, "Classifier", "Background")
        
        if not os.path.exists(worm_dir) or not os.path.exists(background_dir):
            raise ValueError(f"Generated dataset not found. Expected {worm_dir} and {background_dir}")
        
        # Collect all image paths and labels (support multiple image formats)
        image_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.tif')
        worm_files = [os.path.join(worm_dir, f) for f in os.listdir(worm_dir) 
                     if f.lower().endswith(image_extensions)]
        bg_files = [os.path.join(background_dir, f) for f in os.listdir(background_dir) 
                   if f.lower().endswith(image_extensions)]
        
        all_paths = worm_files + bg_files
        all_labels = [1] * len(worm_files) + [0] * len(bg_files)  # 1=worm, 0=background
        
        print(f"📚 Found {len(worm_files)} worm samples and {len(bg_files)} background samples")
        
        # Shuffle data
        combined = list(zip(all_paths, all_labels))
        np.random.shuffle(combined)
        all_paths, all_labels = zip(*combined)
        
        # Split data
        n_total = len(all_paths)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        train_paths = all_paths[:n_train]
        train_labels = all_labels[:n_train]
        val_paths = all_paths[n_train:n_train + n_val]
        val_labels = all_labels[n_train:n_train + n_val]
        test_paths = all_paths[n_train + n_val:]
        test_labels = all_labels[n_train + n_val:]
        
        print(f"📊 Data split - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Train the model
        print(f"\n🚀 Starting training...")
        training_history = self.train(train_paths, train_labels, val_paths, val_labels)
        
        # Evaluate on test set
        print(f"\n📈 Evaluating on test set...")
        test_metrics = self.evaluate(test_paths, test_labels)
        
        # Save the trained model in Classifier/Models directory
        models_dir = os.path.join(classifier_dir, "Classifier", "Models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Temporarily update model_save_path to save in the correct location
        original_save_path = self.model_save_path
        self.model_save_path = models_dir
        
        model_path = self.save_model()
        
        # Restore original save path
        self.model_save_path = original_save_path
        
        print(f"\n✅ Training workflow complete!")
        print(f"   Model saved to: {model_path}")
        
        return {
            'training_history': training_history,
            'test_metrics': test_metrics,
            'data_splits': {
                'train': len(train_paths),
                'validation': len(val_paths), 
                'test': len(test_paths)
            },
            'model_path': model_path
        }


# Example usage and configuration
def get_default_config(window_size=None):
    """
    Get default configuration for the worm classifier
    
    Args:
        window_size (int): Required window size for input images. Must be provided.
    
    Returns:
        dict: Configuration dictionary
    """
    if window_size is None:
        raise ValueError("window_size is required but not provided. Use get_default_config(window_size=64) or similar.")
    
    return {
        'window_size': window_size,     # Input image size (REQUIRED)
        'learning_rate': 0.001,         # Learning rate
        'batch_size': 32,               # Batch size
        'num_epochs': 50,               # Maximum epochs
        'dropout_rate': 0.3,            # Dropout rate
        'device': 'auto',               # 'cuda', 'cpu', or 'auto'
        'model_save_path': './models/', # Directory to save models
        'early_stopping_patience': 10,  # Early stopping patience
        'confidence_threshold': 0.5      # Classification threshold
    }


if __name__ == "__main__":
    # Example usage
    config = get_default_config(window_size=64)  # Must specify window_size
    classifier = WormClassifier(config)
    
    print("WormClassifier ready for training!")
    print(f"Configuration: {config}") 