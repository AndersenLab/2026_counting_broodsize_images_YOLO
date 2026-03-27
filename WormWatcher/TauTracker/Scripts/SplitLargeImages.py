"""
SplitLargeImages.py

This script splits large images into smaller tiles to reduce memory usage during training
or to create more training data from high-resolution images.

Usage:
- Run the script and it will prompt for source and destination folders
- Specify tile size, overlap, and other parameters
- The script will process all images in the source folder and save tiles to destination folder

Features:
- Configurable tile size (default 512x512)
- Configurable overlap between tiles
- Maintains original image format
- Creates organized output directory structure
- Progress tracking
- Handles various image formats (jpg, png, tiff, etc.)
"""

import os
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path


class ImageSplitter:
    def __init__(self):
        self.source_folder = ""
        self.destination_folder = ""
        self.tile_width = 512
        self.tile_height = 512
        self.overlap_ratio = 0.1
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        
    def get_user_inputs(self):
        """Get user inputs for source folder, destination folder, and parameters"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Get source folder
        messagebox.showinfo("Select Source Folder", "Please select the folder containing images to split")
        self.source_folder = filedialog.askdirectory(title="Select Source Folder")
        if not self.source_folder:
            print("No source folder selected. Exiting.")
            return False
        
        # Get destination folder
        messagebox.showinfo("Select Destination Folder", "Please select the destination folder for split images")
        self.destination_folder = filedialog.askdirectory(title="Select Destination Folder")
        if not self.destination_folder:
            print("No destination folder selected. Exiting.")
            return False
        
        # Get tile dimensions
        tile_size = simpledialog.askstring("Tile Size", 
                                          "Enter tile size (format: WIDTHxHEIGHT, e.g., 512x512):",
                                          initialvalue="512x512")
        if tile_size:
            try:
                width_str, height_str = tile_size.split('x')
                self.tile_width = int(width_str)
                self.tile_height = int(height_str)
            except:
                messagebox.showwarning("Invalid Input", "Invalid tile size format. Using default 512x512")
        
        # Get overlap ratio
        overlap = simpledialog.askfloat("Overlap Ratio", 
                                       "Enter overlap ratio between tiles (0.0 to 0.5):",
                                       initialvalue=0.1, minvalue=0.0, maxvalue=0.5)
        if overlap is not None:
            self.overlap_ratio = overlap
        
        root.destroy()
        return True
    
    def get_image_files(self):
        """Get list of supported image files from source folder"""
        image_files = []
        for file in os.listdir(self.source_folder):
            file_ext = Path(file).suffix.lower()
            if file_ext in self.supported_formats:
                image_files.append(file)
        return sorted(image_files)
    
    def create_tiles(self, image, image_name):
        """Split a single image into tiles"""
        img_height, img_width = image.shape[:2]
        
        # Calculate step size based on overlap
        step_x = int(self.tile_width * (1 - self.overlap_ratio))
        step_y = int(self.tile_height * (1 - self.overlap_ratio))
        
        tiles = []
        tile_info = []
        
        tile_count = 0
        for y in range(0, img_height - self.tile_height + 1, step_y):
            for x in range(0, img_width - self.tile_width + 1, step_x):
                # Extract tile
                tile = image[y:y + self.tile_height, x:x + self.tile_width]
                
                # Create tile filename
                base_name = Path(image_name).stem
                tile_name = f"{base_name}_tile_{tile_count:04d}_{x}_{y}.png"
                
                tiles.append(tile)
                tile_info.append((tile_name, x, y))
                tile_count += 1
        
        # Handle remaining edges if image doesn't divide evenly
        # Right edge
        if img_width % step_x != 0:
            x = img_width - self.tile_width
            for y in range(0, img_height - self.tile_height + 1, step_y):
                tile = image[y:y + self.tile_height, x:x + self.tile_width]
                base_name = Path(image_name).stem
                tile_name = f"{base_name}_tile_{tile_count:04d}_{x}_{y}.png"
                tiles.append(tile)
                tile_info.append((tile_name, x, y))
                tile_count += 1
        
        # Bottom edge
        if img_height % step_y != 0:
            y = img_height - self.tile_height
            for x in range(0, img_width - self.tile_width + 1, step_x):
                tile = image[y:y + self.tile_height, x:x + self.tile_width]
                base_name = Path(image_name).stem
                tile_name = f"{base_name}_tile_{tile_count:04d}_{x}_{y}.png"
                tiles.append(tile)
                tile_info.append((tile_name, x, y))
                tile_count += 1
        
        # Bottom-right corner
        if img_width % step_x != 0 and img_height % step_y != 0:
            x = img_width - self.tile_width
            y = img_height - self.tile_height
            tile = image[y:y + self.tile_height, x:x + self.tile_width]
            base_name = Path(image_name).stem
            tile_name = f"{base_name}_tile_{tile_count:04d}_{x}_{y}.png"
            tiles.append(tile)
            tile_info.append((tile_name, x, y))
            tile_count += 1
            
        return tiles, tile_info
    
    def save_tiles(self, tiles, tile_info):
        """Save tiles to destination folder"""
        os.makedirs(self.destination_folder, exist_ok=True)
        
        for tile, (tile_name, x, y) in zip(tiles, tile_info):
            tile_path = os.path.join(self.destination_folder, tile_name)
            cv2.imwrite(tile_path, tile)
    
    def process_images(self):
        """Process all images in the source folder"""
        image_files = self.get_image_files()
        
        if not image_files:
            print("No supported image files found in source folder.")
            return
        
        print(f"Found {len(image_files)} images to process")
        print(f"Tile size: {self.tile_width}x{self.tile_height}")
        print(f"Overlap ratio: {self.overlap_ratio}")
        print(f"Source: {self.source_folder}")
        print(f"Destination: {self.destination_folder}")
        print("-" * 50)
        
        total_tiles = 0
        
        for i, image_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_file}")
            
            # Load image
            image_path = os.path.join(self.source_folder, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"  Warning: Could not load {image_file}")
                continue
            
            img_height, img_width = image.shape[:2]
            print(f"  Image size: {img_width}x{img_height}")
            
            # Check if image needs splitting
            if img_width <= self.tile_width and img_height <= self.tile_height:
                print(f"  Image smaller than tile size, copying as-is")
                # Copy original image
                dest_path = os.path.join(self.destination_folder, image_file)
                cv2.imwrite(dest_path, image)
                continue
            
            # Create tiles
            tiles, tile_info = self.create_tiles(image, image_file)
            print(f"  Created {len(tiles)} tiles")
            
            # Save tiles
            self.save_tiles(tiles, tile_info)
            
            total_tiles += len(tiles)
        
        print("-" * 50)
        print(f"Processing complete!")
        print(f"Total tiles created: {total_tiles}")
        print(f"Average tiles per image: {total_tiles/len(image_files):.1f}")
    
    def run(self):
        """Main execution function"""
        print("=== Image Splitter ===")
        print("This tool splits large images into smaller tiles")
        print()
        
        if not self.get_user_inputs():
            return
        
        # Confirm settings
        print(f"Settings:")
        print(f"  Source folder: {self.source_folder}")
        print(f"  Destination folder: {self.destination_folder}")
        print(f"  Tile size: {self.tile_width}x{self.tile_height}")
        print(f"  Overlap ratio: {self.overlap_ratio}")
        print()
        
        proceed = input("Proceed with these settings? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Operation cancelled.")
            return
        
        self.process_images()


def main():
    """Main function with command line interface option"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # Command line interface
        splitter = ImageSplitter()
        
        # Get inputs via command line
        splitter.source_folder = input("Enter source folder path: ").strip()
        splitter.destination_folder = input("Enter destination folder path: ").strip()
        
        # Validate folders
        if not os.path.exists(splitter.source_folder):
            print("Source folder does not exist!")
            return
        
        # Get tile size
        tile_size = input("Enter tile size (WIDTHxHEIGHT, default 512x512): ").strip()
        if tile_size:
            try:
                width_str, height_str = tile_size.split('x')
                splitter.tile_width = int(width_str)
                splitter.tile_height = int(height_str)
            except:
                print("Invalid tile size format. Using default 512x512")
        
        # Get overlap
        overlap_input = input("Enter overlap ratio (0.0-0.5, default 0.1): ").strip()
        if overlap_input:
            try:
                splitter.overlap_ratio = float(overlap_input)
            except:
                print("Invalid overlap ratio. Using default 0.1")
        
        splitter.process_images()
    else:
        # GUI interface
        splitter = ImageSplitter()
        splitter.run()


if __name__ == "__main__":
    main() 