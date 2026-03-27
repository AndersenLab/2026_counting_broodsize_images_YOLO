#!/usr/bin/env python3
"""
Image Consolidation Script
Organizes images from nested folder structures into a single consolidated folder.

Author: Generated for TauTracker project
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Set, Dict, Tuple
import json
import time
import tkinter as tk
from tkinter import filedialog, messagebox

class ImageConsolidator:
    
    # Common image file extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif', '.webp', '.svg'}
    
    def __init__(self, config: Dict):
        self.parent_folder = Path(config['parent_folder'])
        self.consolidated_folder_name = config.get('consolidated_folder_name', '_consolidated')
        self.images_per_folder = config.get('images_per_folder', 1)
        self.exclude_patterns = config.get('exclude_patterns', ['_Analysis', '_consolidated'])
        self.dry_run = config.get('dry_run', False)
        self.overwrite_existing = config.get('overwrite_existing', False)
        self.selection_mode = config.get('selection_mode', 'largest')
        
        # Create consolidated/output folder path (support explicit output folder)
        output_folder = config.get('output_folder')
        if output_folder:
            self.consolidated_folder = Path(output_folder)
        else:
            self.consolidated_folder = self.parent_folder / self.consolidated_folder_name
        
        # Statistics
        self.stats = {
            'folders_processed': 0,
            'folders_skipped': 0,
            'images_found': 0,
            'images_copied': 0,
            'errors': []
        }
    
    def should_exclude_folder(self, folder_path: Path) -> bool:
        """Check if folder should be excluded based on exclude patterns."""
        folder_name = folder_path.name
        
        # Always exclude the consolidated/output folder itself
        try:
            if folder_path.resolve() == self.consolidated_folder.resolve():
                return True
        except Exception:
            # Fallback to name-based check below if resolve fails
            pass
        if folder_name == self.consolidated_folder_name:
            return True
        
        # Check against exclude patterns
        for pattern in self.exclude_patterns:
            if pattern.lower() in folder_name.lower():
                return True
        
        return False
    
    def is_image_file(self, file_path: Path) -> bool:
        """Check if file is a supported image type."""
        return file_path.suffix.lower() in self.IMAGE_EXTENSIONS
    
    def get_largest_images(self, folder_path: Path, n: int) -> List[Path]:
        """Get the N largest image files from a folder."""
        try:
            # Find all image files in the folder
            image_files = [f for f in folder_path.iterdir() 
                          if f.is_file() and self.is_image_file(f)]
            
            if not image_files:
                return []
            
            # Sort by file size (descending) and take the top N
            image_files.sort(key=lambda x: x.stat().st_size, reverse=True)
            return image_files[:n]
            
        except Exception as e:
            self.stats['errors'].append(f"Error processing folder {folder_path}: {e}")
            return []
    
    def select_images_from_folder(self, folder_path: Path, n: int) -> List[Path]:
        """Select images from a folder based on the configured selection mode."""
        try:
            image_files = [f for f in folder_path.iterdir()
                           if f.is_file() and self.is_image_file(f)]
            if not image_files:
                return []

            if self.selection_mode == 'first':
                image_files.sort(key=lambda x: x.name.lower())
                return image_files[:n]
            else:
                image_files.sort(key=lambda x: x.stat().st_size, reverse=True)
                return image_files[:n]
        except Exception as e:
            self.stats['errors'].append(f"Error processing folder {folder_path}: {e}")
            return []
    
    def create_consolidated_filename(self, original_path: Path, relative_path: Path) -> str:
        """Create new filename with folder structure as prefix."""
        # Get the folder structure relative to parent folder
        folder_structure = str(relative_path.parent).replace(os.sep, '_')
        
        # Remove any leading dots or separators
        folder_structure = folder_structure.strip('._')
        
        # If we're at the root, use a simple prefix
        if not folder_structure or folder_structure == '.':
            folder_structure = 'root'
        
        # Create new filename: folderstructure_originalname.ext
        new_filename = f"{folder_structure}_{original_path.name}"
        
        # Handle potential filename conflicts by adding numbers
        counter = 1
        base_name = new_filename
        while (self.consolidated_folder / new_filename).exists():
            name_parts = base_name.rsplit('.', 1)
            if len(name_parts) == 2:
                new_filename = f"{name_parts[0]}_{counter}.{name_parts[1]}"
            else:
                new_filename = f"{base_name}_{counter}"
            counter += 1
        
        return new_filename
    
    def find_terminal_folders(self) -> List[Path]:
        """Find all terminal folders (folders with no subfolders containing images)."""
        terminal_folders = []
        
        for root, dirs, files in os.walk(self.parent_folder):
            root_path = Path(root)
            
            # Skip excluded folders
            if self.should_exclude_folder(root_path):
                continue
            
            # Remove excluded directories from dirs to prevent walking into them
            dirs[:] = [d for d in dirs if not self.should_exclude_folder(root_path / d)]
            
            # Check if this folder has image files
            has_images = any(self.is_image_file(Path(root) / f) for f in files)
            
            # Check if this folder has any subdirectories (after filtering)
            has_subdirs = len(dirs) > 0
            
            # If it has images and no subdirectories, it's a terminal folder
            if has_images and not has_subdirs:
                terminal_folders.append(root_path)
        
        return terminal_folders
    
    def consolidate_images(self, show_gui_confirmation: bool = False) -> Dict:
        """Main function to consolidate images."""
        print(f"Starting image consolidation from: {self.parent_folder}")
        print(f"Consolidated folder: {self.consolidated_folder}")
        print(f"Images per folder: {self.images_per_folder}")
        print(f"Exclude patterns: {self.exclude_patterns}")
        print(f"Dry run: {self.dry_run}")
        print(f"Selection mode: {self.selection_mode}")
        print("-" * 50)
        
        # Show GUI confirmation if requested
        if show_gui_confirmation:
            try:
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)
                
                message = f"""Image Consolidation Settings:
                
Source folder: {self.parent_folder}
Destination: {self.consolidated_folder}
Images per folder: {self.images_per_folder}
Exclude patterns: {', '.join(self.exclude_patterns)}
Dry run: {self.dry_run}
Selection mode: {self.selection_mode}

Note: Images will be written directly into the destination folder.

Continue with consolidation?"""
                
                result = messagebox.askyesno("Confirm Consolidation", message)
                root.destroy()
                
                if not result:
                    print("Consolidation cancelled by user.")
                    return self.stats
                    
            except Exception as e:
                print(f"Error showing confirmation dialog: {e}")
                print("Continuing with consolidation...")
        
        # Create consolidated folder if it doesn't exist
        if not self.dry_run:
            self.consolidated_folder.mkdir(exist_ok=True)
        
        # Find all terminal folders
        print("Finding terminal folders...")
        terminal_folders = self.find_terminal_folders()
        print(f"Found {len(terminal_folders)} terminal folders")
        
        # Process each terminal folder
        for i, folder_path in enumerate(terminal_folders, 1):
            print(f"Processing folder {i}/{len(terminal_folders)}: {folder_path.name}")
            
            try:
                # Get the selected images from this folder based on mode
                selected_images = self.select_images_from_folder(folder_path, self.images_per_folder)
                
                if not selected_images:
                    print(f"  No images found in {folder_path}")
                    self.stats['folders_skipped'] += 1
                    continue
                
                self.stats['folders_processed'] += 1
                self.stats['images_found'] += len(selected_images)
                
                # Copy selected images to consolidated folder
                for image_path in selected_images:
                    try:
                        # Create new filename with folder structure
                        relative_path = image_path.relative_to(self.parent_folder)
                        new_filename = self.create_consolidated_filename(image_path, relative_path)
                        destination = self.consolidated_folder / new_filename
                        
                        print(f"  {image_path.name} -> {new_filename}")
                        
                        if not self.dry_run:
                            # Copy the file
                            if not destination.exists() or self.overwrite_existing:
                                shutil.copy2(image_path, destination)
                                self.stats['images_copied'] += 1
                            else:
                                print(f"    Skipped (already exists): {new_filename}")
                        else:
                            self.stats['images_copied'] += 1
                            
                    except Exception as e:
                        error_msg = f"Error copying {image_path}: {e}"
                        print(f"    ERROR: {error_msg}")
                        self.stats['errors'].append(error_msg)
                        
            except Exception as e:
                error_msg = f"Error processing folder {folder_path}: {e}"
                print(f"  ERROR: {error_msg}")
                self.stats['errors'].append(error_msg)
                self.stats['folders_skipped'] += 1
        
        # Print final statistics
        print("\n" + "=" * 50)
        print("CONSOLIDATION COMPLETE")
        print("=" * 50)
        print(f"Folders processed: {self.stats['folders_processed']}")
        print(f"Folders skipped: {self.stats['folders_skipped']}")
        print(f"Images found: {self.stats['images_found']}")
        print(f"Images copied: {self.stats['images_copied']}")
        print(f"Errors: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\nErrors encountered:")
            for error in self.stats['errors']:
                print(f"  - {error}")
        
        return self.stats

def load_config_from_file(config_file: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default configuration.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return {}

def select_folder_gui() -> str:
    """Open GUI dialog to select parent folder."""
    # Create a root window (but hide it)
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.wm_attributes('-topmost', 1)  # Bring to front
    
    # Open folder selection dialog
    folder_path = filedialog.askdirectory(
        title="Select Parent Folder to Consolidate Images From",
        mustexist=True
    )
    
    root.destroy()  # Clean up the root window
    return folder_path


def select_output_folder_gui() -> str:
    """Open GUI to select output folder (must be empty)."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    try:
        while True:
            output_folder = filedialog.askdirectory(
                title="Select EMPTY Output Folder (Images will be stored directly here)",
                mustexist=True
            )
            if not output_folder:
                return ""

            # Inform the user about direct writes
            messagebox.showinfo(
                "Output Folder",
                "Images will be written directly into the selected folder."
            )

            # Require directory to be completely empty (no files or subfolders)
            try:
                entries = list(Path(output_folder).iterdir())
                if len(entries) > 0:
                    messagebox.showerror(
                        "Output Folder Not Empty",
                        "The selected directory is not empty. Please choose an EMPTY folder."
                    )
                    continue
            except Exception:
                # If we can't list entries, disallow selection
                messagebox.showerror(
                    "Output Folder Error",
                    "Unable to access the selected directory. Please choose a different folder."
                )
                continue

            return output_folder
    finally:
        root.destroy()


def select_selection_mode_gui() -> str:
    """Open a GUI dialog to choose the image selection mode."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)

    message = (
        "Which image(s) to take from each folder?\n\n"
        "Yes: Largest by file size\n"
        "No: First by name"
    )

    try:
        result = messagebox.askyesno("Select Image Selection Mode", message, parent=root)
    finally:
        root.destroy()

    return 'largest' if result else 'first'

def create_example_config():
    """Create an example configuration file."""
    example_config = {
        "parent_folder": "/path/to/your/parent/folder",
        "output_folder": "/path/to/your/output/folder",
        "consolidated_folder_name": "_consolidated",
        "images_per_folder": 1,
        "exclude_patterns": ["_Analysis", "_consolidated", "temp", "backup"],
        "dry_run": True,
        "overwrite_existing": False,
        "selection_mode": "largest"
    }
    
    with open("consolidate_config_example.json", "w") as f:
        json.dump(example_config, f, indent=4)
    
    print("Created example configuration file: consolidate_config_example.json")

def main():
    parser = argparse.ArgumentParser(
        description="Consolidate images from nested folder structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python consolidate_images.py --gui                    # Use GUI to select folder
  python consolidate_images.py /path/to/parent/folder   # Specify folder directly
  python consolidate_images.py --gui --images-per-folder 3 --dry-run
  python consolidate_images.py --config config.json
  python consolidate_images.py --create-example-config
        """
    )
    
    parser.add_argument('parent_folder', nargs='?', help='Parent folder to process')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--gui', action='store_true',
                       help='Use GUI dialog to select parent folder')
    parser.add_argument('--images-per-folder', type=int, default=1, 
                       help='Number of images to take from each folder (default: 1)')
    parser.add_argument('--consolidated-name', default='_consolidated',
                       help='Name of consolidated folder (default: _consolidated)')
    parser.add_argument('--output', help='Output folder to store consolidated images directly')
    parser.add_argument('--exclude', action='append', default=[],
                       help='Folder name patterns to exclude (can be used multiple times)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without actually copying files')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files in consolidated folder')
    parser.add_argument('--create-example-config', action='store_true',
                       help='Create an example configuration file')
    
    args = parser.parse_args()
    
    # Handle example config creation
    if args.create_example_config:
        create_example_config()
        return
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config_from_file(args.config)
    
    # Override with command line arguments
    if args.parent_folder:
        config['parent_folder'] = args.parent_folder
    if args.output:
        config['output_folder'] = args.output
    
    # Use GUI to select folder if requested or if no folder specified
    # Default to GUI mode when no arguments are provided
    use_gui = args.gui or (not config.get('parent_folder') and not args.create_example_config)
    
    if use_gui:
        print("Opening folder selection dialog...")
        try:
            selected_folder = select_folder_gui()
            if selected_folder:
                config['parent_folder'] = selected_folder
                print(f"Selected folder: {selected_folder}")
                # Ask user for selection mode
                try:
                    mode = select_selection_mode_gui()
                    config['selection_mode'] = mode
                    print(f"Selected selection mode: {mode}")
                except Exception as e:
                    print(f"Error with selection mode dialog: {e}")
                    print("Defaulting to 'largest' selection mode.")
                    config['selection_mode'] = 'largest'
                # Ask user for output folder
                try:
                    output_folder = select_output_folder_gui()
                    if not output_folder:
                        print("No output folder selected. Exiting.")
                        return
                    # Prevent output folder from being the same as the source folder
                    try:
                        if Path(output_folder).resolve() == Path(config['parent_folder']).resolve():
                            warn_root = tk.Tk()
                            warn_root.withdraw()
                            warn_root.wm_attributes('-topmost', 1)
                            messagebox.showwarning(
                                "Invalid Output Folder",
                                "Output folder cannot be the same as the source folder.\n\nPlease choose a different folder."
                            )
                            warn_root.destroy()
                            # Allow one more selection attempt
                            output_folder_retry = select_output_folder_gui()
                            if (not output_folder_retry) or (Path(output_folder_retry).resolve() == Path(config['parent_folder']).resolve()):
                                print("Invalid output folder selected. Exiting.")
                                return
                            output_folder = output_folder_retry
                    except Exception:
                        pass
                    config['output_folder'] = output_folder
                    print(f"Selected output folder: {output_folder}")
                except Exception as e:
                    print(f"Error with output folder selection: {e}")
                    print("Falling back to default output folder under parent.")
            else:
                print("No folder selected. Exiting.")
                return
        except Exception as e:
            print(f"Error with GUI folder selection: {e}")
            print("Falling back to command line...")
    
    if not config.get('parent_folder'):
        print("Error: Parent folder must be specified either via command line, config file, or GUI")
        parser.print_help()
        return
    
    # Set other configuration options
    config['images_per_folder'] = args.images_per_folder
    config['consolidated_folder_name'] = args.consolidated_name
    config['dry_run'] = args.dry_run
    config['overwrite_existing'] = args.overwrite

    # Ensure defaults when not using GUI or not provided
    if 'selection_mode' not in config:
        config['selection_mode'] = 'largest'
    
    # Merge exclude patterns
    default_excludes = ['_Analysis', '_consolidated']
    config_excludes = config.get('exclude_patterns', [])
    cli_excludes = args.exclude
    config['exclude_patterns'] = list(set(default_excludes + config_excludes + cli_excludes))
    
    # Validate parent folder exists
    parent_path = Path(config['parent_folder'])
    if not parent_path.exists():
        print(f"Error: Parent folder does not exist: {parent_path}")
        return
    
    if not parent_path.is_dir():
        print(f"Error: Parent folder is not a directory: {parent_path}")
        return

    # Validate output folder constraints when provided non-GUI
    output_folder_cfg = config.get('output_folder')
    if output_folder_cfg:
        output_path = Path(output_folder_cfg)
        try:
            if output_path.exists():
                # Must be empty
                entries = list(output_path.iterdir())
                if len(entries) > 0:
                    print(f"Error: Output folder must be empty: {output_path}")
                    return
            # Prevent using source folder as output
            try:
                if output_path.resolve() == parent_path.resolve():
                    print("Error: Output folder cannot be the same as the source folder.")
                    return
            except Exception:
                pass
        except Exception as e:
            print(f"Error validating output folder: {e}")
            return
    
    # Create consolidator and run
    consolidator = ImageConsolidator(config)
    stats = consolidator.consolidate_images(show_gui_confirmation=use_gui)
    
    # Return appropriate exit code
    if stats['errors']:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main() 