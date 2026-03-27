import json
import os
import tkinter as tk
from tkinter import messagebox, filedialog
from pathlib import Path

class FileCache:
    def __init__(self, cache_dir=".file_cache"):
        """
        Initialize file cache system
        
        Args:
            cache_dir: Directory to store cache files (should be in .gitignore)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create .gitignore entry if it doesn't exist
        gitignore_path = Path(".gitignore")
        if gitignore_path.exists():
            with open(gitignore_path, 'r') as f:
                content = f.read()
            if cache_dir not in content:
                with open(gitignore_path, 'a') as f:
                    f.write(f"\n# File cache directory\n{cache_dir}/\n")
        else:
            with open(gitignore_path, 'w') as f:
                f.write(f"# File cache directory\n{cache_dir}/\n")
    
    def _get_cache_file(self, script_name):
        """Get cache file path for a given script"""
        # Remove .py extension and clean script name
        clean_name = Path(script_name).stem
        return self.cache_dir / f"{clean_name}_file_cache.json"
    
    def save_file_paths(self, script_name, file_paths, labels=None):
        """
        Save file paths to cache
        
        Args:
            script_name: Name of the calling script
            file_paths: List of file paths or single file path
            labels: Optional list of labels for the files (e.g., ['software_file', 'manual_file'])
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        if labels is None:
            labels = [f"file_{i+1}" for i in range(len(file_paths))]
        elif len(labels) != len(file_paths):
            raise ValueError("Number of labels must match number of file paths")
        
        cache_data = {
            'script_name': script_name,
            'files': {}
        }
        
        for label, file_path in zip(labels, file_paths):
            cache_data['files'][label] = {
                'path': str(file_path),
                'exists': os.path.exists(file_path)
            }
        
        cache_file = self._get_cache_file(script_name)
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def load_file_paths(self, script_name):
        """
        Load cached file paths for a script
        
        Args:
            script_name: Name of the calling script
            
        Returns:
            dict: Dictionary with file labels as keys and paths as values, or None if no cache
        """
        cache_file = self._get_cache_file(script_name)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if files still exist
            result = {}
            all_exist = True
            
            for label, file_info in cache_data['files'].items():
                file_path = file_info['path']
                if os.path.exists(file_path):
                    result[label] = file_path
                else:
                    all_exist = False
            
            return result if all_exist else None
            
        except (json.JSONDecodeError, KeyError):
            return None
    
    def prompt_use_cache_or_select(self, script_name, file_labels, file_types=None):
        """
        Check for cached files and prompt user to use cache or select new files
        
        Args:
            script_name: Name of the calling script
            file_labels: List of labels for files to select (e.g., ['software_file', 'manual_file'])
            file_types: List of tuples for file types (e.g., [("Excel files", "*.xlsx")])
            
        Returns:
            dict: Dictionary with file labels as keys and selected paths as values
        """
        if file_types is None:
            file_types = [("All files", "*.*") for _ in file_labels]
        elif len(file_types) != len(file_labels):
            raise ValueError("Number of file types must match number of file labels")
        
        # Try to load cached paths
        cached_paths = self.load_file_paths(script_name)
        
        if cached_paths and len(cached_paths) == len(file_labels):
            # Show cache options to user
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            cache_info = "\n".join([f"{label}: {os.path.basename(path)}" 
                                  for label, path in cached_paths.items()])
            
            use_cache = messagebox.askyesno(
                "Use Cached Files?",
                f"Found previously selected files:\n\n{cache_info}\n\nWould you like to use these files again?",
                parent=root
            )
            
            root.destroy()
            
            if use_cache:
                return cached_paths
        
        # Select new files
        selected_paths = {}
        
        for i, (label, file_type) in enumerate(zip(file_labels, file_types)):
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            file_path = filedialog.askopenfilename(
                title=f"Select {label.replace('_', ' ').title()}",
                filetypes=[file_type]
            )
            
            root.destroy()
            
            if not file_path:
                return None  # User cancelled
            
            selected_paths[label] = file_path
        
        # Save new selections to cache
        if selected_paths:
            file_paths = [selected_paths[label] for label in file_labels]
            self.save_file_paths(script_name, file_paths, file_labels)
        
        return selected_paths

# Global instance for easy use
file_cache = FileCache()

def prompt_files_with_cache(script_name, file_configs):
    """
    Convenience function to prompt for files with caching
    
    Args:
        script_name: Name of the calling script (use __file__)
        file_configs: List of tuples (label, title, file_types)
                     e.g., [('software_file', 'Software Analysis File', [("Excel files", "*.xlsx")])]
    
    Returns:
        dict: Dictionary with file labels as keys and selected paths as values
    """
    labels = [config[0] for config in file_configs]
    file_types = [config[2][0] if config[2] else ("All files", "*.*") for config in file_configs]
    
    return file_cache.prompt_use_cache_or_select(script_name, labels, file_types) 