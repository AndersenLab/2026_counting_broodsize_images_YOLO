import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
import tkinter as tk
import re
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import os
import json
import glob
import xml.etree.ElementTree as ET

# Check for required dependencies
try:
    import openpyxl
except ImportError:
    print("Error: openpyxl is required to read Excel files.")
    print("Please install it with: pip install openpyxl")
    exit(1)

def find_cvat_xml_files(directory):
    """Find CVAT XML annotation files in directory"""
    xml_files = glob.glob(os.path.join(directory, "*.xml"))
    
    # Filter for likely CVAT annotation files
    cvat_candidates = []
    for xml_file in xml_files:
        basename = os.path.basename(xml_file).lower()
        # Prefer files with annotation, cvat, or labels in name
        if any(keyword in basename for keyword in ["annotation", "cvat", "labels"]):
            cvat_candidates.append(xml_file)
    
    # If no files with keywords found, use all XML files
    if not cvat_candidates:
        cvat_candidates = xml_files
    
    return cvat_candidates

def find_raw_excel_file(directory):
    """Find _raw.xlsx file in directory"""
    excel_files = glob.glob(os.path.join(directory, "*_raw.xlsx"))
    if excel_files:
        return excel_files[0]
    return None

def load_cvat_xml(xml_path):
    """
    Load CVAT XML file and extract manual counts per image filename.
    
    Args:
        xml_path: Path to CVAT XML annotation file
    
    Returns:
        Dictionary mapping filename -> manual_count (number of worm boxes)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Failed to parse CVAT XML file: {e}")
    
    manual_counts = {}
    
    # CVAT XML structure: <annotations> -> <image> -> <box>
    for image_elem in root.findall('image'):
        # Get image filename from 'name' attribute
        filename = image_elem.get('name', None)
        if not filename:
            continue
        
        # Count all <box> elements with label="worm" (or any box if no label filter)
        worm_boxes = image_elem.findall('box[@label="worm"]')
        # If no boxes with label="worm", count all boxes
        if len(worm_boxes) == 0:
            worm_boxes = image_elem.findall('box')
        
        # If filename already exists (from another XML file), sum the counts
        if filename in manual_counts:
            manual_counts[filename] += len(worm_boxes)
        else:
            manual_counts[filename] = len(worm_boxes)
    
    return manual_counts

def load_all_cvat_xml_files(xml_paths):
    """
    Load multiple CVAT XML files and combine their manual counts.
    
    Args:
        xml_paths: List of paths to CVAT XML annotation files
    
    Returns:
        Dictionary mapping filename -> manual_count (combined from all XML files)
    """
    all_manual_counts = {}
    total_images = 0
    
    for xml_path in xml_paths:
        try:
            counts = load_cvat_xml(xml_path)
            # Merge counts (sum if filename appears in multiple files)
            for filename, count in counts.items():
                if filename in all_manual_counts:
                    all_manual_counts[filename] += count
                else:
                    all_manual_counts[filename] = count
            total_images += len(counts)
            print(f"✅ Loaded {len(counts)} images from: {os.path.basename(xml_path)}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to load {os.path.basename(xml_path)}: {e}")
            continue
    
    print(f"✅ Total: {len(all_manual_counts)} unique images from {len(xml_paths)} XML file(s)")
    return all_manual_counts

def load_software_data(excel_path):
    """
    Load software analysis data from Excel file.
    Expects "N worms" sheet with 'fname' (or 'filename') and 'Well 1' columns.
    
    Returns:
        DataFrame with filename and software_count columns
    """
    try:
        # Read the "N worms" sheet
        df = pd.read_excel(excel_path, sheet_name="N worms")
        
        # Find filename column - try multiple possible names
        filename_col = None
        for col_name in ['fname', 'filename', 'fname (hr)', 'time (hr)']:
            if col_name in df.columns:
                # Check if this column contains filenames (strings that look like filenames)
                sample_val = df[col_name].dropna().iloc[0] if len(df[col_name].dropna()) > 0 else None
                if sample_val is not None:
                    # If it's a string that looks like a filename (contains .png, .jpg, etc.) or is a string
                    if isinstance(sample_val, str) and ('.png' in sample_val.lower() or '.jpg' in sample_val.lower() or len(str(sample_val)) > 10):
                        filename_col = col_name
                        break
                    # If x_axis is "time", we can't use it - need to look for another column
                    if col_name == 'time (hr)' and isinstance(sample_val, (int, float)):
                        continue  # This is time, not filename
        
        # If no filename column found, check if first column looks like filenames
        if filename_col is None and len(df.columns) > 0:
            first_col = df.columns[0]
            sample_val = df[first_col].dropna().iloc[0] if len(df[first_col].dropna()) > 0 else None
            if sample_val is not None and isinstance(sample_val, str) and ('.png' in sample_val.lower() or '.jpg' in sample_val.lower()):
                filename_col = first_col
                print(f"   Using first column '{first_col}' as filename column")
        
        if filename_col is None:
            raise ValueError("Software data must contain 'fname' or 'filename' column, or first column must contain filenames")
        
        # Check for Well 1 column
        if 'Well 1' not in df.columns:
            raise ValueError("Software data must contain 'Well 1' column")
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'filename': df[filename_col],
            'software_count': df['Well 1']
        })
        
        # Remove rows with NaN values
        result_df = result_df.dropna(subset=['filename', 'software_count'])
        
        # Convert filename to string and normalize
        result_df['filename'] = result_df['filename'].astype(str)
        
        print(f"✅ Loaded {len(result_df)} software analysis entries from: {os.path.basename(excel_path)}")
        print(f"   Using '{filename_col}' column for filenames")
        return result_df
    
    except Exception as e:
        raise ValueError(f"Error loading software data: {str(e)}")

def normalize_filename(filename):
    """
    Normalize filename for matching (remove path, handle case, etc.)
    """
    # Extract just the filename if it includes a path
    basename = os.path.basename(str(filename))
    # Remove any leading/trailing whitespace
    basename = basename.strip()
    return basename

def match_data(software_df, manual_counts_dict):
    """
    Match software and manual data based on filename.
    
    Args:
        software_df: DataFrame with 'filename' and 'software_count' columns
        manual_counts_dict: Dictionary mapping filename -> manual_count
    
    Returns:
        DataFrame with matched data
    """
    matched_data = []
    unmatched_software = []
    unmatched_manual = []
    
    print(f"\nMatching {len(software_df)} software entries with {len(manual_counts_dict)} manual entries...")
    
    # Normalize manual counts dictionary keys
    normalized_manual = {}
    for filename, count in manual_counts_dict.items():
        normalized_key = normalize_filename(filename)
        # Handle duplicates by keeping the first one (or you could sum them)
        if normalized_key not in normalized_manual:
            normalized_manual[normalized_key] = count
        else:
            print(f"⚠️  Warning: Duplicate filename in VGG JSON: {normalized_key}, using first count")
    
    # Match software entries
    for idx, row in software_df.iterrows():
        software_filename = normalize_filename(row['filename'])
        software_count = row['software_count']
        
        if software_filename in normalized_manual:
            manual_count = normalized_manual[software_filename]
            matched_data.append({
                'filename': software_filename,
                'software_count': software_count,
                'manual_count': manual_count
            })
        else:
            unmatched_software.append({
                'filename': software_filename,
                'software_count': software_count
            })
    
    # Find unmatched manual entries
    matched_filenames = {normalize_filename(row['filename']) for row in matched_data}
    for filename, count in normalized_manual.items():
        if filename not in matched_filenames:
            unmatched_manual.append({
                'filename': filename,
                'manual_count': count
            })
    
    # Print detailed unmatched analysis
    if unmatched_manual:
        print(f"\n❌ UNMATCHED MANUAL ENTRIES ({len(unmatched_manual)}):")
        print("=" * 60)
        print("These manual count entries could not be matched with software analysis:")
        for entry in unmatched_manual[:20]:  # Show first 20
            print(f"    {entry['filename']} → Manual count: {entry['manual_count']}")
        if len(unmatched_manual) > 20:
            print(f"    ... and {len(unmatched_manual) - 20} more")
    
    if unmatched_software:
        print(f"\n❌ UNMATCHED SOFTWARE ENTRIES ({len(unmatched_software)}):")
        print("=" * 60)
        print("These software analysis entries could not be matched with manual counts:")
        for entry in unmatched_software[:20]:  # Show first 20
            print(f"    {entry['filename']} → Software count: {entry['software_count']}")
        if len(unmatched_software) > 20:
            print(f"    ... and {len(unmatched_software) - 20} more")
    
    if not unmatched_manual and not unmatched_software:
        print(f"\n✅ Perfect matching! All entries successfully matched.")
    
    return pd.DataFrame(matched_data)

def plot_correlation(matched_df, show_plot=True, save_path=None, analysis_dir=None):
    """Plot correlation between software and manual counts"""
    if matched_df.empty:
        if show_plot:
            messagebox.showwarning("Warning", "No matched data found for correlation plot")
        else:
            print("⚠️  Warning: No matched data found for correlation plot")
        return None, None, 0
    
    # Remove any NaN values
    clean_df = matched_df.dropna(subset=['software_count', 'manual_count'])
    
    if clean_df.empty:
        if show_plot:
            messagebox.showwarning("Warning", "No valid data points for correlation plot")
        else:
            print("⚠️  Warning: No valid data points for correlation plot")
        return None, None, 0
    
    software_counts = clean_df['software_count'].values
    manual_counts = clean_df['manual_count'].values
    
    # Calculate correlation and R-squared
    correlation, p_value = pearsonr(software_counts, manual_counts)
    r_squared = r2_score(manual_counts, software_counts)
    
    # Calculate error in number of worms: software - manual
    # Positive values indicate software over-counting, negative indicate under-counting
    worm_errors = software_counts - manual_counts
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: Correlation plot
    ax1.scatter(manual_counts, software_counts, alpha=0.6, s=50)
    
    # y=x line
    min_val = min(min(software_counts), min(manual_counts))
    max_val = max(max(software_counts), max(manual_counts))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', 
             label=f'y=x line', linewidth=2)
    
    # Best fit line
    z = np.polyfit(manual_counts, software_counts, 1)
    p = np.poly1d(z)
    ax1.plot(manual_counts, p(manual_counts), 'b-', 
             label=f'Best fit: y={z[0]:.3f}x + {z[1]:.3f}', linewidth=2)
    
    # Labels and title for correlation plot
    ax1.set_xlabel('Manual Count (CVAT)', fontsize=12)
    ax1.set_ylabel('Software Count', fontsize=12)
    
    title = f'Software vs Manual (CVAT XML) Worm Count Correlation\n' + \
            f'R² = {r_squared:.3f}, r = {correlation:.3f}, p = {p_value:.3f}'
    
    ax1.set_title(title, fontsize=14)
    
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Make axes equal for better comparison
    ax1.axis('equal')
    
    # Add text box with statistics
    stats_text = f'N = {len(clean_df)}\nR² = {r_squared:.3f}\nr = {correlation:.3f}'
    ax1.text(0.05, 0.75, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Right subplot: Worm count error histogram
    mean_error = np.mean(worm_errors)
    std_error = np.std(worm_errors)
    
    ax2.hist(worm_errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Error (Software - Manual Count)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Distribution of Worm Count Error\nMean: {mean_error:.1f}, Std: {std_error:.1f}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add vertical line at mean
    ax2.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.1f}')
    # Add vertical line at zero for reference
    ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Perfect Agreement')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot if path provided or save to analysis directory
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved correlation plot to: {save_path}")
    elif analysis_dir:
        png_path = os.path.join(analysis_dir, "correlation_vs_cvat.png")
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved correlation plot to: {png_path}")
    else:
        png_path = "correlation_vs_cvat_plot.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved correlation plot to: {png_path}")
    
    if show_plot:
        plt.show()
    
    if not show_plot:
        plt.close()  # Clean up the figure to save memory when not showing
    
    return correlation, r_squared, len(clean_df)

def find_analysis_directory(parent_dir):
    """
    Find the _Analysis subdirectory.
    If parent_dir is already _Analysis, return it.
    Otherwise, look for _Analysis subdirectory.
    """
    # Check if parent_dir is already _Analysis
    if os.path.basename(parent_dir) == "_Analysis":
        return parent_dir
    
    # Look for _Analysis subdirectory
    analysis_dir = os.path.join(parent_dir, "_Analysis")
    if os.path.exists(analysis_dir) and os.path.isdir(analysis_dir):
        return analysis_dir
    
    # If not found, return parent_dir (user might have passed _Analysis directly)
    return parent_dir

def main(parent_directory=None, automated_mode=False, min_manual_count=0):
    """
    Main function to run the correlation analysis between software counts and CVAT manual labels.
    
    Args:
        parent_directory (str): Path to parent directory or _Analysis subfolder
        automated_mode (bool): If True, suppresses plot display and uses provided directory
        min_manual_count (int): Minimum manual count threshold for filtering
    """
    print("Worm Count Correlation Analysis (Software vs CVAT XML)")
    print("=" * 60)
    print(f"Minimum manual count threshold: {min_manual_count}")
    
    # Get parent directory
    if parent_directory and os.path.exists(parent_directory):
        parent_dir = parent_directory
        print(f"✅ Using provided directory: {parent_dir}")
    elif automated_mode:
        print("❌ No valid parent directory provided for automated mode!")
        return
    else:
        # Interactive mode - prompt for directory
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        parent_dir = filedialog.askdirectory(
            title="Select Parent Directory (or _Analysis subfolder)"
        )
        root.destroy()
        
        if not parent_dir:
            print("❌ No directory selected")
            return
    
    # Find analysis directory
    analysis_dir = find_analysis_directory(parent_dir)
    print(f"📁 Analysis directory: {analysis_dir}")
    
    # Find CVAT XML files
    print("\nStep 1: Looking for CVAT XML annotation files...")
    cvat_xml_paths = find_cvat_xml_files(analysis_dir)
    
    if not cvat_xml_paths:
        print("❌ CVAT XML annotation files not found in analysis directory!")
        if not automated_mode:
            messagebox.showerror("Error", "CVAT XML annotation files not found in analysis directory!")
        return
    
    print(f"✅ Found {len(cvat_xml_paths)} CVAT XML file(s):")
    for xml_path in cvat_xml_paths:
        print(f"   - {os.path.basename(xml_path)}")
    
    # Find Excel file
    print("\nStep 2: Looking for _raw.xlsx file...")
    excel_path = find_raw_excel_file(analysis_dir)
    
    if not excel_path:
        print("❌ _raw.xlsx file not found in analysis directory!")
        if not automated_mode:
            messagebox.showerror("Error", "_raw.xlsx file not found in analysis directory!")
        return
    
    print(f"✅ Found Excel file: {os.path.basename(excel_path)}")
    
    # Load CVAT XML files (manual counts)
    print("\nStep 3: Loading manual counts from CVAT XML files...")
    try:
        manual_counts = load_all_cvat_xml_files(cvat_xml_paths)
    except Exception as e:
        print(f"❌ Error loading CVAT XML files: {e}")
        if not automated_mode:
            messagebox.showerror("Error", f"Error loading CVAT XML files: {e}")
        return
    
    # Load software data
    print("\nStep 4: Loading software counts from Excel...")
    try:
        software_df = load_software_data(excel_path)
    except Exception as e:
        print(f"❌ Error loading software data: {e}")
        if not automated_mode:
            messagebox.showerror("Error", f"Error loading software data: {e}")
        return
    
    # Match the data
    print("\nStep 5: Matching software and manual data...")
    matched_df = match_data(software_df, manual_counts)
    
    if matched_df.empty:
        print("❌ No matched data found. Cannot create correlation plot.")
        if not automated_mode:
            messagebox.showwarning("Warning", "No matched data found. Cannot create correlation plot.")
        return
    
    # Apply minimum count filter
    if len(matched_df) > 0:
        print(f"\nStep 5.5: Applying minimum manual count filter (>= {min_manual_count})...")
        original_count = len(matched_df)
        
        # Identify entries to be filtered out
        filtered_out = matched_df[matched_df['manual_count'] < min_manual_count]
        
        if len(filtered_out) > 0:
            print(f"Filtering out {len(filtered_out)} entries with manual count < {min_manual_count}")
        
        # Apply the filter
        matched_df = matched_df[matched_df['manual_count'] >= min_manual_count]
        
        print(f"Retained {len(matched_df)} entries after filtering (was {original_count})")
        
        if len(matched_df) == 0:
            print("No entries remaining after filtering. Consider lowering the minimum count threshold.")
            return
    
    # Print matching report
    print("\nMatching Report:")
    print("=" * 40)
    print(f"Total manual entries: {len(manual_counts)}")
    print(f"Total software entries: {len(software_df)}")
    print(f"Successfully matched: {len(matched_df)}")
    if len(manual_counts) > 0:
        print(f"Match rate (manual): {len(matched_df)/len(manual_counts)*100:.1f}%")
    if len(software_df) > 0:
        print(f"Match rate (software): {len(matched_df)/len(software_df)*100:.1f}%")
    
    # Plot correlation
    if len(matched_df) > 0:
        if automated_mode:
            print("\nStep 6: Analyzing correlation and saving plot...")
        else:
            print("\nStep 6: Plotting correlation...")
        show_plot = not automated_mode  # Show plot only in interactive mode
        correlation, r_squared, n_points = plot_correlation(
            matched_df, 
            show_plot=show_plot, 
            save_path=None,
            analysis_dir=analysis_dir
        )
        
        print(f"\nCorrelation Results:")
        print(f"  Data points: {n_points}")
        print(f"  R-squared: {r_squared:.3f}")
        print(f"  Correlation: {correlation:.3f}")
        
        # Save matched data to CSV for further analysis
        output_file = os.path.join(analysis_dir, "matched_worm_counts_vs_cvat.csv")
        matched_df.to_csv(output_file, index=False)
        print(f"\nMatched data saved to: {output_file}")
        
        # Calculate and display top 20 frames with largest errors
        print(f"\nTop 20 Frames with Largest Errors:")
        print("=" * 80)
        
        # Calculate absolute error
        matched_df_copy = matched_df.copy()
        matched_df_copy['absolute_error'] = abs(matched_df_copy['software_count'] - matched_df_copy['manual_count'])
        
        # Sort by absolute error (descending) and take top 20
        top_errors = matched_df_copy.nlargest(20, 'absolute_error')
        
        if len(top_errors) > 0:
            print(f"{'Rank':<6} {'Filename':<50} {'Manual':<8} {'Software':<10} {'Error':<8}")
            print("-" * 80)
            
            for rank, (idx, row) in enumerate(top_errors.iterrows(), start=1):
                error = row['software_count'] - row['manual_count']
                filename = row['filename'][:47] + "..." if len(row['filename']) > 50 else row['filename']
                print(f"{rank:<6} {filename:<50} {int(row['manual_count']):<8} {int(row['software_count']):<10} {error:+.1f}")
        else:
            print("No data available for error analysis.")
    
    else:
        print("No matched data found. Cannot create correlation plot.")

if __name__ == "__main__":
    # Set minimum manual count threshold (default: 0)
    MIN_MANUAL_COUNT_THRESHOLD = 0
    
    # When run directly, use interactive mode (GUI prompts enabled)
    main(automated_mode=False, min_manual_count=MIN_MANUAL_COUNT_THRESHOLD)

