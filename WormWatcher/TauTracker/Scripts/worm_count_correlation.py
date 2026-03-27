import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
import tkinter as tk
import re
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import os
from file_cache import file_cache

# Check for required dependencies
try:
    import openpyxl
except ImportError:
    print("Error: openpyxl is required to read Excel files.")
    print("Please install it with: pip install openpyxl")
    exit(1)

def prompt_file_selection(title, filetypes=[("Excel files", "*.xlsx")]):
    """Prompt user to select a file"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    
    root.destroy()
    return file_path

def parse_image_filename(filename):
    """
    Parse image filename to extract strain, day, and replicate info
    Example: 20250415_day1_BRC20530_r5_series0000frame00000_t=0.00 s.png
    Returns: {'strain': 'BRC20530', 'day': 'day1', 'replicate': 'r5'}
    """
    # Remove file extension
    base_name = os.path.splitext(filename)[0]
    
    # Extract strain (comes after day pattern in filename) - case insensitive
    # Pattern: date_day#_STRAIN_r#_ where STRAIN can be BRC20530, ECA2266, VX34, etc.
    strain_match = re.search(r'_(?:day|d)\d+_([A-Z]{2,4}\d+)_', base_name, re.IGNORECASE)
    strain = strain_match.group(1).upper() if strain_match else None
    
    # Extract day (pattern like day1, day2, d1, d2, etc.) - case insensitive
    day_match = re.search(r'(?:day|d)(\d+)', base_name, re.IGNORECASE)
    if day_match:
        day_number = day_match.group(1)
        day = f"day{day_number}"  # Normalize to "dayN" format
    else:
        day = None
    
    # Extract replicate (pattern like r1, r2, r5, etc.) - case insensitive
    replicate_match = re.search(r'_r(\d+)_', base_name, re.IGNORECASE)
    replicate = f"r{replicate_match.group(1)}" if replicate_match else None
    
    return {
        'strain': strain,
        'day': day,
        'replicate': replicate
    }

def parse_day_replicate(day_replicate_str):
    """
    Parse Day_Replicate column to extract day and replicate
    Example: "day1_r1" or "d1_r1" -> {'day': 'day1', 'replicate': 'r1'}
    """
    if pd.isna(day_replicate_str):
        return {'day': None, 'replicate': None}
    
    parts = str(day_replicate_str).split('_')
    if len(parts) >= 2:
        day_part = parts[0].lower()
        replicate_part = parts[1].lower()
        
        # Normalize day format: both "day5" and "d5" become "day5"
        day_match = re.search(r'(?:day|d)(\d+)', day_part, re.IGNORECASE)
        if day_match:
            day_number = day_match.group(1)
            day = f"day{day_number}"
        else:
            day = day_part
        
        return {'day': day, 'replicate': replicate_part}
    
    return {'day': None, 'replicate': None}

def load_software_data(file_path):
    """Load software analysis data"""
    try:
        df = pd.read_excel(file_path)
        
        # Check if required columns exist
        if 'fname' not in df.columns or 'Well 1' not in df.columns:
            raise ValueError("Software data must contain 'fname' and 'Well 1' columns")
        
        # Parse image filenames
        parsed_data = []
        for idx, row in df.iterrows():
            parsed_info = parse_image_filename(row['fname'])
            parsed_data.append({
                'filename': row['fname'],
                'software_count': row['Well 1'],
                'strain': parsed_info['strain'],
                'day': parsed_info['day'],
                'replicate': parsed_info['replicate']
            })
        
        return pd.DataFrame(parsed_data)
    
    except Exception as e:
        messagebox.showerror("Error", f"Error loading software data: {str(e)}")
        return None

def load_manual_data(file_path):
    """Load manual count data from Excel file with multiple tabs"""
    try:
        # Get all sheet names
        xl_file = pd.ExcelFile(file_path)
        sheet_names = xl_file.sheet_names
        
        print(f"Found {len(sheet_names)} sheets in manual data file:")
        for sheet in sheet_names:
            print(f"  - {sheet}")
        
        manual_data = []
        processed_sheets = []
        skipped_sheets = []
        
        for sheet_name in sheet_names:
            # More flexible strain pattern - matches various strain naming conventions
            # (BRC20530, ECA2266, ED3102, JU3237, NIC1667, PB420, QG1005, VX34, etc.)
            strain_pattern_match = re.search(r'[A-Z]{2,4}\d+', sheet_name, re.IGNORECASE)
            
            # Try to read the sheet to check for data structure
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Check if this looks like a strain data sheet
                has_day_replicate = 'Day_Replicate' in df.columns
                has_counts = 'Counts' in df.columns
                
                if not (has_day_replicate and has_counts):
                    # Try alternative column names
                    day_rep_cols = [col for col in df.columns if 'day' in col.lower() and ('rep' in col.lower() or 'r' in col.lower())]
                    count_cols = [col for col in df.columns if 'count' in col.lower()]
                    
                    if day_rep_cols and count_cols:
                        print(f"Sheet '{sheet_name}': Using alternative columns '{day_rep_cols[0]}' and '{count_cols[0]}'")
                        df = df.rename(columns={day_rep_cols[0]: 'Day_Replicate', count_cols[0]: 'Counts'})
                        has_day_replicate = has_counts = True
                
                # If it has the right structure OR matches strain pattern, process it
                if (has_day_replicate and has_counts) or strain_pattern_match:
                    if not (has_day_replicate and has_counts):
                        print(f"Warning: Sheet '{sheet_name}' missing required columns, skipping")
                        skipped_sheets.append(f"{sheet_name} (missing columns)")
                        continue
                    
                    processed_sheets.append(sheet_name)
                    print(f"Processing sheet: {sheet_name}")
                    
                    # Parse each row
                    row_count = 0
                    for idx, row in df.iterrows():
                        if pd.isna(row['Day_Replicate']) or pd.isna(row['Counts']):
                            continue
                        
                        day_rep_info = parse_day_replicate(row['Day_Replicate'])
                        
                        if day_rep_info['day'] and day_rep_info['replicate']:
                            manual_data.append({
                                'strain': sheet_name.upper(),  # Keep original sheet name as strain
                                'day': day_rep_info['day'],
                                'replicate': day_rep_info['replicate'],
                                'manual_count': row['Counts']
                            })
                            row_count += 1
                    
                    print(f"  Loaded {row_count} data points from {sheet_name}")
                else:
                    skipped_sheets.append(f"{sheet_name} (not strain data)")
                    
            except Exception as e:
                print(f"Error reading sheet '{sheet_name}': {e}")
                skipped_sheets.append(f"{sheet_name} (read error)")
                continue
        
        print(f"\nSummary:")
        print(f"  Processed sheets: {len(processed_sheets)}")
        for sheet in processed_sheets:
            print(f"    - {sheet}")
        
        if skipped_sheets:
            print(f"  Skipped sheets: {len(skipped_sheets)}")
            for sheet in skipped_sheets:
                print(f"    - {sheet}")
        
        result_df = pd.DataFrame(manual_data)
        print(f"  Total manual data points loaded: {len(result_df)}")
        
        return result_df
    
    except Exception as e:
        messagebox.showerror("Error", f"Error loading manual data: {str(e)}")
        return None

def match_data(software_df, manual_df):
    """Match software and manual data based on strain, day, and replicate"""
    matched_data = []
    unmatched_manual = []
    unmatched_software = []
    
    print(f"\nMatching {len(manual_df)} manual entries with {len(software_df)} software entries...")
    
    for idx, manual_row in manual_df.iterrows():
        # Find matching software entry
        matches = software_df[
            (software_df['strain'] == manual_row['strain']) &
            (software_df['day'] == manual_row['day']) &
            (software_df['replicate'] == manual_row['replicate'])
        ]
        
        if not matches.empty:
            # Take the first match if multiple exist
            software_row = matches.iloc[0]
            matched_data.append({
                'strain': manual_row['strain'],
                'day': manual_row['day'],
                'replicate': manual_row['replicate'],
                'filename': software_row['filename'],
                'software_count': software_row['software_count'],
                'manual_count': manual_row['manual_count']
            })
            
            if len(matches) > 1:
                print(f"Warning: Multiple software matches found for {manual_row['strain']} {manual_row['day']} {manual_row['replicate']}, using first match")
        else:
            # Track unmatched manual entry
            unmatched_manual.append({
                'strain': manual_row['strain'],
                'day': manual_row['day'],
                'replicate': manual_row['replicate'],
                'manual_count': manual_row['manual_count']
            })
    
    # Find software entries that didn't match any manual entry
    matched_software_indices = []
    for matched_entry in matched_data:
        software_matches = software_df[
            (software_df['strain'] == matched_entry['strain']) &
            (software_df['day'] == matched_entry['day']) &
            (software_df['replicate'] == matched_entry['replicate'])
        ]
        if not software_matches.empty:
            matched_software_indices.extend(software_matches.index.tolist())
    
    unmatched_software_df = software_df[~software_df.index.isin(matched_software_indices)]
    for idx, software_row in unmatched_software_df.iterrows():
        unmatched_software.append({
            'strain': software_row['strain'],
            'day': software_row['day'],
            'replicate': software_row['replicate'],
            'software_count': software_row['software_count'],
            'filename': software_row['filename']
        })
    
    # Print detailed unmatched analysis
    if unmatched_manual:
        print(f"\n❌ UNMATCHED MANUAL ENTRIES ({len(unmatched_manual)}):")
        print("=" * 60)
        print("These manual count entries could not be matched with software analysis:")
        
        # Group by strain for better organization
        manual_by_strain = {}
        for entry in unmatched_manual:
            strain = entry['strain']
            if strain not in manual_by_strain:
                manual_by_strain[strain] = []
            manual_by_strain[strain].append(entry)
        
        for strain, entries in manual_by_strain.items():
            print(f"\n  {strain}:")
            for entry in entries:
                print(f"    {entry['day']} {entry['replicate']} → Manual count: {entry['manual_count']}")
                
                # Suggest possible matches
                possible_matches = software_df[software_df['strain'] == strain]
                if not possible_matches.empty:
                    available_combos = possible_matches[['day', 'replicate']].drop_duplicates()
                    combo_strings = [f"{row['day']} {row['replicate']}" for _, row in available_combos.iterrows()]
                    print(f"      Available software data for {strain}: {', '.join(combo_strings)}")
                else:
                    print(f"      No software data found for strain {strain}")
    
    if unmatched_software:
        print(f"\n❌ UNMATCHED SOFTWARE ENTRIES ({len(unmatched_software)}):")
        print("=" * 60)
        print("These software analysis entries could not be matched with manual counts:")
        
        # Group by strain for better organization
        software_by_strain = {}
        for entry in unmatched_software:
            strain = entry['strain']
            if strain not in software_by_strain:
                software_by_strain[strain] = []
            software_by_strain[strain].append(entry)
        
        for strain, entries in software_by_strain.items():
            print(f"\n  {strain}:")
            for entry in entries:
                print(f"    {entry['day']} {entry['replicate']} → Software count: {entry['software_count']}")
                print(f"      File: {entry['filename']}")
                
                # Suggest possible matches
                possible_matches = manual_df[manual_df['strain'] == strain]
                if not possible_matches.empty:
                    available_combos = possible_matches[['day', 'replicate']].drop_duplicates()
                    combo_strings = [f"{row['day']} {row['replicate']}" for _, row in available_combos.iterrows()]
                    print(f"      Available manual data for {strain}: {', '.join(combo_strings)}")
                else:
                    print(f"      No manual data found for strain {strain}")
    
    if not unmatched_manual and not unmatched_software:
        print(f"\n✅ Perfect matching! All entries successfully matched.")
    
    return pd.DataFrame(matched_data)

def plot_correlation(matched_df, show_plot=True, save_path=None, param_info=None, software_file_path=None):
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
    ax1.set_xlabel('Manual Count', fontsize=12)
    ax1.set_ylabel('Software Count', fontsize=12)
    
    # Create title with parameter info if provided
    if param_info:
        title = f'Software vs Manual Worm Count Correlation\n' + \
                f'Parameters: pix_conf_thresh={param_info["pix_conf_thresh"]}, overlap_thresh={param_info["overlap_thresh"]}\n' + \
                f'R² = {r_squared:.3f}, r = {correlation:.3f}, p = {p_value:.3f}'
    else:
        title = f'Software vs Manual Worm Count Correlation\n' + \
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
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved correlation plot to: {save_path}")
    else:
        # Always save a PNG in the software file directory with the same base name
        if software_file_path:
            software_dir = os.path.dirname(software_file_path)
            software_base = os.path.splitext(os.path.basename(software_file_path))[0]
            png_path = os.path.join(software_dir, f"{software_base}_correlation.png")
        else:
            png_path = "correlation_plot.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"📊 Saved correlation plot to: {png_path}")
    
    if show_plot:
        plt.show()
    
    if not show_plot:
        plt.close()  # Clean up the figure to save memory when not showing
    
    return correlation, r_squared, len(clean_df)

def main(min_manual_count=5, automated_mode=False, software_file=None, manual_file=None, param_info=None, save_plot_path=None):
    """
    Main function to run the correlation analysis
    
    Args:
        min_manual_count (int): Minimum manual count threshold for filtering
        automated_mode (bool): If True, uses cached file paths without GUI prompts
                              and suppresses plot display. If False, shows interactive 
                              file selection dialogs and displays plots.
        software_file (str): Path to software analysis Excel file (overrides cache)
        manual_file (str): Path to manual count Excel file (if None, uses cache)
        param_info (dict): Parameter information for plot titles and saved stats
        save_plot_path (str): Path to save correlation plot (if provided)
    """
    print("Worm Count Correlation Analysis")
    print("=" * 40)
    print(f"Minimum manual count threshold: {min_manual_count}")
    
    # Use file cache system to get file paths
    print("Step 1: Getting file paths (checking cache)...")
    
    # Determine file paths based on provided arguments and mode
    file_paths = {}
    
    # Handle software file
    if software_file and os.path.exists(software_file):
        file_paths['software_file'] = software_file
        print(f"✅ Using provided software file: {os.path.basename(software_file)}")
    elif automated_mode:
        print("❌ No valid software file provided for automated mode!")
        return
    else:
        # Interactive mode - prompt for software file
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        software_path = filedialog.askopenfilename(
            title="Select Software Analysis Excel File",
            filetypes=[("Excel files", "*.xlsx")]
        )
        root.destroy()
        
        if not software_path:
            print("❌ No software file selected")
            return
        file_paths['software_file'] = software_path
    
    # Handle manual file
    if manual_file and os.path.exists(manual_file):
        file_paths['manual_file'] = manual_file
        print(f"✅ Using provided manual file: {os.path.basename(manual_file)}")
    else:
        # Try to get manual file from cache first
        cached_paths = file_cache.load_file_paths(__file__)
        cached_manual = cached_paths.get('manual_file', '') if cached_paths else ''
        
        if cached_manual and os.path.exists(cached_manual):
            file_paths['manual_file'] = cached_manual
            print(f"✅ Using cached manual file: {os.path.basename(cached_manual)}")
        elif automated_mode:
            print("❌ No valid manual file found in cache for automated mode!")
            print("   Please run the script manually once to select and cache the manual file.")
            return
        else:
            # Interactive mode - prompt for manual file
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            
            manual_path = filedialog.askopenfilename(
                title="Select Manual Count Excel File", 
                filetypes=[("Excel files", "*.xlsx")]
            )
            root.destroy()
            
            if not manual_path:
                print("❌ No manual file selected")
                return
            file_paths['manual_file'] = manual_path
            
            # Cache the selected files for future use
            file_cache.save_file_paths(__file__, file_paths)
    
    if not file_paths:
        print("No files selected. Exiting.")
        return
    
    software_file = file_paths['software_file']
    manual_file = file_paths['manual_file']
    
    # Load software data
    print(f"Loading software data from: {software_file}")
    software_df = load_software_data(software_file)
    
    if software_df is None:
        return
    
    print(f"Loaded {len(software_df)} software analysis entries")
    
    # Load manual data
    print(f"Loading manual data from: {manual_file}")
    manual_df = load_manual_data(manual_file)
    
    if manual_df is None:
        return
    
    print(f"Loaded {len(manual_df)} manual count entries")
    
    # Match the data
    print("\nStep 2: Matching software and manual data...")
    matched_df = match_data(software_df, manual_df)
    
    # Apply minimum count filter
    if len(matched_df) > 0:
        print(f"\nStep 2.5: Applying minimum manual count filter (>= {min_manual_count})...")
        original_count = len(matched_df)
        
        # Identify entries to be filtered out
        filtered_out = matched_df[matched_df['manual_count'] < min_manual_count]
        
        if len(filtered_out) > 0:
            print(f"Filtering out {len(filtered_out)} entries with manual count < {min_manual_count}:")
            
            # Group filtered entries by strain for better reporting
            filtered_by_strain = {}
            for _, row in filtered_out.iterrows():
                strain = row['strain']
                if strain not in filtered_by_strain:
                    filtered_by_strain[strain] = []
                filtered_by_strain[strain].append(row)
            
            for strain, entries in filtered_by_strain.items():
                print(f"  {strain}:")
                for entry in entries:
                    print(f"    {entry['day']} {entry['replicate']} → Manual: {entry['manual_count']}, Software: {entry['software_count']}")
        
        # Apply the filter
        matched_df = matched_df[matched_df['manual_count'] >= min_manual_count]
        
        print(f"Retained {len(matched_df)} entries after filtering (was {original_count})")
        
        if len(matched_df) == 0:
            print("No entries remaining after filtering. Consider lowering the minimum count threshold.")
    else:
        print("No matched data to filter.")
    
    # Print matching report
    print("\nMatching Report:")
    print("=" * 20)
    print(f"Total manual entries: {len(manual_df)}")
    print(f"Successfully matched: {len(matched_df)}")
    print(f"Match rate: {len(matched_df)/len(manual_df)*100:.1f}%")
    
    if len(matched_df) > 0:
        print(f"\nMatched entries by strain:")
        strain_counts = matched_df.groupby('strain').size()
        for strain, count in strain_counts.items():
            print(f"  {strain}: {count} matches")
    
    # Plot correlation
    if len(matched_df) > 0:
        if automated_mode:
            print("\nStep 3: Analyzing correlation and saving plot...")
        else:
            print("\nStep 3: Plotting correlation...")
        show_plot = not automated_mode  # Show plot only in interactive mode
        correlation, r_squared, n_points = plot_correlation(
            matched_df, 
            show_plot=show_plot, 
            save_path=save_plot_path,
            param_info=param_info,
            software_file_path=software_file
        )
        
        print(f"\nCorrelation Results:")
        print(f"  Data points: {n_points}")
        print(f"  R-squared: {r_squared:.3f}")
        print(f"  Correlation: {correlation:.3f}")
        
        # Save matched data to CSV for further analysis
        manual_dir = os.path.dirname(manual_file)
        output_file = os.path.join(manual_dir, "matched_worm_counts.csv")
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
            print(f"{'Rank':<6} {'Strain':<12} {'Day':<8} {'Rep':<6} {'Manual':<8} {'Software':<10} {'Error':<8} {'Filename'}")
            print("-" * 80)
            
            for rank, (idx, row) in enumerate(top_errors.iterrows(), start=1):
                error = row['software_count'] - row['manual_count']
                filename = os.path.basename(row['filename']) if 'filename' in row else 'N/A'
                print(f"{rank:<6} {row['strain']:<12} {row['day']:<8} {row['replicate']:<6} "
                      f"{int(row['manual_count']):<8} {int(row['software_count']):<10} "
                      f"{error:+.1f}{'':<4} {filename}")
        else:
            print("No data available for error analysis.")
    
    else:
        print("No matched data found. Cannot create correlation plot.")

if __name__ == "__main__":
    # Set minimum manual count threshold (default: 5)
    # Change this value to adjust the filtering threshold
    MIN_MANUAL_COUNT_THRESHOLD = 0
    
    # When run directly, use interactive mode (GUI prompts enabled)
    main(min_manual_count=MIN_MANUAL_COUNT_THRESHOLD, automated_mode=False) 