#!/usr/bin/env python3
"""
OptimizePositionOnSinglePlate.py

This script systematically tests different parameter combinations for worm position analysis
to find optimal settings. It varies 'pix conf thresh' and 'overlap thresh' parameters,
runs the analysis for each combination, preserves results, and evaluates performance
using worm count correlation analysis.

USAGE:
    python OptimizePositionOnSinglePlate.py

PREREQUISITES:
    1. AnalyzePositionOnSinglePlate.json configuration file must exist
    2. MultiWellAnalysis.py module must be available
    3. Scripts/worm_count_correlation.py must be available
    4. Manual count data file must be accessible and cached
       (run worm_count_correlation.py manually ONCE to select and cache the manual file,
        software files are automatically used from each analysis run)
    5. Ensure Excel is closed to avoid permission errors during automated runs

PARAMETERS TESTED:
    - pix conf thresh: [0.1, 0.3, 0.5]  
    - overlap thresh: [0.2, 0.4, 0.6]
    Total: 9 combinations

PROCESS FOR EACH COMBINATION:
    1. Creates temporary config with new parameters
    2. Runs MultiWellAnalysis with automated mode (may overwrite existing _Analysis)
    3. Preserves successful results to _Analysis_pcf{X}_ot{Y} folder  
    4. Runs correlation analysis using current results vs cached manual data
    5. Saves correlation plots and parameter stats to _Optimization folder
    6. Saves correlation data and overall results
    7. Cleans up temporary files

OUTPUT:
    - optimization_results_YYYYMMDD_HHMMSS.csv: Summary of all results
    - _Analysis_pcf{X}_ot{Y}/: Preserved analysis folders for each successful combination
    - _Optimization/correlation_pcf{X}_ot{Y}.png: Correlation plots with parameters
    - _Optimization/stats_pcf{X}_ot{Y}.txt: Detailed parameter statistics
    - Scripts/matched_worm_counts_pcf{X}_ot{Y}.csv: Correlation data per combination
    
NOTE: The script preserves results AFTER each run, so you know exactly which 
parameters generated each preserved folder. Correlation analysis uses the fresh 
results from each run (not cached software data).
    
The script will automatically identify the best parameter combination based on R-squared values.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import copy

# Check for required dependencies
try:
    import MultiWellAnalysis
except ImportError:
    print("❌ Error: MultiWellAnalysis module not found")
    print("Make sure you're running this script from the correct directory")
    sys.exit(1)

try:
    from scipy.stats import pearsonr
    from sklearn.metrics import r2_score
except ImportError:
    print("❌ Error: scipy and sklearn are required for correlation analysis")
    print("Please install them with: pip install scipy scikit-learn")
    sys.exit(1)

# Parameter combinations to test
PIX_CONF_THRESH_VALUES = [0.1, 0.3, 0.5]
OVERLAP_THRESH_VALUES = [0.2, 0.4, 0.6]

# Base configuration file
BASE_CONFIG_FILE = "AnalyzePositionOnSinglePlate.json"
TEMP_CONFIG_FILE = "temp_AnalyzePositionOnSinglePlate.json"

def load_base_config():
    """Load the base configuration from JSON file"""
    try:
        with open(BASE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded base configuration from {BASE_CONFIG_FILE}")
        return config
    except FileNotFoundError:
        print(f"❌ Error: Base configuration file {BASE_CONFIG_FILE} not found")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in {BASE_CONFIG_FILE}: {e}")
        sys.exit(1)

def save_temp_config(config, pix_conf_thresh, overlap_thresh):
    """Save temporary configuration with modified parameters"""
    # Create a deep copy of the config to avoid modifying the original
    temp_config = copy.deepcopy(config)
    
    # Modify the parameters
    temp_config["worm"]["pix conf thresh"] = pix_conf_thresh
    temp_config["worm"]["overlap thresh"] = overlap_thresh
    
    # Save to temporary file
    with open(TEMP_CONFIG_FILE, 'w') as f:
        json.dump(temp_config, f, indent=4)
    
    print(f"📝 Created temporary config with pix_conf_thresh={pix_conf_thresh}, overlap_thresh={overlap_thresh}")
    return TEMP_CONFIG_FILE

def preserve_analysis_folder_after_run(pix_conf_thresh, overlap_thresh, base_config):
    """Copy _Analysis folder after successful run to preserve results with known parameters"""
    # Get the correct path for _Analysis folder from config
    data_path = base_config["plate"]["paths"]["dark field"]
    analysis_folder = os.path.join(data_path, "_Analysis")
    
    if os.path.exists(analysis_folder):
        # Create descriptive folder name with the parameters that were just used
        preserved_name = os.path.join(data_path, f"_Analysis_pcf{pix_conf_thresh}_ot{overlap_thresh}")
        
        # If the preserved folder already exists, add timestamp
        if os.path.exists(preserved_name):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            preserved_name = os.path.join(data_path, f"_Analysis_pcf{pix_conf_thresh}_ot{overlap_thresh}_{timestamp}")
        
        try:
            shutil.copytree(analysis_folder, preserved_name)
            preserved_folder_name = os.path.basename(preserved_name)
            print(f"📁 Preserved analysis results (pcf={pix_conf_thresh}, ot={overlap_thresh}) to: {preserved_folder_name}")
            return preserved_name
        except Exception as e:
            print(f"⚠️  Warning: Could not preserve analysis folder from {analysis_folder}: {e}")
            return None
    else:
        print(f"⚠️  No _Analysis folder found to preserve at: {data_path}")
        return None

def run_analysis(config_file):
    """Run the multi-well analysis with the given configuration"""
    print(f"🔄 Running analysis with configuration: {config_file}")
    print("⚡ Note: This will overwrite any existing _Analysis folder")
    
    try:
        # Set up analysis for automated optimization (bypass all GUI popups)
        print("📵 Running in automated mode (no GUI popups)")
        MWA = MultiWellAnalysis.MultiWellAnalysis(config_file, skip_menus=True)
        
        MWA.analyzeImageSeries()
        print("✅ Analysis completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def run_correlation_analysis(pix_conf_thresh, overlap_thresh, base_config):
    """Run the worm count correlation analysis and return results"""
    print("📊 Running correlation analysis...")
    
    try:
        # Import and run the correlation analysis from Scripts folder
        sys.path.append(os.path.join("Scripts"))
        import worm_count_correlation
        
        # Get the current analysis Excel file (software file)
        data_path = base_config["plate"]["paths"]["dark field"]
        analysis_folder = os.path.join(data_path, "_Analysis")
        
        # Find the Excel file in the _Analysis folder
        excel_files = [f for f in os.listdir(analysis_folder) if f.endswith('.xlsx') and 'raw' in f]
        if not excel_files:
            print("❌ No Excel file found in _Analysis folder")
            return {'n_points': 0, 'correlation': None, 'r_squared': None, 'p_value': None, 'error': 'No Excel file found'}
        
        software_file = os.path.join(analysis_folder, excel_files[0])
        print(f"📊 Using software file: {os.path.basename(software_file)}")
        
        # Create _Optimization folder in the data directory
        optimization_folder = os.path.join(data_path, "_Optimization")
        os.makedirs(optimization_folder, exist_ok=True)
        
        # Set up plot save path
        plot_filename = f"correlation_pcf{pix_conf_thresh}_ot{overlap_thresh}.png"
        save_plot_path = os.path.join(optimization_folder, plot_filename)
        
        # Parameter info for plot titles and stats
        param_info = {
            'pix_conf_thresh': pix_conf_thresh,
            'overlap_thresh': overlap_thresh
        }
        
        # Run correlation analysis with current software file
        worm_count_correlation.main(
            min_manual_count=5, 
            automated_mode=True,
            software_file=software_file,
            param_info=param_info,
            save_plot_path=save_plot_path
        )
        
        # Check if results file was created
        results_file = os.path.join("Scripts", "matched_worm_counts.csv")
        if os.path.exists(results_file):
            # Read the results to get some basic stats
            df = pd.read_csv(results_file)
            if len(df) > 0:
                # Calculate basic correlation stats
                software_counts = df['software_count'].values
                manual_counts = df['manual_count'].values
                
                if len(software_counts) > 1:
                    correlation, p_value = pearsonr(software_counts, manual_counts)
                    r_squared = r2_score(manual_counts, software_counts)
                    
                    correlation_results = {
                        'n_points': len(df),
                        'correlation': correlation,
                        'r_squared': r_squared,
                        'p_value': p_value
                    }
                    print(f"📈 Correlation results: R²={r_squared:.3f}, r={correlation:.3f}, n={len(df)}")
                    
                    # Save parameter statistics to _Optimization folder
                    stats_filename = f"stats_pcf{pix_conf_thresh}_ot{overlap_thresh}.txt"
                    stats_file_path = os.path.join(optimization_folder, stats_filename)
                    
                    with open(stats_file_path, 'w') as f:
                        f.write(f"Parameter Optimization Results\n")
                        f.write(f"=" * 40 + "\n")
                        f.write(f"Parameters:\n")
                        f.write(f"  pix_conf_thresh: {pix_conf_thresh}\n")
                        f.write(f"  overlap_thresh: {overlap_thresh}\n")
                        f.write(f"\nCorrelation Results:\n")
                        f.write(f"  R-squared: {r_squared:.6f}\n")
                        f.write(f"  Correlation: {correlation:.6f}\n")
                        f.write(f"  P-value: {p_value:.6f}\n")
                        f.write(f"  Data points: {len(df)}\n")
                        f.write(f"\nFiles Used:\n")
                        f.write(f"  Software file: {os.path.basename(software_file)}\n")
                        f.write(f"  Analysis timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    print(f"📋 Saved parameter stats to: {stats_filename}")
                    
                    # Preserve the correlation results file with parameter info
                    preserved_correlation_file = f"Scripts/matched_worm_counts_pcf{pix_conf_thresh}_ot{overlap_thresh}.csv"
                    shutil.copy2(results_file, preserved_correlation_file)
                    print(f"📁 Saved correlation data to: {preserved_correlation_file}")
                else:
                    correlation_results = {'n_points': len(df), 'correlation': None, 'r_squared': None, 'p_value': None}
            else:
                correlation_results = {'n_points': 0, 'correlation': None, 'r_squared': None, 'p_value': None}
        else:
            print("⚠️  Warning: Correlation analysis did not produce expected output file")
            correlation_results = {'n_points': 0, 'correlation': None, 'r_squared': None, 'p_value': None}
        
        return correlation_results
        
    except Exception as e:
        print(f"❌ Error during correlation analysis: {e}")
        return {'n_points': 0, 'correlation': None, 'r_squared': None, 'p_value': None, 'error': str(e)}

def cleanup_temp_files():
    """Clean up temporary files"""
    if os.path.exists(TEMP_CONFIG_FILE):
        os.remove(TEMP_CONFIG_FILE)
        print(f"🧹 Cleaned up temporary config file: {TEMP_CONFIG_FILE}")

def main():
    """Main optimization function"""
    print("🔍 Parameter Optimization for Position Analysis")
    print("=" * 60)
    print(f"Testing pix conf thresh values: {PIX_CONF_THRESH_VALUES}")
    print(f"Testing overlap thresh values: {OVERLAP_THRESH_VALUES}")
    print(f"Total combinations: {len(PIX_CONF_THRESH_VALUES) * len(OVERLAP_THRESH_VALUES)}")
    print()
    print("⚠️  WARNING: Each analysis will overwrite the _Analysis folder")
    print("   Results are preserved AFTER each successful run with parameter info")
    print("   Correlation plots and stats saved to _Optimization folder")
    print()
    
    # Load base configuration
    base_config = load_base_config()
    
    # Initialize results tracking
    optimization_results = []
    
    try:
        combination_count = 0
        total_combinations = len(PIX_CONF_THRESH_VALUES) * len(OVERLAP_THRESH_VALUES)
        
        for pix_conf_thresh in PIX_CONF_THRESH_VALUES:
            for overlap_thresh in OVERLAP_THRESH_VALUES:
                combination_count += 1
                print(f"\n🧪 Testing combination {combination_count}/{total_combinations}")
                print(f"   pix conf thresh: {pix_conf_thresh}")
                print(f"   overlap thresh: {overlap_thresh}")
                print("-" * 40)
                
                # Step 1: Create temporary config with new parameters  
                temp_config_file = save_temp_config(base_config, pix_conf_thresh, overlap_thresh)
                
                # Step 2: Run analysis (may overwrite existing _Analysis folder)
                analysis_success = run_analysis(temp_config_file)
                
                if analysis_success:
                    # Step 3: Preserve the results we just generated
                    preserved_folder = preserve_analysis_folder_after_run(pix_conf_thresh, overlap_thresh, base_config)
                    
                    # Step 4: Evaluate results with correlation analysis
                    correlation_results = run_correlation_analysis(pix_conf_thresh, overlap_thresh, base_config)
                    
                    # Step 5: Store results
                    result_entry = {
                        'pix_conf_thresh': pix_conf_thresh,
                        'overlap_thresh': overlap_thresh,
                        'preserved_folder': preserved_folder,
                        'analysis_success': True,
                        **correlation_results
                    }
                    
                    print(f"✅ Combination completed successfully")
                    if correlation_results.get('r_squared') is not None:
                        print(f"   R² = {correlation_results['r_squared']:.3f}")
                else:
                    # Analysis failed
                    result_entry = {
                        'pix_conf_thresh': pix_conf_thresh,
                        'overlap_thresh': overlap_thresh,
                        'preserved_folder': None,
                        'analysis_success': False,
                        'n_points': 0,
                        'correlation': None,
                        'r_squared': None,
                        'p_value': None
                    }
                    print(f"❌ Analysis failed for this combination")
                
                optimization_results.append(result_entry)
                
                # Clean up temporary config file
                cleanup_temp_files()
                
                print(f"🏁 Completed {combination_count}/{total_combinations} combinations")
        
        # Save final optimization results
        results_df = pd.DataFrame(optimization_results)
        results_file = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(results_file, index=False)
        
        print(f"\n🎯 OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print(f"Results saved to: {results_file}")
        
        # Display summary of best results
        successful_results = results_df[results_df['analysis_success'] == True]
        if len(successful_results) > 0:
            valid_results = successful_results.dropna(subset=['r_squared'])
            if len(valid_results) > 0:
                best_result = valid_results.loc[valid_results['r_squared'].idxmax()]
                print(f"\n🏆 BEST COMBINATION:")
                print(f"   pix conf thresh: {best_result['pix_conf_thresh']}")
                print(f"   overlap thresh: {best_result['overlap_thresh']}")
                print(f"   R-squared: {best_result['r_squared']:.3f}")
                print(f"   Correlation: {best_result['correlation']:.3f}")
                print(f"   Data points: {best_result['n_points']}")
            else:
                print("\n⚠️  No valid correlation results found")
        else:
            print("\n❌ No successful analysis runs")
            
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Optimization interrupted by user")
        print(f"Completed {combination_count}/{total_combinations} combinations")
        if optimization_results:
            # Save partial results
            results_df = pd.DataFrame(optimization_results)
            results_file = f"optimization_results_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            results_df.to_csv(results_file, index=False)
            print(f"Partial results saved to: {results_file}")
    
    finally:
        # Clean up any remaining temporary files
        cleanup_temp_files()

if __name__ == "__main__":
    main() 