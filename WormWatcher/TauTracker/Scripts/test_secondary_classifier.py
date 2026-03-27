#!/usr/bin/env python3
"""
Test Secondary Classifier Integration

This script demonstrates how to use the secondary classifier in MultiWellAnalysis.

Usage:
    python test_secondary_classifier.py

Author: AI Assistant
Date: 2024
"""

import json
import os

def enable_secondary_classifier(config_file="AnalyzePositionOnSinglePlate.json"):
    """Enable secondary classifier in the configuration file"""
    
    # Load current configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Ensure secondary_classifier section exists
    if "secondary_classifier" not in config:
        config["secondary_classifier"] = {}
    
    # Set secondary classifier parameters
    config["secondary_classifier"]["secondary_classification_mode"] = "filter"  # or "mark_only"
    
    # Update model path to point to your trained classifier
    # config["secondary_classifier"]["secondary_classifier_path"] = "./classifier_data/Classifier/Models/worm_classifier_latest.pth"
    
    # Save updated configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Secondary classifier configuration updated in {config_file}")
    print("   Set secondary_classification_mode: filter")
    print("   Update secondary_classifier_path to point to your trained model")

def disable_secondary_classifier(config_file="AnalyzePositionOnSinglePlate.json"):
    """Remove secondary classifier configuration"""
    
    # Load current configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Remove secondary classifier section
    if "secondary_classifier" in config:
        del config["secondary_classifier"]
    
    # Save updated configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"✅ Secondary classifier configuration removed from {config_file}")

def show_usage_examples():
    """Show usage examples for the secondary classifier"""
    
    print("\n" + "="*60)
    print("🪱 SECONDARY CLASSIFIER USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. ENABLE SECONDARY CLASSIFIER:")
    print("   python test_secondary_classifier.py --enable")
    
    print("\n2. RUN ANALYSIS:")
    print("   python AnalyzePositionOnSinglePlate.py")
    
    print("\n3. APPLY SECONDARY VERIFICATION:")
    print("   # Uncomment the line in AnalyzePositionOnSinglePlate.py:")
    print("   # MWA.secondaryObjectVerification()")
    print("   # Or run directly:")
    print("   from MultiWellAnalysis import MultiWellAnalysis")
    print("   MWA = MultiWellAnalysis('AnalyzePositionOnSinglePlate.json')")
    print("   MWA.secondaryObjectVerification()")
    
    print("\n4. REMOVE SECONDARY CLASSIFIER CONFIG:")
    print("   python test_secondary_classifier.py --disable")
    
    print("\n" + "="*60)
    print("CONFIGURATION OPTIONS:")
    print("="*60)
    print("secondary_classifier_path: path to trained model")
    print("secondary_classifier_window_size: 64 (default)")
    print("secondary_confidence_threshold: 0.5 (default)")
    print("secondary_classification_mode: 'filter' or 'mark_only'")
    
    print("\n" + "="*60)
    print("OUTPUT DATA:")
    print("="*60)
    print("N worms confirmed: Number of worms confirmed by secondary classifier")
    print("N worms rejected: Number of worms rejected by secondary classifier")
    print("Secondary confidence avg: Average confidence of confirmed worms")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--enable":
            enable_secondary_classifier()
        elif sys.argv[1] == "--disable":
            disable_secondary_classifier()
        else:
            print("Usage: python test_secondary_classifier.py [--enable|--disable]")
    else:
        show_usage_examples() 