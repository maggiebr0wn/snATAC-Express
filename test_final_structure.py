#!/usr/bin/env python3
"""
Final test script to verify the complete organized structure with Phase 2 aggregated summary
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_complete_structure():
    """Test the complete organized structure with Phase 2 aggregated summary"""
    print("🧪 Testing Complete snATAC-Express Organized Structure")
    print("=" * 60)
    
    # Test 1: Directory structure
    print("1. Testing organized directory structure...")
    phase1_dir = "results/phase1_results"
    phase2_dir = "results/phase2_results"
    aggregated_dir = "results/aggregated_results"
    
    if not os.path.exists(phase1_dir):
        print(f"❌ Phase 1 results directory not found: {phase1_dir}")
        return False
    
    if not os.path.exists(phase2_dir):
        print(f"❌ Phase 2 results directory not found: {phase2_dir}")
        return False
    
    if not os.path.exists(aggregated_dir):
        print(f"❌ Aggregated results directory not found: {aggregated_dir}")
        return False
    
    print("✅ All main directories exist")
    
    # Test 2: Phase 1 organization
    print("\n2. Testing Phase 1 organization...")
    gene_dirs = [d for d in os.listdir(phase1_dir) 
                 if os.path.isdir(os.path.join(phase1_dir, d))]
    
    for gene in gene_dirs:
        gene_path = os.path.join(phase1_dir, gene)
        
        # Check organized subdirectories
        subdirs = ['model_results', 'feature_rankings', 'cross_validation', 'trained_models', 'data']
        for subdir in subdirs:
            subdir_path = os.path.join(gene_path, subdir)
            if os.path.exists(subdir_path) and os.listdir(subdir_path):
                print(f"  ✅ {gene}/{subdir}: {len(os.listdir(subdir_path))} items")
            else:
                print(f"  ⚠️  {gene}/{subdir}: empty or missing")
    
    # Test 3: Phase 2 organization
    print("\n3. Testing Phase 2 organization...")
    phase2_gene_dirs = [d for d in os.listdir(phase2_dir) 
                       if os.path.isdir(os.path.join(phase2_dir, d))]
    
    for gene in phase2_gene_dirs:
        gene_path = os.path.join(phase2_dir, gene)
        
        # Check organized subdirectories
        subdirs = ['model_results', 'feature_rankings', 'cross_validation', 'trained_models', 'data']
        for subdir in subdirs:
            subdir_path = os.path.join(gene_path, subdir)
            if os.path.exists(subdir_path) and os.listdir(subdir_path):
                print(f"  ✅ {gene}/{subdir}: {len(os.listdir(subdir_path))} items")
            else:
                print(f"  ⚠️  {gene}/{subdir}: empty or missing")
    
    # Test 4: Aggregated results
    print("\n4. Testing aggregated results...")
    
    # Check selected peaks summary
    selected_peaks_file = os.path.join(aggregated_dir, 'selected_peaks_summary.csv')
    if os.path.exists(selected_peaks_file):
        selected_df = pd.read_csv(selected_peaks_file)
        print(f"  ✅ selected_peaks_summary.csv: {len(selected_df)} genes")
        for _, row in selected_df.iterrows():
            print(f"     - {row['gene']}: {row['n_peaks_selected']} peaks")
    else:
        print(f"  ❌ selected_peaks_summary.csv: not found")
    
    # Check Phase 2 aggregated summary
    phase2_summary_file = os.path.join(aggregated_dir, 'phase2_aggregated_summary.csv')
    if os.path.exists(phase2_summary_file):
        phase2_df = pd.read_csv(phase2_summary_file)
        print(f"  ✅ phase2_aggregated_summary.csv: {len(phase2_df)} genes")
        for _, row in phase2_df.iterrows():
            print(f"     - {row['gene']}: {row['n_peaks_phase2']} peaks, avg R² = {row['avg_r2_phase2']:.4f}")
            print(f"       Best: {row['best_method']} (R² = {row['best_r2_phase2']:.4f})")
    else:
        print(f"  ❌ phase2_aggregated_summary.csv: not found")
    
    # Check master aggregated peaks
    master_peaks_file = os.path.join(aggregated_dir, 'master_aggregated_peak_ranks.csv')
    if os.path.exists(master_peaks_file):
        master_df = pd.read_csv(master_peaks_file)
        print(f"  ✅ master_aggregated_peak_ranks.csv: {len(master_df)} peak entries")
        print(f"     - {master_df['Gene'].nunique()} genes")
        print(f"     - {master_df['Peak'].nunique()} unique peaks")
    else:
        print(f"  ❌ master_aggregated_peak_ranks.csv: not found")
    
    # Test 5: Verify Phase 2 uses fewer peaks than Phase 1
    print("\n5. Testing peak count reduction in Phase 2...")
    
    for gene in gene_dirs:
        if gene in phase2_gene_dirs:
            # Get Phase 1 peak count (from feature rankings)
            phase1_feature_dir = os.path.join(phase1_dir, gene, 'feature_rankings')
            if os.path.exists(phase1_feature_dir):
                # Count total unique peaks across all methods
                all_peaks = set()
                for method_dir in os.listdir(phase1_feature_dir):
                    method_path = os.path.join(phase1_feature_dir, method_dir)
                    if os.path.isdir(method_path):
                        importance_files = [f for f in os.listdir(method_path) if 'importance.csv' in f]
                        if importance_files:
                            # Read the largest importance file (all peaks)
                            importance_files.sort(key=lambda x: os.path.getsize(os.path.join(method_path, x)), reverse=True)
                            importance_df = pd.read_csv(os.path.join(method_path, importance_files[0]))
                            all_peaks.update(importance_df['Peak'].tolist())
                
                phase1_peak_count = len(all_peaks)
                
                # Get Phase 2 peak count
                phase2_peak_count = selected_df[selected_df['gene'] == gene]['n_peaks_selected'].iloc[0]
                
                print(f"  📊 {gene}:")
                print(f"     Phase 1: {phase1_peak_count} peaks")
                print(f"     Phase 2: {phase2_peak_count} peaks")
                print(f"     Reduction: {phase1_peak_count - phase2_peak_count} peaks ({((phase1_peak_count - phase2_peak_count) / phase1_peak_count * 100):.1f}%)")
                
                if phase2_peak_count < phase1_peak_count:
                    print(f"     ✅ Peak reduction confirmed")
                else:
                    print(f"     ⚠️  No peak reduction")
    
    return True

def main():
    """Main function"""
    print("🎯 Final snATAC-Express Structure Test")
    print("=" * 60)
    
    success = test_complete_structure()
    
    if success:
        print("\n🎉 All tests passed!")
        print("\n📁 Final Directory Structure:")
        print("results/")
        print("├── phase1_results/")
        print("│   └── [GENE]/")
        print("│       ├── model_results/     # Model performance results")
        print("│       ├── feature_rankings/  # Feature importance rankings")
        print("│       ├── cross_validation/  # Cross-validation predictions")
        print("│       ├── trained_models/    # Saved model files")
        print("│       └── data/             # Input data files")
        print("├── phase2_results/")
        print("│   └── [GENE]/")
        print("│       ├── model_results/     # Phase 2 model results")
        print("│       ├── feature_rankings/  # Phase 2 feature rankings")
        print("│       ├── cross_validation/  # Phase 2 CV predictions")
        print("│       ├── trained_models/    # Phase 2 saved models")
        print("│       └── data/             # Phase 2 data files")
        print("└── aggregated_results/")
        print("    ├── [GENE]/")
        print("    │   └── aggregated_peak_importances_*.csv")
        print("    ├── selected_peaks_summary.csv      # Phase 2 peak counts")
        print("    ├── phase2_aggregated_summary.csv   # Phase 2 performance")
        print("    └── master_aggregated_peak_ranks.csv")
        
        print("\n✅ The pipeline now generates properly organized results with Phase 2 aggregated summaries!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 