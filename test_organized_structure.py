#!/usr/bin/env python3
"""
Test script to verify the organized directory structure works correctly
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the snatac_express package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'snatac_express'))

def test_directory_structure():
    """Test that the organized directory structure is working correctly"""
    print("Testing organized directory structure...")
    
    # Check if Phase 1 results exist
    phase1_dir = "results/phase1_results"
    if not os.path.exists(phase1_dir):
        print(f"❌ Phase 1 results directory not found: {phase1_dir}")
        return False
    
    # Check each gene directory
    gene_dirs = [d for d in os.listdir(phase1_dir) 
                 if os.path.isdir(os.path.join(phase1_dir, d))]
    
    if not gene_dirs:
        print(f"❌ No gene directories found in {phase1_dir}")
        return False
    
    print(f"✅ Found {len(gene_dirs)} gene directories: {gene_dirs}")
    
    # Check organization for each gene
    for gene in gene_dirs:
        gene_path = os.path.join(phase1_dir, gene)
        print(f"\nChecking gene: {gene}")
        
        # Check for organized subdirectories
        expected_subdirs = ['model_results', 'feature_rankings', 'cross_validation', 'trained_models', 'data']
        
        for subdir in expected_subdirs:
            subdir_path = os.path.join(gene_path, subdir)
            if os.path.exists(subdir_path):
                print(f"  ✅ {subdir}: exists")
                
                # Check if subdirectory has content
                if os.listdir(subdir_path):
                    print(f"     - Contains {len(os.listdir(subdir_path))} items")
                else:
                    print(f"     - Empty")
            else:
                print(f"  ❌ {subdir}: missing")
        
        # Check for model results files
        model_results_dir = os.path.join(gene_path, 'model_results')
        if os.path.exists(model_results_dir):
            result_files = [f for f in os.listdir(model_results_dir) if f.endswith('_results.txt')]
            print(f"  📊 Model results: {len(result_files)} files found")
            for f in result_files:
                print(f"     - {f}")
        
        # Check for feature ranking files
        feature_rankings_dir = os.path.join(gene_path, 'feature_rankings')
        if os.path.exists(feature_rankings_dir):
            ranking_dirs = [d for d in os.listdir(feature_rankings_dir) 
                           if os.path.isdir(os.path.join(feature_rankings_dir, d))]
            print(f"  🎯 Feature rankings: {len(ranking_dirs)} methods found")
            for d in ranking_dirs:
                method_dir = os.path.join(feature_rankings_dir, d)
                importance_files = [f for f in os.listdir(method_dir) if 'importance.csv' in f]
                print(f"     - {d}: {len(importance_files)} importance files")
    
    return True

def test_phase2_aggregation():
    """Test that Phase 2 aggregation can find the organized files"""
    print("\nTesting Phase 2 aggregation...")
    
    # Import the aggregation function
    from snatac_express.scripts.run_phase2 import aggregate_peak_importances
    
    phase1_dir = "results/phase1_results"
    aggregated_dir = "results/aggregated_results"
    
    if not os.path.exists(phase1_dir):
        print(f"❌ Phase 1 results directory not found: {phase1_dir}")
        return False
    
    try:
        # Run aggregation
        aggregated_results = aggregate_peak_importances(
            phase1_dir, aggregated_dir, include_lr=False
        )
        
        if aggregated_results:
            print(f"✅ Aggregation successful! Found results for {len(aggregated_results)} genes")
            
            # Check aggregated results
            for gene, result_df in aggregated_results.items():
                print(f"  📈 {gene}: {len(result_df)} peaks aggregated")
                
                # Check if we have z-scores
                zscore_cols = [col for col in result_df.columns if 'Zscore' in col]
                print(f"     - Methods: {len(zscore_cols)} ({', '.join(zscore_cols)})")
                
                if 'Average_Zscore' in result_df.columns:
                    print(f"     - Average Z-score range: {result_df['Average_Zscore'].min():.3f} to {result_df['Average_Zscore'].max():.3f}")
            
            # Check if aggregated files were created
            if os.path.exists(aggregated_dir):
                gene_dirs = [d for d in os.listdir(aggregated_dir) 
                           if os.path.isdir(os.path.join(aggregated_dir, d))]
                print(f"  💾 Aggregated files saved for {len(gene_dirs)} genes")
                
                # Check selected peaks summary
                summary_file = os.path.join(aggregated_dir, 'selected_peaks_summary.csv')
                if os.path.exists(summary_file):
                    summary_df = pd.read_csv(summary_file)
                    print(f"  📋 Selected peaks summary: {len(summary_df)} genes")
                    if not summary_df.empty:
                        print(f"     - Total peaks selected: {summary_df['n_peaks_selected'].sum()}")
                        print(f"     - Average peaks per gene: {summary_df['n_peaks_selected'].mean():.1f}")
                else:
                    print(f"  ⚠️  Selected peaks summary not found")
            
            return True
        else:
            print("❌ No aggregated results found")
            return False
            
    except Exception as e:
        print(f"❌ Aggregation failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing snATAC-Express Organized Directory Structure")
    print("=" * 60)
    
    # Test 1: Directory structure
    structure_ok = test_directory_structure()
    
    # Test 2: Phase 2 aggregation
    aggregation_ok = test_phase2_aggregation()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Directory Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"Phase 2 Aggregation: {'✅ PASS' if aggregation_ok else '❌ FAIL'}")
    
    if structure_ok and aggregation_ok:
        print("\n🎉 All tests passed! The organized directory structure is working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 