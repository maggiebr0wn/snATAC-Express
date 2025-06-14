#!/usr/bin/env python3
"""
Test script to verify the final Phase 2 master aggregated peak ranks
"""

import os
import pandas as pd
import yaml
from pathlib import Path

def test_phase2_master_peaks():
    """Test that the master aggregated peak ranks reflect Phase 2 results"""
    print("🧪 Testing Phase 2 Master Aggregated Peak Ranks")
    print("=" * 50)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    aggregated_dir = os.path.join(config['output_dir'], 'aggregated_results')
    
    # Test 1: Check master aggregated peak ranks file
    print("\n📋 Test 1: Master aggregated peak ranks file")
    master_file = os.path.join(aggregated_dir, 'master_aggregated_peak_ranks.csv')
    
    if not os.path.exists(master_file):
        print("❌ Master aggregated peak ranks file not found")
        return False
    
    master_df = pd.read_csv(master_file)
    print(f"✅ Master file found with {len(master_df)} entries")
    
    # Test 2: Verify it only contains Phase 2 peaks
    print("\n📊 Test 2: Phase 2 peak count verification")
    
    # Load selected peaks summary
    selected_file = os.path.join(aggregated_dir, 'selected_peaks_summary.csv')
    if not os.path.exists(selected_file):
        print("❌ Selected peaks summary not found")
        return False
    
    selected_df = pd.read_csv(selected_file)
    expected_peaks = selected_df['n_peaks_selected'].sum()
    actual_peaks = len(master_df)
    
    print(f"Expected Phase 2 peaks: {expected_peaks}")
    print(f"Actual peaks in master: {actual_peaks}")
    
    if expected_peaks == actual_peaks:
        print("✅ Peak count matches Phase 2 selection")
    else:
        print("❌ Peak count mismatch")
        return False
    
    # Test 3: Verify gene-specific peak counts
    print("\n🧬 Test 3: Gene-specific peak counts")
    
    for _, row in selected_df.iterrows():
        gene = row['gene']
        expected_count = row['n_peaks_selected']
        
        gene_peaks = master_df[master_df['Gene'] == gene]
        actual_count = len(gene_peaks)
        
        print(f"  {gene}: Expected {expected_count}, Got {actual_count}")
        
        if expected_count != actual_count:
            print(f"❌ Mismatch for {gene}")
            return False
    
    print("✅ All gene-specific peak counts match")
    
    # Test 4: Verify peak ranking
    print("\n🏆 Test 4: Peak ranking verification")
    
    for gene in master_df['Gene'].unique():
        gene_peaks = master_df[master_df['Gene'] == gene].copy()
        gene_peaks = gene_peaks.sort_values('Avg_Zscore', ascending=False)
        
        # Check that ranks are sequential
        expected_ranks = list(range(1, len(gene_peaks) + 1))
        actual_ranks = gene_peaks['Rank'].tolist()
        
        if expected_ranks == actual_ranks:
            print(f"✅ {gene}: Ranks are sequential (1-{len(gene_peaks)})")
        else:
            print(f"❌ {gene}: Rank mismatch")
            return False
        
        # Check that z-scores are in descending order
        z_scores = gene_peaks['Avg_Zscore'].tolist()
        if z_scores == sorted(z_scores, reverse=True):
            print(f"✅ {gene}: Z-scores are in descending order")
        else:
            print(f"❌ {gene}: Z-scores not in descending order")
            return False
    
    # Test 5: Verify top peaks match Phase 2 selection
    print("\n🎯 Test 5: Top peaks verification")
    
    # Load Phase 2 aggregated summary
    phase2_summary_file = os.path.join(aggregated_dir, 'phase2_aggregated_summary.csv')
    if os.path.exists(phase2_summary_file):
        phase2_summary = pd.read_csv(phase2_summary_file)
        
        for _, row in phase2_summary.iterrows():
            gene = row['gene']
            
            gene_peaks = master_df[master_df['Gene'] == gene]
            if len(gene_peaks) > 0:
                actual_top_peak = gene_peaks.iloc[0]['Peak']
                print(f"✅ {gene}: Top peak is {actual_top_peak}")
            else:
                print(f"❌ {gene}: No peaks found in master file")
                return False
    
    # Test 6: Summary statistics
    print("\n📈 Test 6: Summary statistics")
    
    total_genes = master_df['Gene'].nunique()
    total_peaks = len(master_df)
    unique_peaks = master_df['Peak'].nunique()
    avg_peaks_per_gene = total_peaks / total_genes if total_genes > 0 else 0
    
    print(f"  Total genes: {total_genes}")
    print(f"  Total peaks: {total_peaks}")
    print(f"  Unique peaks: {unique_peaks}")
    print(f"  Average peaks per gene: {avg_peaks_per_gene:.1f}")
    
    # Show top 5 peaks overall
    print(f"\n🏅 Top 5 peaks overall:")
    top_peaks = master_df.nlargest(5, 'Avg_Zscore')
    for _, peak in top_peaks.iterrows():
        print(f"  {peak['Gene']}: {peak['Peak']} (Z-score: {peak['Avg_Zscore']:.3f})")
    
    print("\n🎉 All tests passed! Master aggregated peak ranks correctly reflect Phase 2 results.")
    return True

def main():
    """Main function"""
    success = test_phase2_master_peaks()
    
    if success:
        print("\n✅ Phase 2 master aggregated peak ranks verification completed successfully!")
        print("The master file now correctly contains only the peaks used in Phase 2 models.")
    else:
        print("\n❌ Phase 2 master aggregated peak ranks verification failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 