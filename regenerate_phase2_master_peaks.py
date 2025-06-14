#!/usr/bin/env python3
"""
Regenerate master aggregated peak ranks using only Phase 2 selected peaks
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def regenerate_phase2_master_peaks():
    """Regenerate master aggregated peak ranks from Phase 2 results"""
    print("🔄 Regenerating master aggregated peak ranks from Phase 2 results...")
    
    # Load configuration
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    aggregated_dir = os.path.join(config['output_dir'], 'aggregated_results')
    
    # Load selected peaks summary
    selected_peaks_file = os.path.join(aggregated_dir, 'selected_peaks_summary.csv')
    if not os.path.exists(selected_peaks_file):
        print(f"❌ Selected peaks summary not found: {selected_peaks_file}")
        return False
    
    selected_df = pd.read_csv(selected_peaks_file)
    print(f"📊 Found {len(selected_df)} genes with selected peaks")
    
    # Load aggregated peak importances for each gene
    all_peaks_data = []
    
    for _, row in selected_df.iterrows():
        gene = row['gene']
        n_peaks_selected = row['n_peaks_selected']
        
        # Load gene-specific aggregated importances
        gene_dir = os.path.join(aggregated_dir, gene)
        if not os.path.exists(gene_dir):
            print(f"⚠️  Gene directory not found: {gene_dir}")
            continue
        
        # Find aggregated importances file
        importances_files = glob.glob(os.path.join(gene_dir, 'aggregated_peak_importances_*.csv'))
        if not importances_files:
            print(f"⚠️  No aggregated importances found for {gene}")
            continue
        
        # Use the first file (should be the main one)
        importances_file = importances_files[0]
        importance_df = pd.read_csv(importances_file)
        
        print(f"📈 {gene}: {len(importance_df)} total peaks, {n_peaks_selected} selected for Phase 2")
        
        # Get the top N peaks based on average z-score
        if 'Average_Zscore' in importance_df.columns:
            # Sort by average z-score (descending)
            sorted_df = importance_df.sort_values('Average_Zscore', ascending=False)
            
            # Take only the top N peaks that are used in Phase 2
            top_peaks_df = sorted_df.head(n_peaks_selected).copy()
            
            # Create gene-specific data
            gene_df = top_peaks_df[['Peaks', 'Average_Zscore']].copy()
            gene_df['Gene'] = gene
            gene_df['Rank'] = range(1, len(gene_df) + 1)
            gene_df = gene_df[['Gene', 'Peaks', 'Average_Zscore', 'Rank']]
            gene_df.columns = ['Gene', 'Peak', 'Avg_Zscore', 'Rank']
            
            all_peaks_data.append(gene_df)
            
            print(f"  ✅ Added {len(gene_df)} peaks for {gene}")
        else:
            print(f"  ❌ No Average_Zscore column found for {gene}")
    
    if all_peaks_data:
        # Combine all data
        master_df = pd.concat(all_peaks_data, ignore_index=True)
        
        # Save master file
        output_file = os.path.join(aggregated_dir, 'master_aggregated_peak_ranks.csv')
        master_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Saved Phase 2 master aggregated peak ranks to {output_file}")
        print(f"📊 Summary:")
        print(f"  - Total genes: {master_df['Gene'].nunique()}")
        print(f"  - Total peaks: {len(master_df)}")
        print(f"  - Unique peaks: {master_df['Peak'].nunique()}")
        print(f"  - Average peaks per gene: {len(master_df) / master_df['Gene'].nunique():.1f}")
        
        # Show top peaks for each gene
        print(f"\n🏆 Top peaks by gene:")
        for gene in master_df['Gene'].unique():
            gene_data = master_df[master_df['Gene'] == gene]
            top_peak = gene_data.iloc[0]
            print(f"  {gene}: {top_peak['Peak']} (Z-score: {top_peak['Avg_Zscore']:.3f})")
        
        # Create summary statistics
        summary_stats = {
            'Total_Genes': master_df['Gene'].nunique(),
            'Total_Unique_Peaks': master_df['Peak'].nunique(),
            'Avg_Peaks_Per_Gene': len(master_df) / master_df['Gene'].nunique() if master_df['Gene'].nunique() > 0 else 0
        }
        
        # Add gene-specific stats
        for gene in master_df['Gene'].unique():
            gene_data = master_df[master_df['Gene'] == gene]
            summary_stats[f'{gene}_Total_Peaks'] = len(gene_data)
            summary_stats[f'{gene}_Top_Peak'] = gene_data.iloc[0]['Peak'] if len(gene_data) > 0 else 'N/A'
            summary_stats[f'{gene}_Top_Zscore'] = gene_data.iloc[0]['Avg_Zscore'] if len(gene_data) > 0 else 'N/A'
        
        # Save summary
        summary_file = os.path.join(aggregated_dir, 'phase2_aggregation_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("PHASE 2 AGGREGATED PEAK RANKS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write("This summary reflects only the peaks used in Phase 2 models.\n\n")
            for key, value in summary_stats.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✅ Saved Phase 2 aggregation summary to {summary_file}")
        
        return True
    else:
        print("❌ No peak data to create master file")
        return False

def main():
    """Main function"""
    print("🎯 Phase 2 Master Aggregated Peak Ranks Regenerator")
    print("=" * 50)
    
    success = regenerate_phase2_master_peaks()
    
    if success:
        print("\n🎉 Successfully regenerated Phase 2 master aggregated peak ranks!")
        print("The master file now contains only the peaks used in Phase 2 models.")
    else:
        print("\n❌ Failed to regenerate master aggregated peak ranks.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 