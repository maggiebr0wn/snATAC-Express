#!/usr/bin/env python3
"""
Generate aggregated peak summary from Phase 2 results
This reflects the final models that use only the selected peaks
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def get_phase2_peak_counts(config):
    """Get the number of peaks used in Phase 2 for each gene"""
    phase2_dir = os.path.join(config['output_dir'], 'phase2_results')
    
    if not os.path.exists(phase2_dir):
        print(f"❌ Phase 2 results directory not found: {phase2_dir}")
        return {}
    
    peak_counts = {}
    
    # Get all gene directories in Phase 2
    gene_dirs = [d for d in os.listdir(phase2_dir) 
                 if os.path.isdir(os.path.join(phase2_dir, d))]
    
    for gene in gene_dirs:
        gene_path = os.path.join(phase2_dir, gene)
        model_results_dir = os.path.join(gene_path, 'model_results')
        
        if not os.path.exists(model_results_dir):
            continue
            
        # Get all result files
        result_files = glob.glob(os.path.join(model_results_dir, '*_results.txt'))
        
        if not result_files:
            continue
            
        # Read the first result file to get peak count
        # Phase 2 typically has 2 rows: aggregated peaks and 95% of aggregated peaks
        result_df = pd.read_csv(result_files[0])
        
        if len(result_df) >= 2:
            # Use the first row (aggregated peaks) as the main count
            peak_counts[gene] = int(result_df.iloc[0]['nPeaks'])
        elif len(result_df) == 1:
            # If only one row, use that
            peak_counts[gene] = int(result_df.iloc[0]['nPeaks'])
        
        print(f"📊 {gene}: {peak_counts[gene]} peaks used in Phase 2")
    
    return peak_counts

def get_phase2_performance(config):
    """Get the performance metrics from Phase 2 results"""
    phase2_dir = os.path.join(config['output_dir'], 'phase2_results')
    
    if not os.path.exists(phase2_dir):
        return {}
    
    performance = {}
    
    # Get all gene directories in Phase 2
    gene_dirs = [d for d in os.listdir(phase2_dir) 
                 if os.path.isdir(os.path.join(phase2_dir, d))]
    
    for gene in gene_dirs:
        gene_path = os.path.join(phase2_dir, gene)
        model_results_dir = os.path.join(gene_path, 'model_results')
        
        if not os.path.exists(model_results_dir):
            continue
            
        # Get all result files
        result_files = glob.glob(os.path.join(model_results_dir, '*_results.txt'))
        
        gene_performance = {}
        
        for result_file in result_files:
            result_df = pd.read_csv(result_file)
            
            # Extract method name from filename
            filename = os.path.basename(result_file)
            method = filename.replace(f'{gene}_', '').replace('_results.txt', '')
            
            if len(result_df) >= 1:
                # Use the first row (aggregated peaks)
                r2_score = result_df.iloc[0]['R2']
                n_peaks = result_df.iloc[0]['nPeaks']
                
                gene_performance[method] = {
                    'R2': r2_score,
                    'n_peaks': n_peaks
                }
        
        performance[gene] = gene_performance
    
    return performance

def generate_phase2_aggregated_summary(config):
    """Generate the final aggregated peak summary from Phase 2 results"""
    print("🔄 Generating Phase 2 aggregated peak summary...")
    
    # Get Phase 2 peak counts
    peak_counts = get_phase2_peak_counts(config)
    
    # Get Phase 2 performance
    performance = get_phase2_performance(config)
    
    # Create summary data
    summary_data = []
    
    for gene in peak_counts.keys():
        n_peaks = peak_counts[gene]
        
        # Get performance for this gene
        gene_perf = performance.get(gene, {})
        
        # Calculate average R2 across all methods
        r2_scores = [perf['R2'] for perf in gene_perf.values()]
        avg_r2 = np.mean(r2_scores) if r2_scores else 0.0
        
        # Get best performing method
        best_method = None
        best_r2 = 0.0
        for method, perf in gene_perf.items():
            if perf['R2'] > best_r2:
                best_r2 = perf['R2']
                best_method = method
        
        summary_data.append({
            'gene': gene,
            'n_peaks_phase2': n_peaks,
            'avg_r2_phase2': avg_r2,
            'best_method': best_method,
            'best_r2_phase2': best_r2,
            'n_methods': len(gene_perf)
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to aggregated results directory
    aggregated_dir = os.path.join(config['output_dir'], 'aggregated_results')
    os.makedirs(aggregated_dir, exist_ok=True)
    
    output_file = os.path.join(aggregated_dir, 'phase2_aggregated_summary.csv')
    summary_df.to_csv(output_file, index=False)
    
    print(f"✅ Saved Phase 2 aggregated summary to {output_file}")
    
    # Print summary
    print("\n📊 Phase 2 Aggregated Summary:")
    print("=" * 50)
    for _, row in summary_df.iterrows():
        print(f"Gene: {row['gene']}")
        print(f"  Peaks used in Phase 2: {row['n_peaks_phase2']}")
        print(f"  Average R²: {row['avg_r2_phase2']:.4f}")
        print(f"  Best method: {row['best_method']} (R² = {row['best_r2_phase2']:.4f})")
        print(f"  Methods tested: {row['n_methods']}")
        print()
    
    # Also update the selected_peaks_summary.csv to reflect Phase 2
    selected_peaks_file = os.path.join(aggregated_dir, 'selected_peaks_summary.csv')
    selected_df = summary_df[['gene', 'n_peaks_phase2']].copy()
    selected_df.columns = ['gene', 'n_peaks_selected']
    selected_df.to_csv(selected_peaks_file, index=False)
    
    print(f"✅ Updated selected peaks summary to reflect Phase 2 results")
    
    return summary_df

def main():
    """Main function"""
    print("🎯 Phase 2 Aggregated Peak Summary Generator")
    print("=" * 50)
    
    # Load configuration
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate summary
    summary_df = generate_phase2_aggregated_summary(config)
    
    if not summary_df.empty:
        print("\n🎉 Successfully generated Phase 2 aggregated summary!")
        print(f"Total genes analyzed: {len(summary_df)}")
        print(f"Total peaks used in Phase 2: {summary_df['n_peaks_phase2'].sum()}")
        print(f"Average R² across all genes: {summary_df['avg_r2_phase2'].mean():.4f}")
    else:
        print("\n❌ No Phase 2 results found to summarize")
    
    return not summary_df.empty

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 