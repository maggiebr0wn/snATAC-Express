#!/usr/bin/env python3
"""
Script to reorganize existing results into the new organized directory structure
"""

import os
import shutil
import glob
from pathlib import Path

def reorganize_gene_results(gene_path):
    """Reorganize results for a single gene"""
    gene_name = os.path.basename(gene_path)
    print(f"Reorganizing {gene_name}...")
    
    # Create organized subdirectories if they don't exist
    subdirs = ['model_results', 'feature_rankings', 'cross_validation', 'trained_models', 'data']
    for subdir in subdirs:
        os.makedirs(os.path.join(gene_path, subdir), exist_ok=True)
    
    # Move model results files
    model_results_dir = os.path.join(gene_path, 'model_results')
    result_files = glob.glob(os.path.join(gene_path, '*_results.txt'))
    for file in result_files:
        if os.path.isfile(file):
            shutil.move(file, os.path.join(model_results_dir, os.path.basename(file)))
            print(f"  Moved {os.path.basename(file)} to model_results/")
    
    # Move feature ranking directories
    feature_rankings_dir = os.path.join(gene_path, 'feature_rankings')
    ranking_dirs = [
        'rf_ranker', 'rf_permranker', 'rf_dropcolranker',
        'xgb_ranker', 'xgb_permranker', 'xgb_dropcolranker',
        'lgbm_ranker', 'lgbm_permranker', 'lgbm_dropcolranker',
        'lr_permranker', 'lr_dropcolranker'
    ]
    
    for ranking_dir in ranking_dirs:
        old_path = os.path.join(gene_path, ranking_dir)
        new_path = os.path.join(feature_rankings_dir, ranking_dir)
        
        if os.path.exists(old_path) and os.path.isdir(old_path):
            if os.path.exists(new_path):
                # If destination exists, merge contents
                for item in os.listdir(old_path):
                    old_item = os.path.join(old_path, item)
                    new_item = os.path.join(new_path, item)
                    if os.path.isfile(old_item):
                        shutil.move(old_item, new_item)
                    elif os.path.isdir(old_item):
                        if os.path.exists(new_item):
                            shutil.rmtree(new_item)
                        shutil.move(old_item, new_item)
                # Remove old empty directory
                os.rmdir(old_path)
            else:
                # Move entire directory
                shutil.move(old_path, new_path)
            print(f"  Moved {ranking_dir} to feature_rankings/")
    
    # Move cross-validation files
    cv_dir = os.path.join(gene_path, 'cross_validation')
    cv_files = glob.glob(os.path.join(gene_path, 'cross_validations_*'))
    for cv_file in cv_files:
        if os.path.isdir(cv_file):
            cv_name = os.path.basename(cv_file)
            new_cv_path = os.path.join(cv_dir, cv_name)
            if os.path.exists(new_cv_path):
                shutil.rmtree(new_cv_path)
            shutil.move(cv_file, new_cv_path)
            print(f"  Moved {cv_name} to cross_validation/")
    
    # Move trained model files
    trained_models_dir = os.path.join(gene_path, 'trained_models')
    model_files = glob.glob(os.path.join(gene_path, 'trained_model_*.pkl'))
    for model_file in model_files:
        if os.path.isfile(model_file):
            shutil.move(model_file, os.path.join(trained_models_dir, os.path.basename(model_file)))
            print(f"  Moved {os.path.basename(model_file)} to trained_models/")
    
    # Data directory should already be organized
    print(f"  ✅ {gene_name} reorganization complete")

def reorganize_all_results():
    """Reorganize all results in the results directory"""
    print("🔄 Reorganizing existing results into organized directory structure...")
    
    # Check if results directory exists
    results_dir = "results"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    # Check Phase 1 results
    phase1_dir = os.path.join(results_dir, "phase1_results")
    if not os.path.exists(phase1_dir):
        print(f"❌ Phase 1 results directory not found: {phase1_dir}")
        return False
    
    # Get all gene directories
    gene_dirs = [d for d in os.listdir(phase1_dir) 
                 if os.path.isdir(os.path.join(phase1_dir, d))]
    
    if not gene_dirs:
        print(f"❌ No gene directories found in {phase1_dir}")
        return False
    
    print(f"Found {len(gene_dirs)} gene directories: {gene_dirs}")
    
    # Reorganize each gene
    for gene in gene_dirs:
        gene_path = os.path.join(phase1_dir, gene)
        try:
            reorganize_gene_results(gene_path)
        except Exception as e:
            print(f"❌ Error reorganizing {gene}: {str(e)}")
            return False
    
    print("\n✅ All results reorganized successfully!")
    return True

def main():
    """Main function"""
    print("📁 snATAC-Express Results Reorganization")
    print("=" * 50)
    
    success = reorganize_all_results()
    
    if success:
        print("\n🎉 Reorganization completed successfully!")
        print("The results are now organized in the proper directory structure:")
        print("  - model_results/     : Model performance results")
        print("  - feature_rankings/  : Feature importance rankings")
        print("  - cross_validation/  : Cross-validation predictions")
        print("  - trained_models/    : Saved model files")
        print("  - data/             : Input data files")
        print("\nYou can now run Phase 2 aggregation successfully.")
    else:
        print("\n❌ Reorganization failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 