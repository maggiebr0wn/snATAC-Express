#!/usr/bin/env python3
"""
Script to subset BACH2 data from the full snATAC-seq dataset
and create proper example files for the tutorial.
"""

import os
import pandas as pd
import numpy as np
from scipy import sparse, io
import shutil

# Configuration
FULL_DATA_DIR = os.path.expanduser("~/scratch/multiomics/maggiebrown/misc_singlecell/snatac")
EXAMPLE_DATA_DIR = "example_data/input_data"
GENE = "BACH2"
GENEBODY_WINDOW = "chr6:89826528-90396843"  # BACH2 gene body window

def main():
    print(f"🔍 Subsetting {GENE} data from full dataset...")
    print(f"📂 Full dataset: {FULL_DATA_DIR}")
    print(f"📁 Example data: {EXAMPLE_DATA_DIR}")
    
    # Create backup of current example data
    backup_dir = f"{EXAMPLE_DATA_DIR}_backup"
    if os.path.exists(EXAMPLE_DATA_DIR):
        print(f"💾 Creating backup of current example data...")
        shutil.copytree(EXAMPLE_DATA_DIR, backup_dir, dirs_exist_ok=True)
    
    # Load full gene expression data
    print("🧬 Loading full gene expression data...")
    gex_rownames_file = os.path.join(FULL_DATA_DIR, "sparse_gex_matrix_rownames.txt")
    gex_colnames_file = os.path.join(FULL_DATA_DIR, "sparse_gex_matrix_colnames.txt")
    gex_matrix_file = os.path.join(FULL_DATA_DIR, "sparse_gex_matrix.txt")
    
    # Read gene names
    genes = np.genfromtxt(gex_rownames_file, dtype=str)
    print(f"📊 Total genes in dataset: {len(genes)}")
    
    # Find BACH2 index
    bach2_idx = np.where(genes == GENE)[0]
    if len(bach2_idx) == 0:
        print(f"❌ {GENE} not found in gene list!")
        return
    bach2_idx = bach2_idx[0]
    print(f"🎯 {GENE} found at index {bach2_idx}")
    
    # Read cell names
    cell_names = np.genfromtxt(gex_colnames_file, dtype=str, comments="+")
    print(f"📊 Total cells in dataset: {len(cell_names)}")
    
    # Load sparse gene expression matrix
    print("📈 Loading sparse gene expression matrix...")
    gex_sparse = io.mmread(gex_matrix_file)
    print(f"📊 Gene expression matrix shape: {gex_sparse.shape}")
    
    # Convert to CSR format for efficient indexing
    gex_sparse = gex_sparse.tocsr()
    
    # Extract BACH2 expression
    bach2_expression = gex_sparse[bach2_idx, :].toarray().flatten()
    print(f"🎯 {GENE} expression stats:")
    print(f"  - Mean: {bach2_expression.mean():.4f}")
    print(f"  - Std: {bach2_expression.std():.4f}")
    print(f"  - Min: {bach2_expression.min():.4f}")
    print(f"  - Max: {bach2_expression.max():.4f}")
    
    # Load full peak data
    print("🔍 Loading full peak data...")
    peak_rownames_file = os.path.join(FULL_DATA_DIR, "sparse_peak_matrix_rownames.txt")
    peak_colnames_file = os.path.join(FULL_DATA_DIR, "sparse_peak_matrix_colnames.txt")
    peak_matrix_file = os.path.join(FULL_DATA_DIR, "sparse_peak_matrix.txt")
    
    # Read peak coordinates
    peak_coords = np.genfromtxt(peak_rownames_file, dtype=str)
    print(f"📊 Total peaks in dataset: {len(peak_coords)}")
    
    # Parse BACH2 window
    chr_part = GENEBODY_WINDOW.split(":")[0]
    start_part = int(GENEBODY_WINDOW.split(":")[1].split("-")[0])
    end_part = int(GENEBODY_WINDOW.split(":")[1].split("-")[1])
    
    print(f"🎯 BACH2 window: {GENEBODY_WINDOW}")
    print(f"  - Chromosome: {chr_part}")
    print(f"  - Start: {start_part}")
    print(f"  - End: {end_part}")
    
    # Find peaks in BACH2 window
    bach2_peaks = []
    for i, coord in enumerate(peak_coords):
        if coord.startswith(f"{chr_part}:"):
            parts = coord.split(":")[1].split("-")
            if len(parts) == 2:
                peak_start = int(parts[0])
                peak_end = int(parts[1])
                if peak_start >= start_part and peak_end <= end_part:
                    bach2_peaks.append(i)
    
    print(f"🎯 Found {len(bach2_peaks)} peaks in BACH2 window")
    
    if len(bach2_peaks) == 0:
        print("❌ No peaks found in BACH2 window!")
        return
    
    # Load sparse peak matrix
    print("📈 Loading sparse peak matrix...")
    peak_sparse = io.mmread(peak_matrix_file)
    print(f"📊 Peak matrix shape: {peak_sparse.shape}")
    
    # Convert to CSR format for efficient indexing
    peak_sparse = peak_sparse.tocsr()
    
    # Extract BACH2 peaks and BACH2 expression
    bach2_peak_data = peak_sparse[bach2_peaks, :]
    bach2_gex_data = gex_sparse[bach2_idx:bach2_idx+1, :]
    
    print(f"📊 Subset data shapes:")
    print(f"  - BACH2 peaks: {bach2_peak_data.shape}")
    print(f"  - BACH2 expression: {bach2_gex_data.shape}")
    
    # Create example data directory
    os.makedirs(EXAMPLE_DATA_DIR, exist_ok=True)
    
    # Save subset data
    print("💾 Saving subset data...")
    
    # Save BACH2 peak matrix
    peak_output_file = os.path.join(EXAMPLE_DATA_DIR, "sparse_peak_matrix.txt.mtx")
    io.mmwrite(peak_output_file, bach2_peak_data)
    
    # Save BACH2 gene expression matrix
    gex_output_file = os.path.join(EXAMPLE_DATA_DIR, "sparse_gex_matrix.txt.mtx")
    io.mmwrite(gex_output_file, bach2_gex_data)
    
    # Save peak rownames (coordinates)
    peak_rownames_output = os.path.join(EXAMPLE_DATA_DIR, "sparse_peak_matrix_rownames.txt")
    np.savetxt(peak_rownames_output, peak_coords[bach2_peaks], fmt='%s')
    
    # Save gene rownames
    gex_rownames_output = os.path.join(EXAMPLE_DATA_DIR, "sparse_gex_matrix_rownames.txt")
    np.savetxt(gex_rownames_output, [GENE], fmt='%s')
    
    # Save cell names (colnames)
    peak_colnames_output = os.path.join(EXAMPLE_DATA_DIR, "sparse_peak_matrix_colnames.txt")
    gex_colnames_output = os.path.join(EXAMPLE_DATA_DIR, "sparse_gex_matrix_colnames.txt")
    np.savetxt(peak_colnames_output, cell_names, fmt='%s')
    np.savetxt(gex_colnames_output, cell_names, fmt='%s')
    
    # Create gene list file
    genelist_output = os.path.join(EXAMPLE_DATA_DIR, "genelist_genebody.txt")
    with open(genelist_output, 'w') as f:
        f.write("gene\tgenebody_window_100000\n")
        f.write(f"{GENE}\t{GENEBODY_WINDOW}\n")
    
    # Copy group coverages file
    group_coverages_src = os.path.join(FULL_DATA_DIR, "group_coverages.csv")
    group_coverages_dst = os.path.join(EXAMPLE_DATA_DIR, "group_coverages.csv")
    shutil.copy2(group_coverages_src, group_coverages_dst)
    
    print("✅ BACH2 data subsetting completed!")
    print(f"📁 Example data saved to: {EXAMPLE_DATA_DIR}")
    print(f"📊 Final data shapes:")
    print(f"  - Peaks: {len(bach2_peaks)} peaks")
    print(f"  - Genes: 1 gene ({GENE})")
    print(f"  - Cells: {len(cell_names)} cells")
    
    # Verify the data
    print("\n🔍 Verifying subset data...")
    verify_subset_data()

def verify_subset_data():
    """Verify that the subset data is correct"""
    try:
        # Load subset data
        from snatac_express.scripts.data_preprocessing import load_peak_input, load_gex_input
        
        peak_data = load_peak_input('sparse_peak_matrix.txt.mtx', input_dir=EXAMPLE_DATA_DIR)
        gex_data = load_gex_input('sparse_gex_matrix.txt.mtx', input_dir=EXAMPLE_DATA_DIR)
        
        print(f"✅ Verification successful!")
        print(f"  - Peak data shape: {peak_data.shape}")
        print(f"  - Gene expression data shape: {gex_data.shape}")
        print(f"  - Genes in expression data: {list(gex_data.index)}")
        
        if GENE in gex_data.index:
            bach2_expr = gex_data.loc[GENE]
            print(f"  - {GENE} expression range: [{bach2_expr.min():.4f}, {bach2_expr.max():.4f}]")
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    main() 