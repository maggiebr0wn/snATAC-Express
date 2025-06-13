#!/usr/bin/env python3
"""
Utility functions for snATAC-Express
"""

import os
import numpy as np
import warnings


def resolve_path(filename: str, input_dir: str = None):
    """
    Return `filename` if it is an absolute path otherwise
    join it to the provided `input_dir`.
    
    Args:
        filename: Path to file (absolute or relative)
        input_dir: Base directory for relative paths
        
    Returns:
        Absolute path to the file
        
    Raises:
        ValueError: If input_dir not provided for relative path
        FileNotFoundError: If the resulting path does not exist
    """
    if os.path.isabs(filename):
        path = filename
    else:
        if input_dir is None:
            raise ValueError(f"input_dir must be provided for relative path: {filename}")
        path = os.path.join(input_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def validate_matrix_files(matrix_path, input_dir=None):
    """
    Check if required matrix annotation files exist
    
    Args:
        matrix_path: Path to main matrix file
        input_dir: Base directory for relative paths
        
    Returns:
        Tuple of (rownames_path, colnames_path)
        
    Raises:
        FileNotFoundError: If annotation files are missing
    """
    base_name = os.path.splitext(os.path.basename(matrix_path))[0]
    base_dir = os.path.dirname(matrix_path) if input_dir is None else input_dir
    
    rownames_path = resolve_path(f"{base_name}_rownames.txt", base_dir)
    colnames_path = resolve_path(f"{base_name}_colnames.txt", base_dir)
    
    return rownames_path, colnames_path