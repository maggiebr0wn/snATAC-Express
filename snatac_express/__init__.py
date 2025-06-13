"""
snATAC-Express: Predict gene expression from single-nucleus ATAC-seq data using machine learning.

This package provides tools for predicting gene expression from single-nucleus ATAC-seq data
using various machine learning models including Random Forest, XGBoost, LightGBM, and Linear Regression.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Maggie Brown"
__email__ = "maggie.brown@example.com"

from . import scripts
from . import run_snATAC_Express

__all__ = ["scripts", "run_snATAC_Express", "__version__"] 