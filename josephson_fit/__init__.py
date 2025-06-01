"""
Josephson Junction Current-Phase Relationship Fitting Toolkit

A specialized Python toolkit for fitting Josephson junction current-phase relationships
with background polynomial terms using automatic frequency detection via Lomb-Scargle
periodogram and parameter optimization with lmfit.
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

from .fitter import JosephsonFitter
from .models import Model1, Model2, Model3
from .utils import lombscargle_frequency_detection, generate_synthetic_data

__all__ = [
    "JosephsonFitter",
    "Model1", 
    "Model2", 
    "Model3",
    "lombscargle_frequency_detection",
    "generate_synthetic_data"
]
