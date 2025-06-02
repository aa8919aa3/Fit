"""
Josephson Junction Current-Phase Relationship Fitting Toolkit - Triple Model System

A comprehensive Python toolkit for fitting Josephson junction current-phase relationships
with three different models (standard, second-order, and third-order) using sophisticated
Lomb-Scargle frequency detection, advanced parameter optimization, and comprehensive
statistical analysis and model selection.

Features:
- Three Josephson junction models with increasing complexity
- Automatic frequency detection and harmonic analysis  
- Intelligent parameter initialization and optimization
- Comprehensive model comparison and selection
- Statistical validation and quality assessment
- Physical interpretation of fitted parameters
- Advanced visualization and reporting tools
"""

__version__ = "2.0.0"
__author__ = "Research Team"
__email__ = "research@example.com"

# Import enhanced classes and functions
from .fitter import (
    JosephsonTripleFitter,
    JosephsonFitResult,
    fit_josephson_model1,
    fit_josephson_model2, 
    fit_josephson_model3,
    fit_all_josephson_models,
    select_best_model,
    compare_all_models
)

from .models import (
    JosephsonModelBase,
    Model1,
    Model2, 
    Model3,
    get_model,
    get_lmfit_model,
    MODEL_INFO
)

from .utils import (
    lombscargle_frequency_detection,
    estimate_initial_parameters,
    analyze_harmonic_content,
    calculate_information_criteria,
    perform_f_test,
    calculate_akaike_weights,
    generate_synthetic_data,
    validate_fit_quality,
    interpret_physical_parameters
)

# Import warning configuration utilities
from .warnings_config import (
    configure_josephson_warnings,
    suppress_lmfit_warnings,
    suppress_warnings,
    get_warning_summary
)

# Configure warnings by default (suppress common numerical warnings)
configure_josephson_warnings(verbose=False)

# Backward compatibility aliases
JosephsonFitter = JosephsonTripleFitter  # For backward compatibility

__all__ = [
    # Main classes
    "JosephsonTripleFitter",
    "JosephsonFitResult", 
    "JosephsonFitter",  # Backward compatibility
    
    # Model classes
    "JosephsonModelBase",
    "Model1",
    "Model2", 
    "Model3",
    
    # Model utility functions
    "get_model",
    "get_lmfit_model",
    "MODEL_INFO",
    
    # Fitting functions
    "fit_josephson_model1",
    "fit_josephson_model2",
    "fit_josephson_model3", 
    "fit_all_josephson_models",
    "select_best_model",
    "compare_all_models",
    
    # Utility functions
    "lombscargle_frequency_detection",
    "estimate_initial_parameters",
    "analyze_harmonic_content", 
    "calculate_information_criteria",
    "perform_f_test",
    "calculate_akaike_weights",
    "generate_synthetic_data",
    "validate_fit_quality",
    "interpret_physical_parameters",
    
    # Warning configuration utilities
    "configure_josephson_warnings",
    "suppress_lmfit_warnings", 
    "suppress_warnings",
    "get_warning_summary"
]
