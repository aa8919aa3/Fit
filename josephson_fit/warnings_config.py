"""
Warning configuration for Josephson fitting toolkit.

This module provides utilities to manage and suppress common warnings
from numerical libraries during fitting operations.
"""

import warnings
from functools import wraps
from typing import Callable, Any


def suppress_lmfit_warnings(func: Callable) -> Callable:
    """
    Decorator to suppress common lmfit warnings during fitting operations.
    
    This suppresses:
    - Confidence interval convergence warnings
    - Parameter correlation warnings
    - Model keyword argument warnings
    - Other numerical computation warnings
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with warnings.catch_warnings():
            # Suppress lmfit confidence interval warnings
            warnings.filterwarnings("ignore", 
                                  message=".*rel_change.*", 
                                  category=UserWarning,
                                  module="lmfit.confidence")
            warnings.filterwarnings("ignore", 
                                  message=".*prob.*", 
                                  category=UserWarning,
                                  module="lmfit.confidence")
            
            # Suppress parameter correlation warnings
            warnings.filterwarnings("ignore", 
                                  message=".*correlation matrix.*", 
                                  category=UserWarning,
                                  module="lmfit")
            
            # Suppress lmfit model keyword warnings
            warnings.filterwarnings("ignore", 
                                  message=".*keyword argument.*does not match.*", 
                                  category=UserWarning,
                                  module="lmfit.model")
            
            # Suppress uncertainties package warnings
            warnings.filterwarnings("ignore", 
                                  message=".*Using UFloat objects with std_dev==0.*", 
                                  category=UserWarning,
                                  module="uncertainties.core")
            
            # Suppress astropy/lomb-scargle warnings
            warnings.filterwarnings("ignore", 
                                  message=".*invalid value.*", 
                                  category=RuntimeWarning,
                                  module="astropy")
            
            return func(*args, **kwargs)
    return wrapper


def configure_josephson_warnings(verbose: bool = False):
    """
    Configure warning filters for the entire Josephson fitting session.
    
    Parameters:
    -----------
    verbose : bool
        If True, show warnings. If False, suppress common numerical warnings.
    """
    if not verbose:
        # Suppress lmfit confidence interval warnings
        warnings.filterwarnings("ignore", 
                              message=".*rel_change.*", 
                              category=UserWarning,
                              module="lmfit")
        warnings.filterwarnings("ignore", 
                              message=".*prob.*", 
                              category=UserWarning,
                              module="lmfit")
        warnings.filterwarnings("ignore", 
                              message=".*correlation matrix.*", 
                              category=UserWarning,
                              module="lmfit")
        
        # Suppress lmfit model keyword warnings (these are benign)
        warnings.filterwarnings("ignore", 
                              message=".*keyword argument.*does not match.*", 
                              category=UserWarning,
                              module="lmfit.model")
        
        # Suppress uncertainties package warnings about zero std_dev
        warnings.filterwarnings("ignore", 
                              message=".*Using UFloat objects with std_dev==0.*", 
                              category=UserWarning,
                              module="uncertainties.core")
        
        # Suppress numpy warnings for invalid values (common in optimization)
        warnings.filterwarnings("ignore", 
                              message=".*invalid value.*", 
                              category=RuntimeWarning,
                              module="numpy")
        
        # Suppress astropy warnings
        warnings.filterwarnings("ignore", 
                              message=".*invalid value.*", 
                              category=RuntimeWarning,
                              module="astropy")
        
        print("Josephson fitting warnings suppressed. Use verbose=True to show all warnings.")
    else:
        # Reset warning filters to show all warnings
        warnings.resetwarnings()
        print("All warnings enabled.")


def get_warning_summary() -> dict:
    """
    Get a summary of common warnings and their meanings.
    
    Returns:
    --------
    dict
        Dictionary containing warning types and explanations
    """
    return {
        "lmfit_confidence_warnings": {
            "rel_change warning": "Confidence interval calculation didn't converge well. Usually not critical.",
            "prob warning": "Probability calculation issue in confidence intervals. Usually not critical.",
            "solution": "Consider using more data points or simpler models if this occurs frequently."
        },
        "correlation_warnings": {
            "correlation matrix warning": "Parameters may be strongly correlated. Check parameter relationships.",
            "solution": "Consider fixing some parameters or using different initial guesses."
        },
        "numerical_warnings": {
            "invalid value warning": "Numerical computation encountered inf/nan values. Usually handled gracefully.",
            "solution": "Check data quality and parameter bounds if this occurs frequently."
        },
        "general_advice": [
            "Most warnings during optimization are normal and don't affect final results",
            "Critical errors will raise exceptions, not just warnings",
            "Use warnings_config.configure_josephson_warnings(verbose=True) to see all warnings",
            "Check fit quality using statistical metrics rather than warning count"
        ]
    }


# Context manager for temporary warning suppression
class suppress_warnings:
    """
    Context manager to temporarily suppress specific warnings.
    
    Usage:
    ------
    with suppress_warnings():
        result = fit_model(data)
    """
    
    def __init__(self, include_astropy: bool = True, include_numpy: bool = True):
        self.include_astropy = include_astropy
        self.include_numpy = include_numpy
        
    def __enter__(self):
        self.original_filters = warnings.filters[:]
        
        # Suppress lmfit warnings
        warnings.filterwarnings("ignore", 
                              message=".*rel_change.*", 
                              category=UserWarning,
                              module="lmfit")
        warnings.filterwarnings("ignore", 
                              message=".*prob.*", 
                              category=UserWarning,
                              module="lmfit")
        
        if self.include_numpy:
            warnings.filterwarnings("ignore", 
                                  message=".*invalid value.*", 
                                  category=RuntimeWarning,
                                  module="numpy")
        
        if self.include_astropy:
            warnings.filterwarnings("ignore", 
                                  message=".*invalid value.*", 
                                  category=RuntimeWarning,
                                  module="astropy")
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.filters[:] = self.original_filters
