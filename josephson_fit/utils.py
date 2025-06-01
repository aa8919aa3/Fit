"""
Utility functions for Josephson junction fitting.

This module provides helper functions including Lomb-Scargle frequency detection,
synthetic data generation, and statistical analysis tools.
"""

import numpy as np
from scipy import signal
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any


def lombscargle_frequency_detection(phi_ext: np.ndarray, 
                                   current: np.ndarray,
                                   frequency_range: Optional[Tuple[float, float]] = None,
                                   oversampling: int = 10) -> Tuple[float, float]:
    """
    Detect dominant frequency using Lomb-Scargle periodogram.
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External parameter values
    current : np.ndarray  
        Measured current values
    frequency_range : tuple, optional
        (min_freq, max_freq) range to search. If None, auto-determined.
    oversampling : int
        Oversampling factor for frequency grid
        
    Returns:
    --------
    tuple
        (dominant_frequency, power) - frequency with highest power and its value
    """
    # Remove DC component and linear trend
    current_detrended = signal.detrend(current, type='linear')
    
    # Determine frequency range if not provided
    if frequency_range is None:
        phi_range = np.ptp(phi_ext)
        min_freq = 0.1 / phi_range  # At least 0.1 cycles in data range
        max_freq = 5.0 / np.median(np.diff(np.sort(phi_ext)))  # Nyquist-like limit
    else:
        min_freq, max_freq = frequency_range
    
    # Create frequency grid
    n_freq = int(oversampling * len(phi_ext))
    frequencies = np.linspace(min_freq, max_freq, n_freq)
    
    # Compute Lomb-Scargle periodogram
    ls = LombScargle(phi_ext, current_detrended)
    power = ls.power(frequencies)
    
    # Find peak frequency
    peak_idx = np.argmax(power)
    dominant_freq = frequencies[peak_idx]
    peak_power = power[peak_idx]
    
    return dominant_freq, peak_power


def estimate_initial_parameters(phi_ext: np.ndarray, 
                               current: np.ndarray) -> Dict[str, float]:
    """
    Estimate initial parameters for fitting using Lomb-Scargle and data statistics.
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External parameter values
    current : np.ndarray
        Measured current values
        
    Returns:
    --------
    dict
        Dictionary of initial parameter estimates
    """
    # Frequency detection
    freq, _ = lombscargle_frequency_detection(phi_ext, current)
    
    # Basic statistics
    current_mean = np.mean(current)
    current_std = np.std(current)
    current_range = np.ptp(current)
    phi_center = (np.max(phi_ext) + np.min(phi_ext)) / 2
    
    # Initial parameter estimates
    params = {
        'Ic': current_range / 2,  # Half of current range as critical current
        'T': 0.5,  # Moderate transparency
        'f': freq,  # From Lomb-Scargle
        'd': phi_center,  # Center of phi range
        'phi0': 0.0,  # No initial phase offset
        'k': 0.0,  # No initial quadratic background
        'r': 0.0,  # No initial linear background  
        'C': current_mean  # Mean current as offset
    }
    
    return params


def generate_synthetic_data(phi_ext: np.ndarray,
                          model_type: int = 1,
                          params: Optional[Dict[str, float]] = None,
                          noise_level: float = 0.01) -> np.ndarray:
    """
    Generate synthetic Josephson junction data.
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External parameter values
    model_type : int
        Model type (1, 2, or 3)
    params : dict, optional
        Model parameters. If None, uses default values.
    noise_level : float
        Relative noise level (fraction of signal amplitude)
        
    Returns:
    --------
    np.ndarray
        Synthetic current data with noise
    """
    if params is None:
        params = {
            'Ic': 1.0,
            'T': 0.3,
            'f': 1.0,
            'd': 0.0,
            'phi0': 0.0,
            'k': 0.01,
            'r': 0.0,
            'C': 0.0
        }
    
    # Import models
    from .models import MODELS
    
    # Generate clean signal
    model = MODELS[model_type]()
    clean_signal = model.evaluate(phi_ext, params)
    
    # Add noise
    signal_amplitude = np.std(clean_signal)
    noise = np.random.normal(0, noise_level * signal_amplitude, len(phi_ext))
    
    return clean_signal + noise


def calculate_aic(n_params: int, n_data: int, chi_squared: float) -> float:
    """
    Calculate Akaike Information Criterion (AIC).
    
    Parameters:
    -----------
    n_params : int
        Number of model parameters
    n_data : int
        Number of data points
    chi_squared : float
        Chi-squared value from fit
        
    Returns:
    --------
    float
        AIC value
    """
    return chi_squared + 2 * n_params


def calculate_bic(n_params: int, n_data: int, chi_squared: float) -> float:
    """
    Calculate Bayesian Information Criterion (BIC).
    
    Parameters:
    -----------
    n_params : int
        Number of model parameters
    n_data : int
        Number of data points
    chi_squared : float
        Chi-squared value from fit
        
    Returns:
    --------
    float
        BIC value
    """
    return chi_squared + n_params * np.log(n_data)


def plot_periodogram(phi_ext: np.ndarray, 
                    current: np.ndarray,
                    frequency_range: Optional[Tuple[float, float]] = None,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Lomb-Scargle periodogram for frequency analysis.
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External parameter values
    current : np.ndarray
        Measured current values
    frequency_range : tuple, optional
        (min_freq, max_freq) for plotting
    save_path : str, optional
        Path to save the plot
        
    Returns:
    --------
    plt.Figure
        The matplotlib figure object
    """
    # Remove trend
    current_detrended = signal.detrend(current, type='linear')
    
    # Frequency range
    if frequency_range is None:
        phi_range = np.ptp(phi_ext)
        min_freq = 0.05 / phi_range
        max_freq = 10.0 / np.median(np.diff(np.sort(phi_ext)))
    else:
        min_freq, max_freq = frequency_range
    
    # Compute periodogram
    frequencies = np.linspace(min_freq, max_freq, 1000)
    ls = LombScargle(phi_ext, current_detrended)
    power = ls.power(frequencies)
    
    # Find peak
    peak_idx = np.argmax(power)
    peak_freq = frequencies[peak_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frequencies, power, 'b-', linewidth=1.5)
    ax.axvline(peak_freq, color='red', linestyle='--', 
               label=f'Peak: f = {peak_freq:.4f}')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Lomb-Scargle Power')
    ax.set_title('Frequency Analysis via Lomb-Scargle Periodogram')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
