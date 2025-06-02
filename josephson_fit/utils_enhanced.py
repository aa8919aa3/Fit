"""
Enhanced utility functions for Josephson junction fitting.

This module provides comprehensive utility functions for the triple model system including:
- Lomb-Scargle frequency detection with harmonic analysis
- Advanced parameter estimation and optimization
- Statistical analysis and model comparison
- Synthetic data generation with realistic noise models
- Physical interpretation and validation tools
"""

import numpy as np
from scipy import signal, stats
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any, List, Union
from lmfit import Parameters, minimize, fit_report
import warnings


def lombscargle_frequency_detection(phi_ext: np.ndarray, 
                                   current: np.ndarray,
                                   frequency_range: Optional[Tuple[float, float]] = None,
                                   oversampling: int = 10,
                                   detect_harmonics: bool = True) -> Dict[str, Any]:
    """
    Enhanced Lomb-Scargle frequency detection with harmonic analysis.
    
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
    detect_harmonics : bool
        Whether to detect and analyze harmonic frequencies
        
    Returns:
    --------
    dict
        Comprehensive frequency analysis results including:
        - fundamental_frequency: Primary frequency
        - fundamental_power: Power at fundamental frequency
        - harmonics: Dict of harmonic frequencies and powers
        - frequency_grid: Full frequency array
        - power_spectrum: Full power spectrum
        - snr: Signal-to-noise ratio
    """
    # Remove DC component and linear trend
    current_detrended = signal.detrend(current, type='linear')
    
    # Determine frequency range if not provided
    if frequency_range is None:
        phi_range = np.ptp(phi_ext)
        min_freq = 0.05 / phi_range  # Allow for lower frequencies
        max_freq = 10.0 / np.median(np.diff(np.sort(phi_ext)))  # Extended range
    else:
        min_freq, max_freq = frequency_range
    
    # Create frequency grid
    n_freq = int(oversampling * len(phi_ext))
    frequencies = np.linspace(min_freq, max_freq, n_freq)
    
    # Compute Lomb-Scargle periodogram
    ls = LombScargle(phi_ext, current_detrended)
    power = ls.power(frequencies)
    
    # Find fundamental frequency (highest power peak)
    peak_indices = signal.find_peaks(power, height=np.max(power) * 0.1)[0]
    if len(peak_indices) == 0:
        peak_indices = [np.argmax(power)]
    
    fundamental_idx = peak_indices[np.argmax(power[peak_indices])]
    fundamental_freq = frequencies[fundamental_idx]
    fundamental_power = power[fundamental_idx]
    
    # Calculate signal-to-noise ratio
    noise_power = np.median(power)
    snr = fundamental_power / noise_power if noise_power > 0 else np.inf
    
    result = {
        'fundamental_frequency': fundamental_freq,
        'fundamental_power': fundamental_power,
        'frequency_grid': frequencies,
        'power_spectrum': power,
        'snr': snr,
        'harmonics': {}
    }
    
    # Detect harmonics if requested
    if detect_harmonics:
        harmonics = {}
        for n in [2, 3, 4]:  # Check up to 4th harmonic
            harmonic_freq = n * fundamental_freq
            if harmonic_freq <= max_freq:
                # Find power near harmonic frequency
                freq_tolerance = fundamental_freq * 0.1  # 10% tolerance
                near_harmonic = np.abs(frequencies - harmonic_freq) <= freq_tolerance
                if np.any(near_harmonic):
                    harmonic_power = np.max(power[near_harmonic])
                    harmonic_idx = np.where(near_harmonic)[0][np.argmax(power[near_harmonic])]
                    harmonics[n] = {
                        'frequency': frequencies[harmonic_idx],
                        'power': harmonic_power,
                        'relative_power': harmonic_power / fundamental_power
                    }
        
        result['harmonics'] = harmonics
    
    return result


def estimate_initial_parameters(phi_ext: np.ndarray, 
                               current: np.ndarray,
                               frequency_info: Dict[str, Any]) -> Parameters:
    """
    Estimate initial parameters using physical heuristics and frequency analysis.
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External parameter values
    current : np.ndarray
        Measured current values
    frequency_info : dict
        Results from lombscargle_frequency_detection
        
    Returns:
    --------
    Parameters
        lmfit Parameters object with estimated initial values
    """
    params = Parameters()
    
    # Extract basic statistics
    current_range = np.ptp(current)
    current_mean = np.mean(current)
    current_std = np.std(current)
    phi_range = np.ptp(phi_ext)
    
    # Estimate critical current from signal amplitude
    Ic_estimate = current_range / 2.0  # Rough estimate from peak-to-peak
    
    # Estimate transparency from harmonic content
    harmonic_power = frequency_info.get('harmonics', {})
    if len(harmonic_power) > 0:
        # Higher harmonic content suggests higher transparency
        total_harmonic = sum([h['relative_power'] for h in harmonic_power.values()])
        T_estimate = min(0.8, total_harmonic * 2.0)
    else:
        T_estimate = 0.3  # Conservative default
    
    # Use detected frequency
    f_estimate = frequency_info['fundamental_frequency']
    
    # Estimate phase offset from data symmetry
    phi_mid = np.median(phi_ext)
    d_estimate = phi_mid
    
    # Estimate phase offset from zero-crossing
    if len(current) > 10:
        zero_crossings = np.where(np.diff(np.signbit(current - current_mean)))[0]
        if len(zero_crossings) > 0:
            phi0_estimate = 0.0  # Start with zero
        else:
            phi0_estimate = 0.0
    else:
        phi0_estimate = 0.0
    
    # Estimate background parameters
    # Linear trend
    if len(phi_ext) > 2:
        p = np.polyfit(phi_ext - d_estimate, current, 2)
        k_estimate = p[0] if len(p) > 2 else 0.0
        r_estimate = p[1] if len(p) > 1 else 0.0
        C_estimate = p[2] if len(p) > 0 else current_mean
    else:
        k_estimate = 0.0
        r_estimate = 0.0
        C_estimate = current_mean
    
    # Add parameters with physical bounds
    params.add('Ic', value=Ic_estimate, min=1e-12, max=1e-2, 
              help='Critical current (A)')
    params.add('T', value=T_estimate, min=0.0, max=0.99, 
              help='Junction transparency (0-1)')
    params.add('f', value=f_estimate, min=0.01, max=10.0, 
              help='Conversion factor (Φ_ext to phase scaling)')
    params.add('d', value=d_estimate, min=phi_mid - phi_range, max=phi_mid + phi_range, 
              help='Horizontal shift (zero-point offset)')
    params.add('phi0', value=phi0_estimate, min=-np.pi, max=np.pi, 
              help='Intrinsic phase offset (radians)')
    params.add('k', value=k_estimate, min=-1.0, max=1.0, 
              help='Quadratic background coefficient')
    params.add('r', value=r_estimate, min=-10.0, max=10.0, 
              help='Linear background coefficient')
    params.add('C', value=C_estimate, min=current_mean - 5*current_std, 
              max=current_mean + 5*current_std,
              help='Overall current offset')
    
    return params


def analyze_harmonic_content(phi_ext: np.ndarray, 
                           current: np.ndarray, 
                           current_err: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Analyze harmonic content to recommend appropriate model complexity.
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External parameter values
    current : np.ndarray
        Measured current values
    current_err : np.ndarray, optional
        Current measurement uncertainties
        
    Returns:
    --------
    dict
        Harmonic analysis results with model recommendations
    """
    # Get frequency analysis
    freq_info = lombscargle_frequency_detection(phi_ext, current, detect_harmonics=True)
    
    # Extract harmonic powers
    fundamental_power = freq_info['fundamental_power']
    harmonics = freq_info.get('harmonics', {})
    
    # Calculate relative harmonic strengths
    second_harmonic_power = harmonics.get(2, {}).get('power', 0.0)
    third_harmonic_power = harmonics.get(3, {}).get('power', 0.0)
    
    # Normalize to fundamental
    if fundamental_power > 0:
        second_relative = second_harmonic_power / fundamental_power
        third_relative = third_harmonic_power / fundamental_power
    else:
        second_relative = 0.0
        third_relative = 0.0
    
    # Calculate noise threshold
    noise_power = np.median(freq_info['power_spectrum'])
    signal_threshold = 3.0 * noise_power  # 3-sigma threshold
    
    # Determine significance
    significant_fundamental = fundamental_power > signal_threshold
    significant_second = second_harmonic_power > signal_threshold
    significant_third = third_harmonic_power > signal_threshold
    
    # Model recommendation logic
    if not significant_fundamental:
        recommended_model = "None - insufficient signal"
    elif significant_third and third_relative > 0.1:
        recommended_model = "Model 3 (Third-order)"
    elif significant_second and second_relative > 0.1:
        recommended_model = "Model 2 (Second-order)"
    else:
        recommended_model = "Model 1 (Standard)"
    
    return {
        'fundamental_power': fundamental_power,
        'second_harmonic_power': second_harmonic_power,
        'third_harmonic_power': third_harmonic_power,
        'second_relative': second_relative,
        'third_relative': third_relative,
        'significant_fundamental': significant_fundamental,
        'significant_second': significant_second,
        'significant_third': significant_third,
        'noise_threshold': signal_threshold,
        'snr': freq_info['snr'],
        'recommended_model': recommended_model,
        'frequency_info': freq_info
    }


def calculate_information_criteria(result, n_data: int) -> Dict[str, float]:
    """
    Calculate AIC, BIC, and related information criteria.
    
    Parameters:
    -----------
    result : lmfit.MinimizerResult
        Fit result object
    n_data : int
        Number of data points
        
    Returns:
    --------
    dict
        Information criteria values
    """
    # Extract fit statistics
    chi_square = result.chisqr
    n_params = result.nvarys
    
    # Calculate information criteria
    aic = chi_square + 2 * n_params
    bic = chi_square + n_params * np.log(n_data)
    
    # Corrected AIC for small samples
    if n_data / n_params < 40:
        aicc = aic + (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
    else:
        aicc = aic
    
    # Reduced chi-square
    if (n_data - n_params) > 0:
        redchi = chi_square / (n_data - n_params)
    else:
        redchi = np.inf
    
    return {
        'aic': aic,
        'bic': bic,
        'aicc': aicc,
        'redchi': redchi,
        'chi_square': chi_square,
        'n_params': n_params,
        'n_data': n_data
    }


def perform_f_test(result1, result2) -> Dict[str, float]:
    """
    Perform F-test between two nested models.
    
    Parameters:
    -----------
    result1 : lmfit.MinimizerResult
        Result from simpler model
    result2 : lmfit.MinimizerResult
        Result from more complex model
        
    Returns:
    --------
    dict
        F-test results including F-statistic and p-value
    """
    # Ensure result2 is the more complex model
    if result2.nvarys < result1.nvarys:
        result1, result2 = result2, result1
    
    # Calculate F-statistic
    ssr1 = result1.chisqr  # Sum of squared residuals for simpler model
    ssr2 = result2.chisqr  # Sum of squared residuals for complex model
    
    df1 = result2.nvarys - result1.nvarys  # Difference in parameters
    df2 = result2.ndata - result2.nvarys   # Degrees of freedom for complex model
    
    if df1 <= 0 or df2 <= 0 or ssr2 >= ssr1:
        return {
            'f_statistic': 0.0,
            'p_value': 1.0,
            'df1': df1,
            'df2': df2,
            'significant': False
        }
    
    f_stat = ((ssr1 - ssr2) / df1) / (ssr2 / df2)
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    
    return {
        'f_statistic': f_stat,
        'p_value': p_value,
        'df1': df1,
        'df2': df2,
        'significant': p_value < 0.05
    }


def calculate_akaike_weights(aic_values: List[float]) -> np.ndarray:
    """
    Calculate Akaike weights for model comparison.
    
    Parameters:
    -----------
    aic_values : list
        AIC values for different models
        
    Returns:
    --------
    np.ndarray
        Normalized Akaike weights
    """
    aic_array = np.array(aic_values)
    min_aic = np.min(aic_array)
    
    # Calculate relative likelihoods
    delta_aic = aic_array - min_aic
    rel_likelihood = np.exp(-0.5 * delta_aic)
    
    # Normalize to get weights
    weights = rel_likelihood / np.sum(rel_likelihood)
    
    return weights


def generate_synthetic_data(phi_ext: np.ndarray, 
                          model_func, 
                          params: Dict[str, float],
                          noise_level: float = 0.01,
                          noise_type: str = 'gaussian') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic Josephson junction data with realistic noise.
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External parameter values
    model_func : callable
        Model function to evaluate
    params : dict
        Model parameters
    noise_level : float
        Relative noise level (fraction of signal amplitude)
    noise_type : str
        Type of noise ('gaussian', 'poisson', 'mixed')
        
    Returns:
    --------
    tuple
        (current, current_err) - synthetic data with uncertainties
    """
    # Generate clean signal
    clean_signal = model_func(phi_ext, **params)
    signal_amplitude = np.ptp(clean_signal)
    
    # Generate noise based on type
    if noise_type == 'gaussian':
        noise_std = noise_level * signal_amplitude
        noise = np.random.normal(0, noise_std, len(phi_ext))
        current_err = np.full_like(phi_ext, noise_std)
        
    elif noise_type == 'poisson':
        # Convert to counts assuming some baseline count rate
        baseline_counts = 1000
        signal_normalized = (clean_signal - np.min(clean_signal)) / signal_amplitude
        counts = baseline_counts * (1 + signal_normalized)
        noisy_counts = np.random.poisson(counts)
        noise = (noisy_counts - counts) / baseline_counts * signal_amplitude
        current_err = np.sqrt(counts) / baseline_counts * signal_amplitude
        
    elif noise_type == 'mixed':
        # Combination of Gaussian and 1/f noise
        gaussian_noise = np.random.normal(0, noise_level * signal_amplitude * 0.7, len(phi_ext))
        
        # Generate 1/f noise
        freqs = np.fft.fftfreq(len(phi_ext))
        freqs[0] = 1e-10  # Avoid division by zero
        pink_spectrum = 1 / np.abs(freqs)
        pink_spectrum[0] = 0  # Remove DC component
        
        white_noise = np.random.normal(0, 1, len(phi_ext))
        pink_noise_fft = np.fft.fft(white_noise) * np.sqrt(pink_spectrum)
        pink_noise = np.real(np.fft.ifft(pink_noise_fft))
        pink_noise = pink_noise / np.std(pink_noise) * noise_level * signal_amplitude * 0.3
        
        noise = gaussian_noise + pink_noise
        current_err = np.full_like(phi_ext, noise_level * signal_amplitude)
        
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Add noise to signal
    current = clean_signal + noise
    
    return current, current_err


def validate_fit_quality(result, phi_ext: np.ndarray, current: np.ndarray, 
                        current_err: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Comprehensive validation of fit quality.
    
    Parameters:
    -----------
    result : lmfit.MinimizerResult
        Fit result to validate
    phi_ext : np.ndarray
        External parameter values
    current : np.ndarray
        Measured current values
    current_err : np.ndarray, optional
        Current measurement uncertainties
        
    Returns:
    --------
    dict
        Validation results and quality metrics
    """
    # Calculate residuals
    fitted_values = result.best_fit
    residuals = current - fitted_values
    
    if current_err is not None:
        standardized_residuals = residuals / current_err
    else:
        standardized_residuals = residuals / np.std(residuals)
    
    # Statistical tests
    validation = {}
    
    # Normality test (Shapiro-Wilk)
    if len(standardized_residuals) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(standardized_residuals)
        validation['normality_test'] = {
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'normal': shapiro_p > 0.05
        }
    
    # Autocorrelation test (Durbin-Watson)
    if len(residuals) > 2:
        dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
        validation['autocorrelation_test'] = {
            'durbin_watson': dw_stat,
            'no_autocorr': 1.5 < dw_stat < 2.5
        }
    
    # Homoscedasticity (constant variance)
    if len(phi_ext) > 10:
        # Split data into groups and compare variances
        n_groups = min(5, len(phi_ext) // 5)
        groups = np.array_split(np.argsort(phi_ext), n_groups)
        group_vars = [np.var(residuals[group]) for group in groups if len(group) > 1]
        
        if len(group_vars) > 1:
            # Bartlett test for equal variances
            try:
                bartlett_stat, bartlett_p = stats.bartlett(*[residuals[group] for group in groups if len(group) > 1])
                validation['homoscedasticity_test'] = {
                    'statistic': bartlett_stat,
                    'p_value': bartlett_p,
                    'homoscedastic': bartlett_p > 0.05
                }
            except:
                validation['homoscedasticity_test'] = {'error': 'Could not perform test'}
    
    # Overall quality assessment
    quality_score = 0
    max_score = 0
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((current - np.mean(current))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    validation['r_squared'] = r_squared
    
    if r_squared > 0.9:
        quality_score += 3
    elif r_squared > 0.8:
        quality_score += 2
    elif r_squared > 0.7:
        quality_score += 1
    max_score += 3
    
    # Reduced chi-square
    redchi = result.redchi
    validation['reduced_chi_square'] = redchi
    
    if 0.8 <= redchi <= 1.2:
        quality_score += 3
    elif 0.5 <= redchi <= 2.0:
        quality_score += 2
    elif 0.2 <= redchi <= 5.0:
        quality_score += 1
    max_score += 3
    
    # Parameter uncertainties
    param_uncertainties_ok = all(
        result.params[name].stderr is not None and result.params[name].stderr > 0
        for name in result.params if result.params[name].vary
    )
    validation['parameter_uncertainties_available'] = param_uncertainties_ok
    
    if param_uncertainties_ok:
        quality_score += 2
    max_score += 2
    
    # Overall quality rating
    if max_score > 0:
        quality_percentage = (quality_score / max_score) * 100
    else:
        quality_percentage = 0
    
    if quality_percentage >= 80:
        quality_rating = "Excellent"
    elif quality_percentage >= 60:
        quality_rating = "Good"
    elif quality_percentage >= 40:
        quality_rating = "Fair"
    else:
        quality_rating = "Poor"
    
    validation.update({
        'quality_score': quality_score,
        'max_score': max_score,
        'quality_percentage': quality_percentage,
        'quality_rating': quality_rating,
        'residuals': residuals,
        'standardized_residuals': standardized_residuals
    })
    
    return validation


def interpret_physical_parameters(params: Dict[str, float], 
                                model_type: str = "Model1") -> Dict[str, Any]:
    """
    Provide physical interpretation of fitted parameters.
    
    Parameters:
    -----------
    params : dict
        Fitted parameter values
    model_type : str
        Type of model used
        
    Returns:
    --------
    dict
        Physical interpretation of parameters
    """
    interpretation = {}
    
    # Critical current analysis
    Ic = params.get('Ic', 0)
    interpretation['critical_current'] = {
        'value_A': Ic,
        'value_uA': Ic * 1e6,
        'regime': 'nanoampere' if Ic < 1e-9 else 'microampere' if Ic < 1e-6 else 'milliampere'
    }
    
    # Josephson energy (assuming temperature ~mK range)
    h_bar = 1.055e-34  # J⋅s
    e = 1.602e-19      # C
    phi_0 = h_bar / (2 * e)  # Flux quantum
    
    if Ic > 0:
        E_J = (h_bar * Ic) / (2 * e)  # Josephson energy
        interpretation['josephson_energy'] = {
            'value_J': E_J,
            'value_K': E_J / 1.381e-23,  # Convert to Kelvin
            'value_GHz': E_J / (h_bar * 2 * np.pi * 1e9)  # Convert to GHz
        }
    
    # Transparency analysis
    T = params.get('T', 0)
    interpretation['transparency'] = {
        'value': T,
        'regime': 'tunneling' if T < 0.3 else 'intermediate' if T < 0.7 else 'ballistic',
        'description': f"Junction is in the {'tunneling' if T < 0.3 else 'intermediate' if T < 0.7 else 'ballistic'} regime"
    }
    
    # Frequency analysis
    f = params.get('f', 1)
    interpretation['frequency_factor'] = {
        'value': f,
        'description': f"External parameter converts to phase with factor {f:.3f}"
    }
    
    # Background analysis
    k = params.get('k', 0)
    r = params.get('r', 0)
    C = params.get('C', 0)
    
    background_strength = np.sqrt(k**2 + r**2)
    josephson_strength = abs(Ic) if Ic != 0 else 1
    background_ratio = background_strength / josephson_strength
    
    interpretation['background'] = {
        'quadratic_coeff': k,
        'linear_coeff': r,
        'offset': C,
        'relative_strength': background_ratio,
        'dominance': 'background' if background_ratio > 1 else 'josephson'
    }
    
    # Model-specific analysis
    if model_type in ["Model2", "Model3"]:
        interpretation['harmonic_effects'] = {
            'model': model_type,
            'description': f"Higher-order harmonic terms are included ({model_type})"
        }
    
    return interpretation
