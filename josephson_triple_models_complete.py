import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from lmfit import Model, Parameters, minimize, report_fit
from scipy.signal import find_peaks
from scipy import stats
from scipy.optimize import differential_evolution
import warnings
from datetime import datetime

def josephson_model1(phi_ext, Ic, T, f, d, phi0, k, r, C):
    """
    Model 1: Standard Josephson junction with transparency effects
    """
    phase = 2 * np.pi * f * (phi_ext - d) - phi0
    
    numerator = Ic * np.sin(phase)
    denominator_arg = 1 - T * (np.sin(phase / 2))**2
    denominator = np.sqrt(np.maximum(1e-12, denominator_arg))
    
    josephson_component = numerator / denominator
    displacement = phi_ext - d
    background = k * displacement**2 + r * displacement + C
    
    return josephson_component + background

def josephson_model2(phi_ext, Ic, T, f, d, phi0, k, r, C):
    """
    Model 2: Second-order Josephson junction with harmonic terms
    """
    phase = 2 * np.pi * f * (phi_ext - d) - phi0
    
    first_order = np.sin(phase)
    second_order = (np.sin(2 * phase)**2) / 2
    numerator = Ic * (first_order + second_order)
    
    denominator_arg = 1 - T * (np.sin(phase / 2))**2
    denominator = np.sqrt(np.maximum(1e-12, denominator_arg))
    
    josephson_component = numerator / denominator
    displacement = phi_ext - d
    background = k * displacement**2 + r * displacement + C
    
    return josephson_component + background

def josephson_model3(phi_ext, Ic, T, f, d, phi0, k, r, C):
    """
    Model 3: Third-order Josephson junction with extended harmonic series
    """
    phase = 2 * np.pi * f * (phi_ext - d) - phi0
    
    first_order = np.sin(phase)
    second_order = (np.sin(2 * phase)**2) / 2
    third_order = (np.sin(3 * phase)**3) / 3
    numerator = Ic * (first_order + second_order + third_order)
    
    denominator_arg = 1 - T * (np.sin(phase / 2))**2
    denominator = np.sqrt(np.maximum(1e-12, denominator_arg))
    
    josephson_component = numerator / denominator
    displacement = phi_ext - d
    background = k * displacement**2 + r * displacement + C
    
    return josephson_component + background

def find_frequency_lombscargle(phi_ext, current, current_err=None, frequency_range=None, plot=True):
    """
    Use Lomb-Scargle periodogram to find the dominant frequency
    """
    if current_err is not None:
        ls = LombScargle(phi_ext, current, dy=current_err)
    else:
        ls = LombScargle(phi_ext, current)
    
    if frequency_range is None:
        dt_median = np.median(np.diff(np.sort(phi_ext)))
        nyquist_freq = 0.5 / dt_median
        fundamental_freq = 1.0 / (np.max(phi_ext) - np.min(phi_ext))
        frequency_range = (fundamental_freq * 0.1, nyquist_freq * 0.5)
    
    frequencies = np.linspace(frequency_range[0], frequency_range[1], 10000)
    power = ls.power(frequencies)
    
    peaks, _ = find_peaks(power, height=0.1, distance=50)
    
    if len(peaks) > 0:
        best_peak_idx = peaks[np.argmax(power[peaks])]
        best_frequency = frequencies[best_peak_idx]
        best_power = power[best_peak_idx]
    else:
        best_peak_idx = np.argmax(power)
        best_frequency = frequencies[best_peak_idx]
        best_power = power[best_peak_idx]
    
    false_alarm_levels = {}
    for fap in [0.1, 0.05, 0.01, 0.001]:
        try:
            level = ls.false_alarm_level(fap)
            false_alarm_levels[fap] = level
        except:
            false_alarm_levels[fap] = None
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(frequencies, power, 'b-', alpha=0.7)
        ax.axvline(best_frequency, color='red', linestyle='--', 
                  label=f'Best freq: {best_frequency:.6f}')
        
        for fap, level in false_alarm_levels.items():
            if level is not None:
                ax.axhline(level, color='orange', linestyle=':', alpha=0.7,
                          label=f'FAP {fap*100}%')
        
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Lomb-Scargle Power')
        ax.set_title('Lomb-Scargle Periodogram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
    
    return {
        'best_frequency': best_frequency,
        'best_power': best_power,
        'frequencies': frequencies,
        'power': power,
        'false_alarm_levels': false_alarm_levels
    }

def setup_josephson_parameters(frequency_estimate):
    """
    Set up lmfit parameters for Josephson models
    """
    params = Parameters()
    
    params.add('Ic', value=1e-6, min=1e-12, max=1e-2)
    params.add('T', value=0.5, min=0.0, max=0.99)
    params.add('f', value=frequency_estimate, min=frequency_estimate*0.5, max=frequency_estimate*1.5)
    params.add('d', value=0.0, min=-10.0, max=10.0)
    params.add('phi0', value=0.0, min=-2*np.pi, max=2*np.pi)
    params.add('k', value=0.0, min=-1.0, max=1.0)
    params.add('r', value=0.0, min=-10.0, max=10.0)
    params.add('C', value=0.0, min=-10.0, max=10.0)
    
    return params

def fit_josephson_model_generic(model_func, model_name, phi_ext, current, current_err=None, 
                               frequency_range=None, plot_ls=True, use_global_optimization=False):
    """
    Generic fitting function for any Josephson model
    """
    print(f"Fitting {model_name}...")
    
    # Find frequency
    ls_result = find_frequency_lombscargle(phi_ext, current, current_err, 
                                         frequency_range, plot=plot_ls)
    
    # Set up parameters
    params = setup_josephson_parameters(ls_result['best_frequency'])
    
    # Create model
    model = Model(model_func)
    
    if use_global_optimization:
        def objective(pars_array):
            param_names = [p for p in params if params[p].vary]
            for i, name in enumerate(param_names):
                params[name].value = pars_array[i]
            
            y_model = model.eval(params, phi_ext=phi_ext)
            if current_err is not None:
                residuals = (current - y_model) / current_err
            else:
                residuals = current - y_model
            return np.sum(residuals**2)
        
        bounds = []
        for name in params:
            if params[name].vary:
                bounds.append((params[name].min, params[name].max))
        
        result_global = differential_evolution(objective, bounds, seed=42, maxiter=1000)
        
        param_names = [p for p in params if params[p].vary]
        for i, name in enumerate(param_names):
            params[name].value = result_global.x[i]
    
    # Perform final fit
    if current_err is not None:
        weights = 1.0 / current_err
        result = model.fit(current, params, phi_ext=phi_ext, weights=weights)
    else:
        result = model.fit(current, params, phi_ext=phi_ext)
    
    result.lombscargle_result = ls_result
    result.model_name = model_name
    
    return result

def fit_josephson_model1_with_lombscargle(phi_ext, current, current_err=None, 
                                        frequency_range=None, plot_ls=True):
    """Fit Model 1 using Lomb-Scargle frequency detection"""
    return fit_josephson_model_generic(josephson_model1, "Model 1 (Standard Josephson)",
                                     phi_ext, current, current_err, frequency_range, plot_ls)

def fit_josephson_model2_with_lombscargle(phi_ext, current, current_err=None, 
                                        frequency_range=None, plot_ls=True,
                                        use_global_optimization=False):
    """Fit Model 2 using Lomb-Scargle frequency detection"""
    return fit_josephson_model_generic(josephson_model2, "Model 2 (Second-order Josephson)",
                                     phi_ext, current, current_err, frequency_range, plot_ls,
                                     use_global_optimization)

def fit_josephson_model3_with_lombscargle(phi_ext, current, current_err=None, 
                                        frequency_range=None, plot_ls=True,
                                        use_global_optimization=True):
    """Fit Model 3 using Lomb-Scargle frequency detection"""
    return fit_josephson_model_generic(josephson_model3, "Model 3 (Third-order Josephson)",
                                     phi_ext, current, current_err, frequency_range, plot_ls,
                                     use_global_optimization)

def fit_all_josephson_models(phi_ext, current, current_err=None, frequency_range=None):
    """
    Fit all three Josephson models
    """
    print("Fitting all Josephson models...")
    print("=" * 40)
    
    result1 = fit_josephson_model1_with_lombscargle(phi_ext, current, current_err, 
                                                   frequency_range, plot_ls=False)
    result2 = fit_josephson_model2_with_lombscargle(phi_ext, current, current_err, 
                                                   frequency_range, plot_ls=False)
    result3 = fit_josephson_model3_with_lombscargle(phi_ext, current, current_err, 
                                                   frequency_range, plot_ls=False)
    
    return [result1, result2, result3]

def analyze_harmonic_content(phi_ext, current, current_err=None):
    """
    Analyze harmonic content to guide model selection
    """
    print("Analyzing harmonic content...")
    
    # First fit Model 1 to get basic parameters
    result1 = fit_josephson_model1_with_lombscargle(phi_ext, current, current_err, plot_ls=False)
    f_fundamental = result1.params['f'].value
    
    # Analyze residuals from Model 1
    residuals = current - result1.best_fit
    
    if current_err is not None:
        ls_residuals = LombScargle(phi_ext, residuals, dy=current_err)
    else:
        ls_residuals = LombScargle(phi_ext, residuals)
    
    # Check power at different harmonics
    freq_fund = np.linspace(f_fundamental * 0.8, f_fundamental * 1.2, 1000)
    freq_2nd = np.linspace(f_fundamental * 1.8, f_fundamental * 2.2, 1000)
    freq_3rd = np.linspace(f_fundamental * 2.8, f_fundamental * 3.2, 1000)
    
    power_fund = ls_residuals.power(freq_fund)
    power_2nd = ls_residuals.power(freq_2nd)
    power_3rd = ls_residuals.power(freq_3rd)
    
    # Get maximum powers
    fundamental_power = np.max(power_fund)
    second_harmonic_power = np.max(power_2nd)
    third_harmonic_power = np.max(power_3rd)
    
    # Significance thresholds (empirical)
    sig_fundamental = fundamental_power > 0.05
    sig_second = second_harmonic_power > 0.03
    sig_third = third_harmonic_power > 0.02
    
    # Recommend model based on harmonic content
    if sig_third:
        recommended_model = "Model 3"
    elif sig_second:
        recommended_model = "Model 2"
    else:
        recommended_model = "Model 1"
    
    return {
        'fundamental_power': fundamental_power,
        'second_harmonic_power': second_harmonic_power,
        'third_harmonic_power': third_harmonic_power,
        'significant_fundamental': sig_fundamental,
        'significant_second': sig_second,
        'significant_third': sig_third,
        'recommended_model': recommended_model,
        'residuals': residuals
    }

def perform_nested_f_tests(result1, result2, result3):
    """
    Perform F-tests for nested model comparisons
    """
    def f_test_pair(simple_result, complex_result):
        dof_simple = len(simple_result.data) - simple_result.nvarys
        dof_complex = len(complex_result.data) - complex_result.nvarys
        
        f_stat = ((simple_result.chisqr - complex_result.chisqr) / (dof_simple - dof_complex)) / (complex_result.chisqr / dof_complex)
        p_value = 1 - stats.f.cdf(f_stat, dof_simple - dof_complex, dof_complex)
        
        return f_stat, p_value
    
    f12, p12 = f_test_pair(result1, result2)  # Model 1 vs 2
    f23, p23 = f_test_pair(result2, result3)  # Model 2 vs 3
    f13, p13 = f_test_pair(result1, result3)  # Model 1 vs 3
    
    return {
        'f12': f12, 'p12': p12,
        'f23': f23, 'p23': p23,
        'f13': f13, 'p13': p13
    }

def calculate_information_criteria_weights(results):
    """
    Calculate Akaike weights for model comparison
    """
    aics = [r.aic for r in results]
    aic_min = min(aics)
    aic_differences = [aic - aic_min for aic in aics]
    
    # Akaike weights
    raw_weights = [np.exp(-0.5 * diff) for diff in aic_differences]
    weight_sum = sum(raw_weights)
    akaike_weights = [w / weight_sum for w in raw_weights]
    
    # Evidence ratios
    evidence_ratios = [np.exp(0.5 * diff) for diff in aic_differences]
    
    return {
        'akaike_weights': akaike_weights,
        'evidence_ratios': evidence_ratios,
        'aic_differences': aic_differences
    }

def plot_triple_model_comparison(phi_ext, current, current_err, result1, result2, result3):
    """
    Create comprehensive comparison plot for all three models
    """
    fig = plt.figure(figsize=(18, 14))
    
    # Create grid layout
    gs = fig.add_gridspec(5, 3, height_ratios=[3, 1, 1, 1, 1], hspace=0.4, wspace=0.3)
    
    # Main data and fits
    ax_main = fig.add_subplot(gs[0, :])
    if current_err is not None:
        ax_main.errorbar(phi_ext, current, yerr=current_err, fmt='o', alpha=0.6, 
                        label='Data', markersize=4, color='black')
    else:
        ax_main.plot(phi_ext, current, 'o', alpha=0.6, label='Data', markersize=4, color='black')
    
    ax_main.plot(phi_ext, result1.best_fit, 'r-', linewidth=2, label='Model 1 (Standard)')
    ax_main.plot(phi_ext, result2.best_fit, 'g-', linewidth=2, label='Model 2 (Second-order)')
    ax_main.plot(phi_ext, result3.best_fit, 'b-', linewidth=2, label='Model 3 (Third-order)')
    
    ax_main.set_xlabel('Φ_ext')
    ax_main.set_ylabel('Current (A)')
    ax_main.legend()
    ax_main.set_title('Triple Model Comparison - Josephson Junction Analysis')
    ax_main.grid(True, alpha=0.3)
    
    # Individual residual plots
    results = [result1, result2, result3]
    colors = ['red', 'green', 'blue']
    model_names = ['Model 1', 'Model 2', 'Model 3']
    
    for i, (result, color, name) in enumerate(zip(results, colors, model_names)):
        ax_res = fig.add_subplot(gs[1, i])
        residuals = current - result.best_fit
        
        if current_err is not None:
            ax_res.errorbar(phi_ext, residuals, yerr=current_err, fmt='o', 
                          alpha=0.6, markersize=3, color=color)
        else:
            ax_res.plot(phi_ext, residuals, 'o', alpha=0.6, markersize=3, color=color)
        
        ax_res.axhline(y=0, color=color, linestyle='-', alpha=0.5)
        ax_res.set_xlabel('Φ_ext')
        ax_res.set_ylabel('Residuals')
        ax_res.set_title(f'{name} (χ²ᵣ = {result.redchi:.3f})')
        ax_res.grid(True, alpha=0.3)
    
    # Information criteria comparison
    ax_ic = fig.add_subplot(gs[2, :])
    criteria = ['AIC', 'BIC', 'Reduced χ²']
    model1_values = [result1.aic, result1.bic, result1.redchi]
    model2_values = [result2.aic, result2.bic, result2.redchi]
    model3_values = [result3.aic, result3.bic, result3.redchi]
    
    x = np.arange(len(criteria))
    width = 0.25
    
    ax_ic.bar(x - width, model1_values, width, label='Model 1', alpha=0.7, color='red')
    ax_ic.bar(x, model2_values, width, label='Model 2', alpha=0.7, color='green')
    ax_ic.bar(x + width, model3_values, width, label='Model 3', alpha=0.7, color='blue')
    
    ax_ic.set_xlabel('Criteria')
    ax_ic.set_ylabel('Value')
    ax_ic.set_title('Model Comparison Criteria')
    ax_ic.set_xticks(x)
    ax_ic.set_xticklabels(criteria)
    ax_ic.legend()
    ax_ic.grid(True, alpha=0.3)
    
    # Akaike weights
    ic_weights = calculate_information_criteria_weights(results)
    ax_weights = fig.add_subplot(gs[3, 0])
    
    models = ['Model 1', 'Model 2', 'Model 3']
    ax_weights.bar(models, ic_weights['akaike_weights'], color=['red', 'green', 'blue'], alpha=0.7)
    ax_weights.set_ylabel('Akaike Weight')
    ax_weights.set_title('Model Weights')
    ax_weights.grid(True, alpha=0.3)
    
    # Parameter comparison for critical current
    ax_params = fig.add_subplot(gs[3, 1])
    Ic_values = [r.params['Ic'].value for r in results]
    Ic_errors = [r.params['Ic'].stderr or 0 for r in results]
    
    ax_params.errorbar(models, Ic_values, yerr=Ic_errors, fmt='o', capsize=5)
    ax_params.set_ylabel('Critical Current (A)')
    ax_params.set_title('Critical Current Comparison')
    ax_params.grid(True, alpha=0.3)
    
    # Transparency comparison
    ax_trans = fig.add_subplot(gs[3, 2])
    T_values = [r.params['T'].value for r in results]
    T_errors = [r.params['T'].stderr or 0 for r in results]
    
    ax_trans.errorbar(models, T_values, yerr=T_errors, fmt='s', capsize=5, color='orange')
    ax_trans.set_ylabel('Transparency T')
    ax_trans.set_title('Junction Transparency')
    ax_trans.grid(True, alpha=0.3)
    
    # Model selection summary
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    # Find best model
    aics = [r.aic for r in results]
    best_idx = np.argmin(aics)
    aic_diffs = [aic - aics[best_idx] for aic in aics]
    
    summary_text = f"""
Model Selection Summary:
Best Model: {model_names[best_idx]} (AIC = {aics[best_idx]:.2f})
AIC Differences: Model 1: {aic_diffs[0]:.2f}, Model 2: {aic_diffs[1]:.2f}, Model 3: {aic_diffs[2]:.2f}
Evidence Strength: {"Strong" if max(aic_diffs) > 4 else "Moderate" if max(aic_diffs) > 2 else "Weak"}
Akaike Weights: Model 1: {ic_weights['akaike_weights'][0]:.3f}, Model 2: {ic_weights['akaike_weights'][1]:.3f}, Model 3: {ic_weights['akaike_weights'][2]:.3f}
"""
    
    ax_summary.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.suptitle('Comprehensive Josephson Junction Model Analysis', fontsize=16, y=0.95)
    
    return fig

def select_best_model(phi_ext, current, current_err=None, 
                     selection_criterion='aic', evidence_threshold=2.0):
    """
    Automatically select the best model based on statistical criteria
    """
    print("Performing automated model selection...")
    print("=" * 45)
    
    # Fit all models
    results = fit_all_josephson_models(phi_ext, current, current_err)
    
    # Calculate selection criteria
    if selection_criterion.lower() == 'aic':
        criteria = [r.aic for r in results]
    elif selection_criterion.lower() == 'bic':
        criteria = [r.bic for r in results]
    else:
        raise ValueError("selection_criterion must be 'aic' or 'bic'")
    
    best_idx = np.argmin(criteria)
    best_result = results[best_idx]
    
    # Calculate evidence strength
    criteria_diff = np.array(criteria) - criteria[best_idx]
    evidence_ratios = np.exp(criteria_diff / 2)
    
    # Determine evidence strength
    if len(criteria_diff[criteria_diff > 7]) > 0:
        evidence_strength = "Very Strong"
    elif len(criteria_diff[criteria_diff > 4]) > 0:
        evidence_strength = "Strong"
    elif len(criteria_diff[criteria_diff > evidence_threshold]) > 0:
        evidence_strength = "Moderate"
    else:
        evidence_strength = "Weak"
    
    best_result.model_name = f"Model {best_idx + 1}"
    best_result.evidence_strength = evidence_strength
    best_result.evidence_ratios = evidence_ratios
    
    print(f"Selected Model {best_idx + 1} with {evidence_strength.lower()} evidence")
    print(f"{selection_criterion.upper()}: {criteria[best_idx]:.2f}")
    
    return best_result

def compare_all_models(phi_ext, current, current_err=None):
    """
    Comprehensive comparison of all three models
    """
    print("Fitting and comparing all Josephson models...")
    
    # Fit all models
    results = fit_all_josephson_models(phi_ext, current, current_err)
    result1, result2, result3 = results
    
    # Calculate information criteria
    aics = [r.aic for r in results]
    bics = [r.bic for r in results]
    
    # Find best models
    best_aic_idx = np.argmin(aics)
    best_bic_idx = np.argmin(bics)
    
    # Calculate weights and ratios
    ic_weights = calculate_information_criteria_weights(results)
    
    # Determine evidence strength
    aic_min = min(aics)
    aic_diffs = [aic - aic_min for aic in aics]
    max_diff = max([diff for diff in aic_diffs if diff > 0]) if any(diff > 0 for diff in aic_diffs) else 0
    
    if max_diff >= 7:
        evidence_strength = "Very Strong"
    elif max_diff >= 4:
        evidence_strength = "Strong"
    elif max_diff >= 2:
        evidence_strength = "Moderate"
    else:
        evidence_strength = "Weak"
    
    comparison = {
        'model1': {'result': result1, 'aic': aics[0], 'bic': bics[0], 'redchi': result1.redchi},
        'model2': {'result': result2, 'aic': aics[1], 'bic': bics[1], 'redchi': result2.redchi},
        'model3': {'result': result3, 'aic': aics[2], 'bic': bics[2], 'redchi': result3.redchi},
        'best_model': f"Model {best_aic_idx + 1}",
        'best_result': results[best_aic_idx],
        'akaike_weights': ic_weights['akaike_weights'],
        'evidence_ratios': ic_weights['evidence_ratios'],
        'evidence_strength': evidence_strength,
        'all_results': results,
        'aic_differences': aic_diffs
    }
    
    return comparison

def generate_triple_model_report(phi_ext, current, current_err, result1, result2, result3, comparison):
    """
    Generate comprehensive analysis report for all three models
    """
    
    report = f"""
Josephson Junction Analysis Report - Triple Model Comparison
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

================================================================================
DATA SUMMARY
================================================================================
- Data points: {len(phi_ext)}
- Φ_ext range: {phi_ext.min():.4f} to {phi_ext.max():.4f}
- Current range: {current.min():.3e} to {current.max():.3e} A
- Mean uncertainty: {np.mean(current_err) if current_err is not None else "N/A"} A

================================================================================
MODEL COMPARISON RESULTS
================================================================================

Model 1 (Standard Josephson):
- Critical Current: {result1.params['Ic'].value:.3e} ± {result1.params['Ic'].stderr:.3e} A
- Junction Transparency: {result1.params['T'].value:.4f} ± {result1.params['T'].stderr:.4f}
- Frequency: {result1.params['f'].value:.6f} ± {result1.params['f'].stderr:.6f}
- Reduced χ²: {result1.redchi:.4f}
- AIC: {result1.aic:.2f}
- BIC: {result1.bic:.2f}
- Akaike Weight: {comparison['akaike_weights'][0]:.3f}

Model 2 (Second-order Josephson):
- Critical Current: {result2.params['Ic'].value:.3e} ± {result2.params['Ic'].stderr:.3e} A
- Junction Transparency: {result2.params['T'].value:.4f} ± {result2.params['T'].stderr:.4f}
- Frequency: {result2.params['f'].value:.6f} ± {result2.params['f'].stderr:.6f}
- Reduced χ²: {result2.redchi:.4f}
- AIC: {result2.aic:.2f}
- BIC: {result2.bic:.2f}
- Akaike Weight: {comparison['akaike_weights'][1]:.3f}

Model 3 (Third-order Josephson):
- Critical Current: {result3.params['Ic'].value:.3e} ± {result3.params['Ic'].stderr:.3e} A
- Junction Transparency: {result3.params['T'].value:.4f} ± {result3.params['T'].stderr:.4f}
- Frequency: {result3.params['f'].value:.6f} ± {result3.params['f'].stderr:.6f}
- Reduced χ²: {result3.redchi:.4f}
- AIC: {result3.aic:.2f}
- BIC: {result3.bic:.2f}
- Akaike Weight: {comparison['akaike_weights'][2]:.3f}

================================================================================
MODEL SELECTION
================================================================================
Best model (AIC): {comparison['best_model']}
Evidence strength: {comparison['evidence_strength']}
Evidence ratios: Model 1: {comparison['evidence_ratios'][0]:.1f}, Model 2: {comparison['evidence_ratios'][1]:.1f}, Model 3: {comparison['evidence_ratios'][2]:.1f}

Interpretation:
"""
    
    best_idx = int(comparison['best_model'].split()[1]) - 1
    if comparison['evidence_strength'] == "Very Strong":
        report += f"- {comparison['best_model']} is overwhelmingly superior\n"
    elif comparison['evidence_strength'] == "Strong":
        report += f"- {comparison['best_model']} is significantly better\n"
    elif comparison['evidence_strength'] == "Moderate":
        report += f"- {comparison['best_model']} is moderately better\n"
    else:
        report += f"- Models are comparable, prefer simpler model\n"
    
    # Physical interpretation based on best model
    best_result = comparison['all_results'][best_idx]
    if best_idx == 0:
        report += "- Standard Josephson behavior is adequate\n"
        report += "- No significant higher-order effects detected\n"
    elif best_idx == 1:
        report += "- Second-order effects are significant\n"
        report += "- Moderate anharmonicity or asymmetry present\n"
    else:
        report += "- Third-order effects are significant\n"
        report += "- Strong anharmonicity or complex coupling detected\n"
        report += "- System exhibits rich harmonic structure\n"
    
    report += f"""
================================================================================
RECOMMENDATIONS
================================================================================
1. Use {comparison['best_model']} for analysis and parameter extraction
2. {"Consider physical origins of higher-order terms" if best_idx > 0 else "Standard Josephson model is sufficient"}
3. {"Investigate system complexity and coupling mechanisms" if best_idx == 2 else "Standard or weakly anharmonic system"}

================================================================================
TECHNICAL DETAILS
================================================================================
- Frequency detection: Lomb-Scargle periodogram
- Parameter optimization: Levenberg-Marquardt algorithm
- Model comparison: Akaike Information Criterion (AIC)
- Statistical framework: Maximum likelihood estimation
- Generated by: Josephson Junction Analysis Toolkit v1.0

================================================================================
"""
    
    return report

def interpret_model_results(result):
    """
    Provide physical interpretation of model results
    """
    Ic = result.params['Ic'].value
    T = result.params['T'].value
    
    # Josephson energy (approximate)
    josephson_energy = Ic * 1.6e-19 / (2 * np.pi)  # in Joules
    
    # Transparency regime
    if T < 0.3:
        transparency_regime = "Low transparency (tunnel junction)"
    elif T < 0.7:
        transparency_regime = "Intermediate transparency"
    else:
        transparency_regime = "High transparency (weak link)"
    
    # Physical regime
    if "Model 1" in result.model_name:
        regime = "Standard Josephson"
    elif "Model 2" in result.model_name:
        regime = "Weakly anharmonic"
    else:
        regime = "Strongly anharmonic"
    
    return {
        'josephson_energy': josephson_energy,
        'transparency_regime': transparency_regime,
        'regime': regime
    }

def calculate_harmonic_strengths(result):
    """
    Calculate relative strengths of harmonic components for Model 3
    """
    if "Model 3" not in result.model_name:
        return None
    
    # For Model 3, estimate harmonic contributions
    # This is approximate and depends on the phase values
    phase_test = np.pi/4  # Test phase
    
    first_order = np.sin(phase_test)
    second_order = (np.sin(2 * phase_test)**2) / 2
    third_order = (np.sin(3 * phase_test)**3) / 3
    
    total = abs(first_order) + abs(second_order) + abs(third_order)
    
    return {
        'second_order': abs(second_order) / total if total > 0 else 0,
        'third_order': abs(third_order) / total if total > 0 else 0,
        'anharmonicity': "Strong" if abs(third_order) / abs(first_order) > 0.1 else "Moderate"
    }

def assess_physical_regime(result):
    """
    Assess the physical regime based on fitted parameters
    """
    T = result.params['T'].value
    Ic = result.params['Ic'].value
    
    # Basic assessments
    regime_info = {
        'junction_type': "Tunnel junction" if T < 0.5 else "Weak link",
        'current_scale': "Nano-scale" if Ic < 1e-9 else "Micro-scale" if Ic < 1e-6 else "Milli-scale",
        'complexity': result.model_name.split()[1]
    }
    
    return regime_info

# Example usage and testing
def main_triple_model_example():
    """
    Complete example demonstrating all three models
    """
    print("Josephson Junction Triple Model Analysis")
    print("=" * 55)
    
    # Generate test data with third-order effects
    np.random.seed(42)
    phi_ext = np.sort(np.random.uniform(-3, 3, 300))
    
    # True parameters with third-order effects
    true_params = {
        'Ic': 3e-6, 'T': 0.4, 'f': 1.2, 'd': 0.3, 'phi0': np.pi/4,
        'k': 0.01, 'r': 0.03, 'C': 5e-8
    }
    
    # Generate data with Model 3 (includes third-order)
    current_true = josephson_model3(phi_ext, **true_params)
    noise_level = 0.03 * np.std(current_true)
    current_noisy = current_true + np.random.normal(0, noise_level, len(phi_ext))
    current_err = np.full_like(current_noisy, noise_level)
    
    print(f"Generated data with true frequency: {true_params['f']:.6f}")
    print(f"Data includes up to third-order effects")
    
    # Automated model selection
    print("\n" + "="*50)
    print("AUTOMATED MODEL SELECTION")
    print("="*50)
    
    best_result = select_best_model(phi_ext, current_noisy, current_err)
    
    print(f"\nSelected: {best_result.model_name}")
    print(f"Evidence: {best_result.evidence_strength}")
    
    # Complete comparison
    print("\n" + "="*50)
    print("COMPLETE MODEL COMPARISON")
    print("="*50)
    
    comparison = compare_all_models(phi_ext, current_noisy, current_err)
    
    print(f"Model 1 AIC: {comparison['model1']['aic']:.2f}")
    print(f"Model 2 AIC: {comparison['model2']['aic']:.2f}")
    print(f"Model 3 AIC: {comparison['model3']['aic']:.2f}")
    print(f"Best model: {comparison['best_model']}")
    print(f"Evidence strength: {comparison['evidence_strength']}")
    
    # Harmonic analysis
    print("\n" + "="*50)
    print("HARMONIC CONTENT ANALYSIS")
    print("="*50)
    
    harmonic_analysis = analyze_harmonic_content(phi_ext, current_noisy, current_err)
    
    print(f"Fundamental power: {harmonic_analysis['fundamental_power']:.3f}")
    print(f"Second harmonic: {harmonic_analysis['second_harmonic_power']:.3f}")
    print(f"Third harmonic: {harmonic_analysis['third_harmonic_power']:.3f}")
    print(f"Recommended: {harmonic_analysis['recommended_model']}")
    
    # Statistical tests
    print("\n" + "="*50)
    print("STATISTICAL TESTS")
    print("="*50)
    
    f_tests = perform_nested_f_tests(*comparison['all_results'])
    
    print(f"F-test 1 vs 2: F={f_tests['f12']:.3f}, p={f_tests['p12']:.6f}")
    print(f"F-test 2 vs 3: F={f_tests['f23']:.3f}, p={f_tests['p23']:.6f}")
    print(f"F-test 1 vs 3: F={f_tests['f13']:.3f}, p={f_tests['p13']:.6f}")
    
    # Create visualization
    print("\n" + "="*50)
    print("GENERATING VISUALIZATION")
    print("="*50)
    
    fig = plot_triple_model_comparison(phi_ext, current_noisy, current_err, 
                                      *comparison['all_results'])
    plt.show()
    
    # Generate report
    report = generate_triple_model_report(phi_ext, current_noisy, current_err, 
                                        *comparison['all_results'], comparison)
    
    print("\n" + "="*50)
    print("ANALYSIS REPORT")
    print("="*50)
    print(report)
    
    return comparison

if __name__ == "__main__":
    comparison = main_triple_model_example()
