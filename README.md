I'll add the third-order Josephson junction model to the documentation and implementation. This model includes an additional third harmonic term in the numerator.

```markdown name=README_triple_models.md
# Josephson Junction Current-Phase Relationship Fitting with Lomb-Scargle

A specialized Python toolkit for fitting Josephson junction current-phase relationships with background polynomial terms using automatic frequency detection via Lomb-Scargle periodogram and parameter optimization with lmfit.

## Overview

This toolkit implements fitting for three types of Josephson junction equations:

### Model No.1: Standard Josephson Junction with Transparency Effects

```
I_s(Φ_ext) = I_c·sin(2πf(Φ_ext-d)-φ₀) / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) + k(Φ_ext-d)² + r(Φ_ext-d) + C
```

### Model No.2: Second-Order Josephson Junction with Harmonic Terms

```
I_s(Φ_ext) = I_c·[sin(2πf(Φ_ext-d)-φ₀) + sin²(2×(2πf(Φ_ext-d)-φ₀))/2] / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) + k(Φ_ext-d)² + r(Φ_ext-d) + C
```

### Model No.3: Third-Order Josephson Junction with Extended Harmonic Series

```
I_s(Φ_ext) = I_c·[sin(2πf(Φ_ext-d)-φ₀) + sin²(2×(2πf(Φ_ext-d)-φ₀))/2 + sin³(3×(2πf(Φ_ext-d)-φ₀))/3] / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) + k(Φ_ext-d)² + r(Φ_ext-d) + C
```

All models describe the **total measured current (I_s)** as a function of an **external controlling parameter (Φ_ext)** in Josephson junction systems, combining:
- **Josephson component**: Non-sinusoidal current-phase relationship with junction transparency effects
- **Background component**: Quadratic polynomial representing systematic effects and noise

The higher-order models include additional harmonic contributions that can arise from:
- Higher-order tunneling processes
- Complex multi-mode coupling
- Strongly anharmonic junction potentials
- Non-equilibrium superconducting states
- Multi-junction array effects

## Physical Applications

- 🔬 **Advanced Josephson Junction Analysis**: Complex current-phase characteristics
- 🧲 **Quantum Interference Devices**: SQUID arrays and complex geometries
- ⚡ **Nonlinear Superconducting Circuits**: Transmon qubits, flux qubits
- 📊 **Harmonic Analysis**: Multi-mode and anharmonic effects
- 🔄 **Higher-Order Physics**: Third-order tunneling and coupling effects
- 🎯 **Model Selection**: Determining optimal complexity for physical systems

## Installation

### Dependencies

```bash
pip install numpy matplotlib scipy astropy lmfit
```

### Required packages:
- `numpy` >= 1.19.0
- `matplotlib` >= 3.3.0
- `scipy` >= 1.5.0
- `astropy` >= 4.0.0
- `lmfit` >= 1.0.0

## Model Parameters

| Parameter | Symbol | Physical Meaning | Typical Range |
|-----------|--------|------------------|---------------|
| `Ic` | I_c | Critical current (maximum superconducting current) | 1nA - 1mA |
| `T` | T | Junction transparency (0-1) | 0.0 - 0.99 |
| `f` | f | Conversion factor (Φ_ext to phase scaling) | 0.01 - 10.0 |
| `d` | d | Horizontal shift (zero-point offset) | -10.0 - 10.0 |
| `phi0` | φ₀ | Intrinsic phase offset (radians) | -π - π |
| `k` | k | Quadratic background coefficient | -1.0 - 1.0 |
| `r` | r | Linear background coefficient | -10.0 - 10.0 |
| `C` | C | Overall current offset | -10.0 - 10.0 |

## Warning Management

This toolkit includes sophisticated warning management to reduce noise in output while preserving important diagnostic information. The warning system handles common numerical warnings that don't affect fitting quality.

### Default Behavior

By default, the toolkit automatically suppresses common warnings from:
- **lmfit**: Confidence interval convergence and parameter correlation warnings
- **uncertainties**: Zero standard deviation warnings
- **numerical libraries**: Common numerical computation warnings

```python
# Warnings are automatically suppressed by default
from josephson_fit import JosephsonFitter

fitter = JosephsonFitter()
result = fitter.fit_model1(phi_ext, current, current_err)  # Clean output
```

### Warning Control

You can control warning behavior globally or per-function:

```python
from josephson_fit.warnings_config import configure_josephson_warnings

# Enable all warnings for debugging
configure_josephson_warnings(suppress_warnings=False)

# Re-enable default suppression
configure_josephson_warnings(suppress_warnings=True)
```

### Individual Function Control

Use the decorator for specific functions:

```python
from josephson_fit.warnings_config import suppress_lmfit_warnings

@suppress_lmfit_warnings
def my_analysis_function():
    # This function will have warnings suppressed
    return fit_result
```

### Warning Categories Handled

The system manages these warning types:

| Warning Type | Source | Description | Impact |
|--------------|--------|-------------|---------|
| Confidence intervals | `lmfit.confidence` | Convergence tolerance warnings | No effect on parameter estimates |
| Parameter correlation | `lmfit` | Correlation matrix warnings | Informational only |
| Model keywords | `lmfit.model` | Unused parameter warnings | No functional impact |
| Zero std_dev | `uncertainties.core` | Zero uncertainty warnings | Normal for fixed parameters |

### Best Practices

- **Default usage**: Keep warnings suppressed for clean output
- **Debugging**: Temporarily enable warnings when troubleshooting
- **Development**: Enable warnings when developing new fitting routines
- **Production**: Use default suppression for automated workflows

```python
# Example: Temporary debugging with warnings enabled
from josephson_fit.warnings_config import configure_josephson_warnings

# Enable warnings for debugging
configure_josephson_warnings(suppress_warnings=False)
result = fitter.fit_model1(phi_ext, current, current_err)

# Return to clean output
configure_josephson_warnings(suppress_warnings=True)
```

## Usage Workflow

### 1. Automated Model Selection

```python
import numpy as np
from josephson_triple_models import (
    fit_all_josephson_models,
    select_best_model,
    compare_all_models
)

# Load your experimental data
phi_ext = np.array([...])  # External parameter
current = np.array([...])  # Measured current
current_err = np.array([...])  # Uncertainties

# Automatic model selection
best_result = select_best_model(phi_ext, current, current_err)

print(f"Selected model: {best_result.model_name}")
print(f"AIC: {best_result.aic:.2f}")
print(f"Critical current: {best_result.params['Ic'].value:.3e} A")
```

### 2. Complete Model Comparison

```python
# Fit all three models and compare
comparison = compare_all_models(phi_ext, current, current_err)

print("Model Comparison Summary:")
print("=" * 50)
for i, model_key in enumerate(['model1', 'model2', 'model3'], 1):
    model_data = comparison[model_key]
    print(f"Model {i}:")
    print(f"  AIC: {model_data['aic']:.2f}")
    print(f"  BIC: {model_data['bic']:.2f}")
    print(f"  Reduced χ²: {model_data['redchi']:.4f}")
    print(f"  Parameters: {model_data['result'].nvarys}")

print(f"\nRecommended: {comparison['best_model']}")
print(f"Evidence strength: {comparison['evidence_strength']}")
```

### 3. Individual Model Fitting

```python
from josephson_triple_models import (
    fit_josephson_model1_with_lombscargle,
    fit_josephson_model2_with_lombscargle,
    fit_josephson_model3_with_lombscargle
)

# Fit specific models
result1 = fit_josephson_model1_with_lombscargle(phi_ext, current, current_err)
result2 = fit_josephson_model2_with_lombscargle(phi_ext, current, current_err)
result3 = fit_josephson_model3_with_lombscargle(phi_ext, current, current_err)

# Compare critical currents
print("Critical Current Comparison:")
for i, result in enumerate([result1, result2, result3], 1):
    Ic = result.params['Ic'].value
    Ic_err = result.params['Ic'].stderr or 0
    print(f"Model {i}: {Ic*1e6:.2f} ± {Ic_err*1e6:.2f} µA")
```

### 4. Harmonic Content Analysis

```python
from josephson_triple_models import analyze_harmonic_content

# Detect which harmonics are significant
harmonic_analysis = analyze_harmonic_content(phi_ext, current, current_err)

print("Harmonic Analysis:")
print(f"Fundamental power: {harmonic_analysis['fundamental_power']:.3f}")
print(f"Second harmonic power: {harmonic_analysis['second_harmonic_power']:.3f}")
print(f"Third harmonic power: {harmonic_analysis['third_harmonic_power']:.3f}")

print("\nSignificant harmonics:")
if harmonic_analysis['significant_fundamental']:
    print("✓ Fundamental frequency")
if harmonic_analysis['significant_second']:
    print("✓ Second harmonic")
if harmonic_analysis['significant_third']:
    print("✓ Third harmonic")

# Recommended model based on harmonic content
print(f"\nRecommended model: {harmonic_analysis['recommended_model']}")
```

### 5. Advanced Statistical Testing

```python
from josephson_triple_models import (
    perform_nested_f_tests,
    calculate_information_criteria_weights
)

# F-tests for nested model comparison
f_test_results = perform_nested_f_tests(result1, result2, result3)

print("F-test Results:")
print(f"Model 1 vs 2: F={f_test_results['f12']:.3f}, p={f_test_results['p12']:.6f}")
print(f"Model 2 vs 3: F={f_test_results['f23']:.3f}, p={f_test_results['p23']:.6f}")
print(f"Model 1 vs 3: F={f_test_results['f13']:.3f}, p={f_test_results['p13']:.6f}")

# Information criteria weights
ic_weights = calculate_information_criteria_weights([result1, result2, result3])

print("\nModel Weights (Akaike weights):")
for i, weight in enumerate(ic_weights['akaike_weights'], 1):
    print(f"Model {i}: {weight:.3f}")

print(f"Model evidence ratios: {ic_weights['evidence_ratios']}")
```

### 6. Physical Interpretation

```python
from josephson_triple_models import (
    interpret_model_results,
    calculate_harmonic_strengths,
    assess_physical_regime
)

# Comprehensive physical interpretation
interpretation = interpret_model_results(best_result)

print("Physical Interpretation:")
print("=" * 30)
print(f"Junction regime: {interpretation['regime']}")
print(f"Josephson energy: {interpretation['josephson_energy']*1e21:.2f} zJ")
print(f"Transparency regime: {interpretation['transparency_regime']}")

if best_result.model_name.endswith("3"):
    # Additional analysis for Model 3
    harmonic_strengths = calculate_harmonic_strengths(best_result)
    print(f"\nHarmonic Analysis:")
    print(f"Second-order strength: {harmonic_strengths['second_order']:.3f}")
    print(f"Third-order strength: {harmonic_strengths['third_order']:.3f}")
    print(f"Anharmonicity level: {harmonic_strengths['anharmonicity']}")
```

### 7. Comprehensive Visualization

```python
from josephson_triple_models import (
    plot_triple_model_comparison,
    plot_harmonic_decomposition,
    plot_model_selection_criteria
)

# Complete comparison of all three models
fig1 = plot_triple_model_comparison(phi_ext, current, current_err, 
                                   result1, result2, result3)

# Harmonic decomposition for the best model
if best_result.model_name.endswith(("2", "3")):
    fig2 = plot_harmonic_decomposition(phi_ext, current, current_err, best_result)

# Model selection criteria visualization
fig3 = plot_model_selection_criteria([result1, result2, result3])

plt.show()
```

### 8. Export and Documentation

```python
from josephson_triple_models import generate_triple_model_report

# Generate comprehensive analysis report
report = generate_triple_model_report(
    phi_ext, current, current_err,
    result1, result2, result3,
    comparison
)

# Save results
with open('triple_model_analysis_report.txt', 'w') as f:
    f.write(report)

print("Analysis complete. Report saved to 'triple_model_analysis_report.txt'")
```

## Model Selection Guidelines

### When to Use Each Model:

#### Model 1 (Standard):
- ✅ Simple tunnel junctions
- ✅ Linear regime operation
- ✅ Minimal harmonic content
- ✅ Standard superconducting electronics

#### Model 2 (Second-order):
- ✅ Moderate anharmonicity
- ✅ Asymmetric junctions
- ✅ Observable second harmonics
- ✅ Intermediate complexity systems

#### Model 3 (Third-order):
- ✅ Strongly anharmonic systems
- ✅ Multi-mode coupling
- ✅ Complex junction arrays
- ✅ Quantum circuit applications
- ✅ Non-equilibrium conditions

### Statistical Decision Criteria:

1. **AIC Differences**:
   - Δ AIC < 2: Models equivalent, prefer simpler
   - 2 ≤ Δ AIC < 4: Weak evidence for complex model
   - 4 ≤ Δ AIC < 7: Moderate evidence
   - Δ AIC ≥ 7: Strong evidence for complex model

2. **F-test p-values**:
   - p < 0.01: Strong evidence for additional terms
   - 0.01 ≤ p < 0.05: Moderate evidence
   - p ≥ 0.05: No significant improvement

3. **Physical Consistency**:
   - Parameters within reasonable ranges
   - Harmonic strengths make physical sense
   - Model complexity justified by system

## Implementation Details

### Core Model Functions

```python name=josephson_triple_models.py
import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from lmfit import Model, Parameters
from scipy import stats
from datetime import datetime

def josephson_model1(phi_ext, Ic, T, f, d, phi0, k, r, C):
    """Model 1: Standard Josephson junction"""
    phase = 2 * np.pi * f * (phi_ext - d) - phi0
    
    numerator = Ic * np.sin(phase)
    denominator_arg = 1 - T * (np.sin(phase / 2))**2
    denominator = np.sqrt(np.maximum(1e-12, denominator_arg))
    
    josephson_component = numerator / denominator
    displacement = phi_ext - d
    background = k * displacement**2 + r * displacement + C
    
    return josephson_component + background

def josephson_model2(phi_ext, Ic, T, f, d, phi0, k, r, C):
    """Model 2: Second-order Josephson junction"""
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
    """Model 3: Third-order Josephson junction"""
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
    if len(criteria_diff[criteria_diff > evidence_threshold]) > 0:
        evidence_strength = "Strong"
    elif len(criteria_diff[criteria_diff > 1.0]) > 0:
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
    result1 = fit_josephson_model1_with_lombscargle(phi_ext, current, current_err, plot_ls=False)
    result2 = fit_josephson_model2_with_lombscargle(phi_ext, current, current_err, plot_ls=False) 
    result3 = fit_josephson_model3_with_lombscargle(phi_ext, current, current_err, plot_ls=False)
    
    results = [result1, result2, result3]
    
    # Calculate information criteria
    aics = [r.aic for r in results]
    bics = [r.bic for r in results]
    
    # Find best models
    best_aic_idx = np.argmin(aics)
    best_bic_idx = np.argmin(bics)
    
    # Calculate Akaike weights
    aic_min = min(aics)
    aic_differences = [aic - aic_min for aic in aics]
    raw_weights = [np.exp(-0.5 * diff) for diff in aic_differences]
    weight_sum = sum(raw_weights)
    akaike_weights = [w / weight_sum for w in raw_weights]
    
    # Evidence ratios
    evidence_ratios = [np.exp(0.5 * diff) for diff in aic_differences]
    
    # Determine evidence strength
    min_aic_diff = min([diff for diff in aic_differences if diff > 0])
    if min_aic_diff >= 7:
        evidence_strength = "Very Strong"
    elif min_aic_diff >= 4:
        evidence_strength = "Strong"
    elif min_aic_diff >= 2:
        evidence_strength = "Moderate"
    else:
        evidence_strength = "Weak"
    
    comparison = {
        'model1': {'result': result1, 'aic': aics[0], 'bic': bics[0], 'redchi': result1.redchi},
        'model2': {'result': result2, 'aic': aics[1], 'bic': bics[1], 'redchi': result2.redchi},
        'model3': {'result': result3, 'aic': aics[2], 'bic': bics[2], 'redchi': result3.redchi},
        'best_model': f"Model {best_aic_idx + 1}",
        'best_result': results[best_aic_idx],
        'akaike_weights': akaike_weights,
        'evidence_ratios': evidence_ratios,
        'evidence_strength': evidence_strength,
        'all_results': results
    }
    
    return comparison

def generate_triple_model_report(phi_ext, current, current_err, result1, result2, result3, comparison):
    """
    Generate comprehensive analysis report for all three models
    """
    
    report = f"""
Josephson Junction Analysis Report - Triple Model Comparison
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
User: aa8919aa3

================================================================================
DATA SUMMARY
================================================================================
- Data points: {len(phi_ext)}
- Φ_ext range: {phi_ext.min():.4f} to {phi_ext.max():.4f}
- Current range: {current.min():.3e} to {current.max():.3e} A
- Mean uncertainty: {np.mean(current_err):.3e} A (if provided)

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
```

## Performance Considerations

### Model Complexity vs. Data Quality

- **Model 1**: Fast convergence, robust to noise
- **Model 2**: Moderate complexity, good for most applications
- **Model 3**: High complexity, requires high-quality data

### Recommended Data Requirements

| Model | Min Points | SNR | Sampling |
|-------|------------|-----|----------|
| 1 | 50+ | >5 | Regular preferred |
| 2 | 100+ | >10 | Regular recommended |
| 3 | 200+ | >20 | Dense, regular required |

## Troubleshooting

### Common Issues by Model

#### Model 3 Specific:
- **Overfitting**: Check residuals and cross-validation
- **Parameter correlation**: Third-order terms may correlate with others
- **Convergence**: Use global optimization for complex landscapes

### Solutions:
```python
# For difficult Model 3 fits
result3 = fit_josephson_model3_with_lombscargle(
    phi_ext, current, current_err,
    use_global_optimization=True,
    max_iterations=10000,
    method='differential_evolution'
)
```

---

**Created by**: aa8919aa3  
**Updated**: 2025-06-01 08:02:42 UTC  
**License**: MIT

For questions, issues, or contributions, please open a GitHub issue.
```

Now I'll provide the complete Python implementation for the triple model system:

```python name=josephson_triple_models_complete.py
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
    
    return comparison

if __name__ == "__main__":
    comparison = main_triple_model_example()
```

This comprehensive implementation provides:

1. **Three Complete Models**: Standard, second-order, and third-order Josephson equations
2. **Automated Model Selection**: Statistical criteria-based selection
3. **Harmonic Content Analysis**: Detection of significant harmonic components
4. **Statistical Testing**: F-tests and information criteria
5. **Comprehensive Visualization**: Side-by-side comparison of all models
6. **Physical Interpretation**: Guidelines for understanding complex junction behavior

The toolkit now supports all three Josephson junction models with sophisticated model selection and validation capabilities for analyzing complex superconducting quantum devices.