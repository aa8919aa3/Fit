# Josephson Junction Fitting Tutorial

This tutorial provides a comprehensive guide to using the Josephson Junction Fitting Toolkit.

## Table of Contents

1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Understanding the Models](#understanding-the-models)
4. [Data Preparation](#data-preparation)
5. [Parameter Estimation](#parameter-estimation)
6. [Model Selection](#model-selection)
7. [Advanced Analysis](#advanced-analysis)
8. [Command Line Usage](#command-line-usage)

## Installation

### From PyPI (when available)
```bash
pip install josephson-fit
```

### From Source
```bash
git clone https://github.com/username/josephson-fit.git
cd josephson-fit
pip install -e .
```

### Dependencies
The toolkit requires:
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- scipy >= 1.5.0
- astropy >= 4.0.0
- lmfit >= 1.0.0

## Basic Usage

### Quick Start Example

```python
import numpy as np
from josephson_fit import JosephsonFitter, generate_synthetic_data
import matplotlib.pyplot as plt

# Load or generate data
phi_ext = np.linspace(-2, 2, 200)
current = generate_synthetic_data(phi_ext, model_type=2, noise_level=0.02)

# Create fitter and fit all models
fitter = JosephsonFitter()
results = fitter.fit_all_models(phi_ext, current)

# Compare models and get the best one
best_model = fitter.compare_models(results)

# Plot results
fitter.plot_fit_comparison(phi_ext, current, results)
plt.show()
```

## Understanding the Models

### Model 1: Standard Josephson Junction
```
I_s(Φ_ext) = I_c·sin(2πf(Φ_ext-d)-φ₀) / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) 
             + k(Φ_ext-d)² + r(Φ_ext-d) + C
```

**Best for**: Simple tunnel junctions, single-mode devices

### Model 2: Second-Order Harmonics
```
I_s(Φ_ext) = I_c·[sin(2πf(Φ_ext-d)-φ₀) + sin²(2×(2πf(Φ_ext-d)-φ₀))/2] 
             / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) + k(Φ_ext-d)² + r(Φ_ext-d) + C
```

**Best for**: Multi-mode junctions, moderate anharmonicity

### Model 3: Third-Order Harmonics
```
I_s(Φ_ext) = I_c·[sin(2πf(Φ_ext-d)-φ₀) + sin²(2×(2πf(Φ_ext-d)-φ₀))/2 
              + sin³(3×(2πf(Φ_ext-d)-φ₀))/3] 
             / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) + k(Φ_ext-d)² + r(Φ_ext-d) + C
```

**Best for**: Strongly anharmonic systems, complex multi-junction arrays

## Data Preparation

### File Format
Data should be in a text file with two columns:
```
# phi_ext    current
-2.0        0.123
-1.9        0.145
...
```

### Loading Data
```python
import numpy as np

# Load from file
data = np.loadtxt('your_data.txt')
phi_ext = data[:, 0]
current = data[:, 1]

# Or define directly
phi_ext = np.linspace(-2, 2, 100)
current = your_measured_values
```

### Data Quality Check
```python
# Check for missing values
print(f"NaN values: {np.sum(np.isnan(current))}")

# Check data range
print(f"Phi range: {np.min(phi_ext):.2f} to {np.max(phi_ext):.2f}")
print(f"Current range: {np.min(current):.2f} to {np.max(current):.2f}")

# Plot raw data
plt.figure(figsize=(10, 6))
plt.plot(phi_ext, current, 'o-', markersize=3)
plt.xlabel('Φ_ext')
plt.ylabel('Current')
plt.title('Raw Data')
plt.grid(True)
plt.show()
```

## Parameter Estimation

### Automatic Initial Guess
```python
from josephson_fit.utils import estimate_initial_parameters

# Get automatic initial parameters
initial_params = estimate_initial_parameters(phi_ext, current)
print("Initial parameter estimates:")
for name, value in initial_params.items():
    print(f"  {name}: {value:.4f}")
```

### Manual Parameter Setting
```python
# Define custom initial parameters
custom_params = {
    'Ic': 2.5,      # Critical current
    'T': 0.4,       # Transparency
    'f': 1.2,       # Frequency
    'd': 0.1,       # Horizontal shift
    'phi0': 0.3,    # Phase offset
    'k': 0.05,      # Quadratic background
    'r': -0.1,      # Linear background
    'C': 0.2        # Constant offset
}

# Use in fitting
result = fitter.fit_model2(phi_ext, current, initial_params=custom_params)
```

### Frequency Analysis
```python
# Analyze frequency content before fitting
detected_freq, freq_fig = fitter.analyze_frequency(phi_ext, current)
print(f"Detected frequency: {detected_freq:.6f}")
```

## Model Selection

### Information Criteria
```python
# Fit all models
results = fitter.fit_all_models(phi_ext, current)

# Compare using AIC/BIC
print("Model Comparison:")
print(f"{'Model':<15} {'AIC':<10} {'BIC':<10} {'R²':<10}")
print("-" * 45)

for i, result in enumerate(results):
    if result is not None:
        print(f"Model {i+1:<9} {result.aic:<10.2f} {result.bic:<10.2f} {result.r_squared:<10.4f}")

# Get best model
best_model = fitter.compare_models(results)
print(f"\nBest model: {best_model.model.name}")
```

### Cross-Validation
```python
from sklearn.model_selection import KFold

def cross_validate_model(fitter, phi_ext, current, model_type, n_folds=5):
    """Perform cross-validation for model selection."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(phi_ext):
        phi_train, phi_val = phi_ext[train_idx], phi_ext[val_idx]
        current_train, current_val = current[train_idx], current[val_idx]
        
        # Fit on training data
        result = fitter.fit_model(phi_train, current_train, model_type)
        
        # Evaluate on validation data
        pred_val = result.model.evaluate(phi_val, result.params)
        score = 1 - np.var(current_val - pred_val) / np.var(current_val)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

# Compare models with cross-validation
for model_type in [1, 2, 3]:
    mean_score, std_score = cross_validate_model(fitter, phi_ext, current, model_type)
    print(f"Model {model_type}: R² = {mean_score:.4f} ± {std_score:.4f}")
```

## Advanced Analysis

### Parameter Uncertainty
```python
# Bootstrap analysis for uncertainty estimation
def bootstrap_uncertainty(fitter, phi_ext, current, model_type, n_bootstrap=100):
    """Estimate parameter uncertainties using bootstrap."""
    n_data = len(current)
    bootstrap_params = []
    
    for i in range(n_bootstrap):
        # Resample data
        indices = np.random.choice(n_data, n_data, replace=True)
        phi_boot = phi_ext[indices]
        current_boot = current[indices]
        
        try:
            result = fitter.fit_model(phi_boot, current_boot, model_type)
            bootstrap_params.append(result.params)
        except:
            continue
    
    # Calculate statistics
    param_stats = {}
    for param_name in ['Ic', 'T', 'f', 'd', 'phi0', 'k', 'r', 'C']:
        values = [params[param_name] for params in bootstrap_params]
        param_stats[param_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, 2.5),
            'ci_upper': np.percentile(values, 97.5)
        }
    
    return param_stats

# Run bootstrap analysis
uncertainty = bootstrap_uncertainty(fitter, phi_ext, current, 2)
print("\nParameter Uncertainties (95% CI):")
for param, stats in uncertainty.items():
    print(f"{param}: {stats['mean']:.4f} ± {stats['std']:.4f} "
          f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]")
```

### Residual Analysis
```python
def analyze_residuals(result):
    """Analyze fit residuals for diagnostic purposes."""
    residuals = result.residuals
    phi_ext = result.phi_ext
    
    # Statistical tests
    from scipy import stats
    
    # Normality test
    _, shapiro_p = stats.shapiro(residuals)
    
    # Autocorrelation test (runs test)
    runs, n_runs = 0, len(residuals)
    for i in range(1, n_runs):
        if (residuals[i] > 0) != (residuals[i-1] > 0):
            runs += 1
    
    expected_runs = (2 * np.sum(residuals > 0) * np.sum(residuals <= 0)) / n_runs + 1
    
    print(f"Residual Analysis:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std: {np.std(residuals):.6f}")
    print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"  Runs test: {runs} (expected: {expected_runs:.1f})")
    
    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs fitted
    axes[0,0].scatter(result.fit_curve, residuals, alpha=0.7)
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_xlabel('Fitted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title('Residuals vs Fitted')
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[0,1])
    axes[0,1].set_title('Q-Q Plot')
    
    # Histogram
    axes[1,0].hist(residuals, bins=20, alpha=0.7, density=True)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1,0].plot(x, stats.norm.pdf(x, np.mean(residuals), np.std(residuals)), 'r-')
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title('Residual Distribution')
    
    # Residuals vs phi_ext
    axes[1,1].scatter(phi_ext, residuals, alpha=0.7)
    axes[1,1].axhline(0, color='red', linestyle='--')
    axes[1,1].set_xlabel('Φ_ext')
    axes[1,1].set_ylabel('Residuals')
    axes[1,1].set_title('Residuals vs Φ_ext')
    
    plt.tight_layout()
    return fig

# Analyze residuals for best fit
analyze_residuals(best_model)
plt.show()
```

## Command Line Usage

### Basic Fitting
```bash
# Fit all models to data file
josephson-fit fit data.txt --model all --plot --output results

# Fit specific model
josephson-fit fit data.txt --model 2 --plot
```

### Generate Synthetic Data
```bash
# Generate Model 2 data with custom parameters
josephson-fit generate --model 2 --output synthetic.txt \
  --Ic 2.5 --T 0.4 --f 1.2 --noise 0.02 --plot
```

### Frequency Analysis
```bash
# Analyze frequency content
josephson-fit frequency data.txt --output freq_analysis --plot
```

## Tips and Best Practices

### Data Quality
- Ensure sufficient data points (>100 recommended)
- Check for outliers and systematic errors
- Verify proper external parameter range (should cover several periods)

### Model Selection
- Start with Model 1 for simple systems
- Use information criteria (AIC/BIC) for objective comparison
- Consider physical plausibility of fitted parameters

### Parameter Interpretation
- **Ic**: Should be positive and physically reasonable
- **T**: Should be between 0 and 1
- **f**: Related to flux-to-phase conversion
- **Background terms**: Should be small relative to Josephson signal

### Troubleshooting
- If fitting fails, try different initial parameters
- Check for data preprocessing needs (smoothing, outlier removal)
- Verify units and scales are appropriate
- Use synthetic data to validate fitting procedure

## Further Reading

- [Josephson Effect Physics](https://en.wikipedia.org/wiki/Josephson_effect)
- [lmfit Documentation](https://lmfit.github.io/lmfit-py/)
- [Astropy TimeSeries](https://docs.astropy.org/en/stable/timeseries/)

## Support

For questions and issues:
- Check the GitHub issues page
- Read the documentation
- Contact the maintainers

Happy fitting!
