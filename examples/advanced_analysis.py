#!/usr/bin/env python3
"""
Advanced Analysis Example

This example demonstrates parameter uncertainty analysis, confidence intervals,
and detailed statistical evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from josephson_fit import JosephsonTripleFitter, generate_synthetic_data, Model2
from scipy import stats


def bootstrap_analysis(phi_ext, current, n_bootstrap=100):
    """
    Perform bootstrap analysis to estimate parameter uncertainties.
    
    Parameters:
    -----------
    phi_ext : np.ndarray
        External parameter values
    current : np.ndarray
        Current measurements
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns:
    --------
    dict
        Bootstrap parameter statistics
    """
    print(f"Performing bootstrap analysis with {n_bootstrap} samples...")
    
    n_data = len(current)
    bootstrap_params = {param: [] for param in ['Ic', 'T', 'f', 'd', 'phi0', 'k', 'r', 'C']}
    
    for i in range(n_bootstrap):
        if (i + 1) % 20 == 0:
            print(f"  Bootstrap sample {i+1}/{n_bootstrap}")
        
        # Resample data with replacement
        indices = np.random.choice(n_data, n_data, replace=True)
        phi_boot = phi_ext[indices]
        current_boot = current[indices]
        
        try:
            # Create fitter for this bootstrap sample
            fitter = JosephsonTripleFitter(phi_boot, current_boot)
            
            # Fit model 2 (assume this is the best model)
            result = fitter.fit_model('model2')
            
            if result is not None:
                # Store parameters
                for param_name in bootstrap_params.keys():
                    if param_name in result.params:
                        bootstrap_params[param_name].append(result.params[param_name])
                
        except Exception:
            # Skip failed fits
            continue
    
    # Calculate statistics
    bootstrap_statistics = {}
    for param_name, values in bootstrap_params.items():
        if values:
            bootstrap_statistics[param_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'values': np.array(values)
            }
    
    return bootstrap_statistics


def plot_parameter_distributions(bootstrap_statistics, true_params=None):
    """Plot parameter distributions from bootstrap analysis."""
    n_params = len(bootstrap_statistics)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for i, (param_name, param_stats) in enumerate(bootstrap_statistics.items()):
        ax = axes[i]
        
        # Histogram
        ax.hist(param_stats['values'], bins=30, alpha=0.7, density=True, 
               color='skyblue', edgecolor='black')
        
        # Statistics lines
        ax.axvline(param_stats['mean'], color='red', linestyle='-', linewidth=2,
                  label=f"Mean: {param_stats['mean']:.4f}")
        ax.axvline(param_stats['ci_lower'], color='orange', linestyle='--',
                  label="95% CI")
        ax.axvline(param_stats['ci_upper'], color='orange', linestyle='--')
        
        # True value if provided
        if true_params and param_name in true_params:
            ax.axvline(true_params[param_name], color='green', 
                      linestyle=':', linewidth=2, label='True value')
        
        ax.set_xlabel(f'{param_name}')
        ax.set_ylabel('Density')
        ax.set_title(f'Parameter {param_name} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def residual_analysis(fit_result):
    """Perform detailed residual analysis."""
    residuals = fit_result.residuals
    phi_ext = fit_result.phi_ext
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Residuals vs phi_ext
    ax1 = axes[0, 0]
    ax1.scatter(phi_ext, residuals, alpha=0.7, s=20)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Φ_ext')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs External Parameter')
    ax1.grid(True, alpha=0.3)
    
    # Residuals vs fitted values
    ax2 = axes[0, 1]
    ax2.scatter(fit_result.fit_curve, residuals, alpha=0.7, s=20)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residuals vs Fitted Values')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot for normality
    ax3 = axes[1, 0]
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot (Normality Test)')
    ax3.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax4 = axes[1, 1]
    ax4.hist(residuals, bins=30, alpha=0.7, density=True, 
            color='lightblue', edgecolor='black')
    
    # Overlay normal distribution
    x = np.linspace(residuals.min(), residuals.max(), 100)
    normal_pdf = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax4.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal fit')
    
    ax4.set_xlabel('Residuals')
    ax4.set_ylabel('Density')
    ax4.set_title('Residual Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Perform statistical tests
    _, shapiro_p = stats.shapiro(residuals)
    _, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
    
    print("\nResidual Analysis:")
    print("-" * 20)
    print(f"Residual mean: {np.mean(residuals):.6f}")
    print(f"Residual std: {np.std(residuals):.6f}")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"Kolmogorov-Smirnov p-value: {ks_p:.4f}")
    
    if shapiro_p > 0.05:
        print("Residuals appear normally distributed (Shapiro-Wilk)")
    else:
        print("Residuals may not be normally distributed (Shapiro-Wilk)")
    
    return fig


def main():
    """Run advanced analysis example."""
    print("Josephson Junction Fitting - Advanced Analysis")
    print("=" * 50)
    
    # Generate synthetic data
    phi_ext = np.linspace(-2, 2, 150)
    
    # True parameters for Model 2
    true_params = {
        'Ic': 3.0,
        'T': 0.35,
        'f': 1.5,
        'd': 0.05,
        'phi0': 0.2,
        'k': 0.03,
        'r': -0.08,
        'C': 0.15
    }
    
    # Generate data with realistic noise
    current, _ = generate_synthetic_data(phi_ext, Model2().function, 
                                       true_params, noise_level=0.025)
    
    print(f"Generated synthetic data with {len(phi_ext)} points")
    
    # Create fitter
    fitter = JosephsonTripleFitter(phi_ext, current)
    
    # Fit Model 2 (assuming we know this is the correct model)
    print("\nFitting Model 2...")
    result = fitter.fit_model('model2')
    
    if result is None:
        print("Fitting failed!")
        return
    
    # Residual analysis
    print("\n" + "="*50)
    residual_analysis(result)
    
    # Bootstrap analysis
    print("\n" + "="*50)
    bootstrap_statistics = bootstrap_analysis(phi_ext, current, n_bootstrap=50)
    
    print("\nBootstrap Parameter Statistics:")
    print("-" * 40)
    for param_name, param_stats in bootstrap_statistics.items():
        print(f"{param_name:>6}: {param_stats['mean']:8.4f} ± {param_stats['std']:8.4f} "
              f"[{param_stats['ci_lower']:7.4f}, {param_stats['ci_upper']:7.4f}]")
    
    # Plot parameter distributions
    plot_parameter_distributions(bootstrap_statistics, true_params)
    
    # Show all plots
    plt.show()
    
    print("\nAdvanced analysis complete!")


if __name__ == "__main__":
    main()
