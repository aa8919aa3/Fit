#!/usr/bin/env python3
"""
Advanced Analysis Example

This example demonstrates parameter uncertainty analysis, confidence intervals,
and detailed statistical evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from josephson_fit import JosephsonFitter, generate_synthetic_data
from scipy import stats


def bootstrap_analysis(fitter, phi_ext, current, n_bootstrap=100):
    """
    Perform bootstrap analysis to estimate parameter uncertainties.
    
    Parameters:
    -----------
    fitter : JosephsonFitter
        Fitter instance
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
            # Fit best model (assume Model 2 for this example)
            result = fitter.fit_model2(phi_boot, current_boot)
            
            # Store parameters
            for param_name, value in result.params.items():
                bootstrap_params[param_name].append(value)
                
        except Exception:
            # Skip failed fits
            continue
    
    # Calculate statistics
    bootstrap_stats = {}
    for param_name, values in bootstrap_params.items():
        if values:
            bootstrap_stats[param_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'values': np.array(values)
            }
    
    return bootstrap_stats


def plot_parameter_distributions(bootstrap_stats, true_params=None):
    """Plot parameter distributions from bootstrap analysis."""
    n_params = len(bootstrap_stats)
    n_cols = 3
    n_rows = (n_params + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]
    
    for i, (param_name, stats) in enumerate(bootstrap_stats.items()):
        ax = axes[i]
        
        # Histogram
        ax.hist(stats['values'], bins=30, alpha=0.7, density=True, 
               color='skyblue', edgecolor='black')
        
        # Statistics lines
        ax.axvline(stats['mean'], color='red', linestyle='-', linewidth=2,
                  label=f"Mean: {stats['mean']:.4f}")
        ax.axvline(stats['ci_lower'], color='orange', linestyle='--',
                  label=f"95% CI")
        ax.axvline(stats['ci_upper'], color='orange', linestyle='--')
        
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
    current = generate_synthetic_data(phi_ext, model_type=2, 
                                    params=true_params, noise_level=0.025)
    
    print(f"Generated synthetic data with {len(phi_ext)} points")
    
    # Create fitter
    fitter = JosephsonFitter(verbose=True)
    
    # Fit Model 2 (assuming we know this is the correct model)
    print("\nFitting Model 2...")
    result = fitter.fit_model2(phi_ext, current)
    
    # Residual analysis
    print("\n" + "="*50)
    residual_fig = residual_analysis(result)
    
    # Bootstrap analysis
    print("\n" + "="*50)
    bootstrap_stats = bootstrap_analysis(fitter, phi_ext, current, n_bootstrap=50)
    
    print("\nBootstrap Parameter Statistics:")
    print("-" * 40)
    for param_name, stats in bootstrap_stats.items():
        print(f"{param_name:>6}: {stats['mean']:8.4f} ± {stats['std']:8.4f} "
              f"[{stats['ci_lower']:7.4f}, {stats['ci_upper']:7.4f}]")
    
    # Plot parameter distributions
    param_dist_fig = plot_parameter_distributions(bootstrap_stats, true_params)
    
    # Show all plots
    plt.show()
    
    print("\nAdvanced analysis complete!")


if __name__ == "__main__":
    main()
