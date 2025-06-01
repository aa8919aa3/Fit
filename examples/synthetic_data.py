#!/usr/bin/env python3
"""
Synthetic Data Generation Example

This example shows how to generate synthetic Josephson junction data
for testing and validation purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from josephson_fit import JosephsonFitter, generate_synthetic_data


def explore_parameter_effects():
    """Explore how different parameters affect the Josephson junction curves."""
    phi_ext = np.linspace(-2, 2, 300)
    
    # Base parameters
    base_params = {
        'Ic': 2.0, 'T': 0.3, 'f': 1.0, 'd': 0.0, 'phi0': 0.0,
        'k': 0.0, 'r': 0.0, 'C': 0.0
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Effect of Critical Current (Ic)
    ax1 = axes[0, 0]
    for ic in [1.0, 2.0, 3.0, 4.0]:
        params = base_params.copy()
        params['Ic'] = ic
        current = generate_synthetic_data(phi_ext, model_type=1, 
                                        params=params, noise_level=0.0)
        ax1.plot(phi_ext, current, label=f'Ic = {ic}', linewidth=2)
    
    ax1.set_xlabel('Φ_ext')
    ax1.set_ylabel('Current')
    ax1.set_title('Effect of Critical Current (Ic)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Effect of Transparency (T)
    ax2 = axes[0, 1]
    for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
        params = base_params.copy()
        params['T'] = t
        params['Ic'] = 2.0
        current = generate_synthetic_data(phi_ext, model_type=1, 
                                        params=params, noise_level=0.0)
        ax2.plot(phi_ext, current, label=f'T = {t}', linewidth=2)
    
    ax2.set_xlabel('Φ_ext')
    ax2.set_ylabel('Current')
    ax2.set_title('Effect of Transparency (T)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Effect of Frequency (f)
    ax3 = axes[0, 2]
    for f in [0.5, 1.0, 1.5, 2.0]:
        params = base_params.copy()
        params['f'] = f
        params['Ic'] = 2.0
        current = generate_synthetic_data(phi_ext, model_type=1, 
                                        params=params, noise_level=0.0)
        ax3.plot(phi_ext, current, label=f'f = {f}', linewidth=2)
    
    ax3.set_xlabel('Φ_ext')
    ax3.set_ylabel('Current')
    ax3.set_title('Effect of Frequency (f)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Effect of Phase Offset (phi0)
    ax4 = axes[1, 0]
    for phi0 in [0.0, np.pi/4, np.pi/2, 3*np.pi/4]:
        params = base_params.copy()
        params['phi0'] = phi0
        params['Ic'] = 2.0
        current = generate_synthetic_data(phi_ext, model_type=1, 
                                        params=params, noise_level=0.0)
        ax4.plot(phi_ext, current, label=f'φ₀ = {phi0:.2f}', linewidth=2)
    
    ax4.set_xlabel('Φ_ext')
    ax4.set_ylabel('Current')
    ax4.set_title('Effect of Phase Offset (φ₀)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Effect of Background (k, r, C)
    ax5 = axes[1, 1]
    params = base_params.copy()
    params['Ic'] = 2.0
    
    # No background
    current1 = generate_synthetic_data(phi_ext, model_type=1, 
                                     params=params, noise_level=0.0)
    ax5.plot(phi_ext, current1, label='No background', linewidth=2)
    
    # Linear background
    params['r'] = 0.3
    current2 = generate_synthetic_data(phi_ext, model_type=1, 
                                     params=params, noise_level=0.0)
    ax5.plot(phi_ext, current2, label='Linear (r=0.3)', linewidth=2)
    
    # Quadratic background
    params['k'] = 0.1
    current3 = generate_synthetic_data(phi_ext, model_type=1, 
                                     params=params, noise_level=0.0)
    ax5.plot(phi_ext, current3, label='Quadratic (k=0.1)', linewidth=2)
    
    # Offset
    params['C'] = 0.5
    current4 = generate_synthetic_data(phi_ext, model_type=1, 
                                     params=params, noise_level=0.0)
    ax5.plot(phi_ext, current4, label='With offset (C=0.5)', linewidth=2)
    
    ax5.set_xlabel('Φ_ext')
    ax5.set_ylabel('Current')
    ax5.set_title('Effect of Background Terms')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Model Comparison
    ax6 = axes[1, 2]
    params = {
        'Ic': 2.0, 'T': 0.4, 'f': 1.2, 'd': 0.0, 'phi0': 0.2,
        'k': 0.02, 'r': 0.0, 'C': 0.0
    }
    
    for model_type in [1, 2, 3]:
        current = generate_synthetic_data(phi_ext, model_type=model_type, 
                                        params=params, noise_level=0.0)
        ax6.plot(phi_ext, current, label=f'Model {model_type}', linewidth=2)
    
    ax6.set_xlabel('Φ_ext')
    ax6.set_ylabel('Current')
    ax6.set_title('Comparison of Model Types')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def noise_level_study():
    """Study the effect of different noise levels on fitting."""
    phi_ext = np.linspace(-2, 2, 150)
    
    # True parameters
    true_params = {
        'Ic': 2.5, 'T': 0.4, 'f': 1.3, 'd': 0.1, 'phi0': 0.3,
        'k': 0.03, 'r': -0.05, 'C': 0.1
    }
    
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    fitter = JosephsonFitter(verbose=False)
    
    results_summary = []
    
    for i, noise_level in enumerate(noise_levels):
        if i >= len(axes):
            break
            
        # Generate data with noise
        current = generate_synthetic_data(phi_ext, model_type=2, 
                                        params=true_params, 
                                        noise_level=noise_level)
        
        # Fit the data
        try:
            result = fitter.fit_model2(phi_ext, current)
            
            # Plot
            ax = axes[i]
            ax.scatter(phi_ext, current, alpha=0.6, s=15, color='blue', 
                      label='Data')
            ax.plot(phi_ext, result.fit_curve, 'red', linewidth=2, 
                   label='Fit')
            
            ax.set_xlabel('Φ_ext')
            ax.set_ylabel('Current')
            ax.set_title(f'Noise Level: {noise_level:.1%}\n'
                        f'R² = {result.r_squared:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Store results
            param_errors = {}
            for param_name in true_params.keys():
                true_val = true_params[param_name]
                fitted_val = result.params[param_name]
                relative_error = abs(fitted_val - true_val) / abs(true_val)
                param_errors[param_name] = relative_error
            
            results_summary.append({
                'noise_level': noise_level,
                'r_squared': result.r_squared,
                'param_errors': param_errors,
                'mean_error': np.mean(list(param_errors.values()))
            })
            
        except Exception as e:
            ax = axes[i]
            ax.text(0.5, 0.5, f'Fit Failed\nNoise: {noise_level:.1%}\nError: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle="round", facecolor='salmon'))
            ax.set_title(f'Noise Level: {noise_level:.1%}')
    
    # Hide unused subplot
    if len(noise_levels) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    
    # Print summary
    print("\nNoise Level Study Results:")
    print("=" * 50)
    print(f"{'Noise':<8} {'R²':<10} {'Mean Param Error':<18}")
    print("-" * 50)
    
    for result in results_summary:
        print(f"{result['noise_level']:<8.1%} {result['r_squared']:<10.4f} "
              f"{result['mean_error']:<18.4f}")
    
    return fig, results_summary


def main():
    """Run synthetic data generation examples."""
    print("Josephson Junction Fitting - Synthetic Data Examples")
    print("=" * 60)
    
    print("\n1. Exploring Parameter Effects...")
    param_fig = explore_parameter_effects()
    
    print("\n2. Noise Level Study...")
    noise_fig, noise_results = noise_level_study()
    
    # Additional demonstration: Generate data for different applications
    print("\n3. Application-Specific Data Generation...")
    
    # High-frequency regime
    phi_ext_hf = np.linspace(-1, 1, 300)
    hf_params = {
        'Ic': 1.5, 'T': 0.2, 'f': 5.0, 'd': 0.0, 'phi0': 0.0,
        'k': 0.01, 'r': 0.0, 'C': 0.0
    }
    current_hf = generate_synthetic_data(phi_ext_hf, model_type=1, 
                                       params=hf_params, noise_level=0.015)
    
    # Low transparency (tunnel junction)
    lt_params = {
        'Ic': 3.0, 'T': 0.05, 'f': 1.0, 'd': 0.0, 'phi0': 0.0,
        'k': 0.0, 'r': 0.0, 'C': 0.0
    }
    current_lt = generate_synthetic_data(phi_ext_hf, model_type=1, 
                                       params=lt_params, noise_level=0.02)
    
    # High transparency (ballistic junction)  
    ht_params = {
        'Ic': 2.0, 'T': 0.85, 'f': 1.0, 'd': 0.0, 'phi0': 0.0,
        'k': 0.0, 'r': 0.0, 'C': 0.0
    }
    current_ht = generate_synthetic_data(phi_ext_hf, model_type=1, 
                                       params=ht_params, noise_level=0.02)
    
    # Plot application examples
    app_fig, app_axes = plt.subplots(1, 3, figsize=(18, 6))
    
    app_axes[0].plot(phi_ext_hf, current_hf, 'b-', linewidth=1.5)
    app_axes[0].set_title('High-Frequency Regime\n(f = 5.0)')
    app_axes[0].set_xlabel('Φ_ext')
    app_axes[0].set_ylabel('Current')
    app_axes[0].grid(True, alpha=0.3)
    
    app_axes[1].plot(phi_ext_hf, current_lt, 'r-', linewidth=1.5)
    app_axes[1].set_title('Tunnel Junction\n(T = 0.05)')
    app_axes[1].set_xlabel('Φ_ext')
    app_axes[1].set_ylabel('Current')
    app_axes[1].grid(True, alpha=0.3)
    
    app_axes[2].plot(phi_ext_hf, current_ht, 'g-', linewidth=1.5)
    app_axes[2].set_title('Ballistic Junction\n(T = 0.85)')
    app_axes[2].set_xlabel('Φ_ext')
    app_axes[2].set_ylabel('Current')
    app_axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Show all plots
    plt.show()
    
    print("\nSynthetic data generation examples complete!")
    print("Generated datasets demonstrate:")
    print("- Parameter sensitivity analysis")
    print("- Noise effects on fitting accuracy")
    print("- Application-specific scenarios")


if __name__ == "__main__":
    main()
