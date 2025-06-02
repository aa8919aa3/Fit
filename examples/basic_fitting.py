#!/usr/bin/env python3
"""
Basic Fitting Example

This example demonstrates the basic workflow of fitting Josephson junction data
using the three different models with the enhanced triple model system.
"""

import numpy as np
import matplotlib.pyplot as plt
from josephson_fit import (
    JosephsonTripleFitter,
    generate_synthetic_data
)
from josephson_fit.models import Model2
from josephson_fit.visualization import quick_plot, plot_fit_result, plot_model_comparison


def main():
    """Run basic fitting example."""
    print("Josephson Junction Fitting - Basic Example")
    print("=" * 50)
    
    # Generate synthetic data
    phi_ext = np.linspace(-2*np.pi, 2*np.pi, 200)
    
    # True parameters for Model2 (second-order model)
    true_params = {
        'Ic': 2.5e-6,
        'T': 0.6,
        'f': 1.2,
        'd': 0.0,
        'phi0': 0.3,
        'k': 0.0,
        'r': 0.0,
        'C': 0.0
    }
    
    # Generate synthetic data with noise using Model2 function
    model2 = Model2()
    current, current_err = generate_synthetic_data(
        phi_ext, 
        model2.function, 
        true_params, 
        noise_level=0.02,
        noise_type='gaussian'
    )
    
    print(f"Generated synthetic data with {len(phi_ext)} points")
    print("True parameters:")
    for name, value in true_params.items():
        print(f"  {name}: {value}")
    
    # Quick visualization of raw data
    print("\nVisualizing raw data...")
    quick_plot(phi_ext, current, title="Raw Current-Phase Data")
    plt.show()
    
    # Fit all models automatically
    print("\nFitting all models...")
    print("-" * 30)
    
    # Create fitter instance and fit all models
    fitter = JosephsonTripleFitter(phi_ext, current, current_err)
    results = fitter.fit_all_models(auto_frequency=True, verbose=True)
    
    # Display fit results for each model
    print("\nFit Results Summary:")
    print("-" * 40)
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        print(f"  R²: {result.r_squared:.4f}")
        print(f"  χ²/ν: {result.reduced_chi_square:.4f}")
        print("  Parameters:")
        for name, value in result.params.items():
            error = result.param_errors.get(name, None)
            if error is not None:
                print(f"    {name}: {value:.4e} ± {error:.4e}")
            else:
                print(f"    {name}: {value:.4e}")
    
    # Select best model using the fitter's method
    best_result = fitter.select_best_model(criterion='aic')
    print(f"\nBest model: {best_result.model_name}")
    print(f"Best model AIC: {best_result.aic:.2f}")
    
    # Create visualization
    print("\nCreating visualizations...")
    
    # Individual fit plot
    plot_fit_result(best_result, show_residuals=True)
    plt.show()
    
    # Model comparison plot - convert dict to list
    results_list = list(results.values())
    plot_model_comparison(results_list, show_aic_weights=True)
    plt.show()
    
    # Advanced analysis
    print("\nAdvanced Analysis:")
    print("-" * 25)
    
    # Physical interpretation
    physical_interp = best_result.fit_quality.get('physical_interpretation', {})
    if physical_interp:
        print("Physical interpretation:")
        for key, value in physical_interp.items():
            print(f"  {key}: {value}")
    
    # Quality assessment
    quality = best_result.fit_quality.get('overall_quality', 'Unknown')
    print(f"Fit quality assessment: {quality}")
    
    print("\nBasic fitting example completed successfully!")


if __name__ == "__main__":
    main()
