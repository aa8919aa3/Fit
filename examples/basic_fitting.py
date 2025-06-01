#!/usr/bin/env python3
"""
Basic Fitting Example

This example demonstrates the basic workflow of fitting Josephson junction data
using the three different models.
"""

import numpy as np
import matplotlib.pyplot as plt
from josephson_fit import JosephsonFitter, generate_synthetic_data


def main():
    """Run basic fitting example."""
    print("Josephson Junction Fitting - Basic Example")
    print("=" * 50)
    
    # Generate synthetic data
    phi_ext = np.linspace(-2, 2, 200)
    
    # True parameters for Model 2
    true_params = {
        'Ic': 2.5,
        'T': 0.4,
        'f': 1.2,
        'd': 0.1,
        'phi0': 0.3,
        'k': 0.05,
        'r': -0.1,
        'C': 0.2
    }
    
    # Generate synthetic data with noise
    current = generate_synthetic_data(phi_ext, model_type=2, 
                                    params=true_params, noise_level=0.02)
    
    print(f"Generated synthetic data with {len(phi_ext)} points")
    print("True parameters:")
    for name, value in true_params.items():
        print(f"  {name}: {value}")
    
    # Create fitter
    fitter = JosephsonFitter(verbose=True)
    
    # Perform frequency analysis
    print("\nFrequency Analysis:")
    print("-" * 20)
    detected_freq, freq_fig = fitter.analyze_frequency(phi_ext, current)
    
    # Fit all models
    print("\nFitting all models...")
    print("-" * 30)
    results = fitter.fit_all_models(phi_ext, current)
    
    # Compare models
    best_model = fitter.compare_models(results)
    
    # Plot comparison
    fig = fitter.plot_fit_comparison(phi_ext, current, results)
    
    # Show plots
    plt.show()
    
    print(f"\nBest model: {best_model.model.name}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
