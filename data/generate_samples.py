#!/usr/bin/env python3
"""
Generate sample data files for testing and demonstration.
"""

import numpy as np
from josephson_fit import generate_synthetic_data, Model1, Model2, Model3


def create_sample_datasets():
    """Create various sample datasets."""
    
    # Dataset 1: Clean Model 1 data
    phi_ext1 = np.linspace(-2, 2, 200)
    params1 = {
        'Ic': 2.0, 'T': 0.3, 'f': 1.0, 'd': 0.0, 'phi0': 0.0,
        'k': 0.01, 'r': 0.0, 'C': 0.0
    }
    current1, _ = generate_synthetic_data(phi_ext1, Model1().function, 
                                        params1, noise_level=0.015)
    
    data1 = np.column_stack((phi_ext1, current1))
    np.savetxt('/workspaces/Fit/data/sample_model1.txt', data1,
               header='phi_ext\tcurrent\nModel 1 data: Ic=2.0, T=0.3, f=1.0, noise=1.5%',
               fmt='%.6f', delimiter='\t')
    
    # Dataset 2: Model 2 with harmonics
    phi_ext2 = np.linspace(-2.5, 2.5, 250)
    params2 = {
        'Ic': 2.8, 'T': 0.45, 'f': 1.3, 'd': 0.1, 'phi0': 0.25,
        'k': 0.03, 'r': -0.05, 'C': 0.15
    }
    current2, _ = generate_synthetic_data(phi_ext2, Model2().function,
                                        params2, noise_level=0.02)
    
    data2 = np.column_stack((phi_ext2, current2))
    np.savetxt('/workspaces/Fit/data/sample_model2.txt', data2,
               header='phi_ext\tcurrent\nModel 2 data: Ic=2.8, T=0.45, f=1.3, noise=2%',
               fmt='%.6f', delimiter='\t')
    
    # Dataset 3: Model 3 with strong higher-order effects
    phi_ext3 = np.linspace(-3, 3, 300)
    params3 = {
        'Ic': 3.5, 'T': 0.6, 'f': 1.5, 'd': -0.2, 'phi0': 0.4,
        'k': 0.04, 'r': 0.02, 'C': -0.1
    }
    current3, _ = generate_synthetic_data(phi_ext3, Model3().function,
                                        params3, noise_level=0.025)
    
    data3 = np.column_stack((phi_ext3, current3))
    np.savetxt('/workspaces/Fit/data/sample_model3.txt', data3,
               header='phi_ext\tcurrent\nModel 3 data: Ic=3.5, T=0.6, f=1.5, noise=2.5%',
               fmt='%.6f', delimiter='\t')
    
    # Dataset 4: High-frequency data
    phi_ext4 = np.linspace(-1, 1, 400)
    params4 = {
        'Ic': 1.8, 'T': 0.25, 'f': 4.0, 'd': 0.0, 'phi0': 0.0,
        'k': 0.005, 'r': 0.0, 'C': 0.0
    }
    current4, _ = generate_synthetic_data(phi_ext4, Model1().function,
                                        params4, noise_level=0.01)
    
    data4 = np.column_stack((phi_ext4, current4))
    np.savetxt('/workspaces/Fit/data/high_frequency.txt', data4,
               header='phi_ext\tcurrent\nHigh-frequency data: f=4.0, noise=1%',
               fmt='%.6f', delimiter='\t')
    
    # Dataset 5: Noisy data for robustness testing
    phi_ext5 = np.linspace(-2, 2, 150)
    params5 = {
        'Ic': 2.2, 'T': 0.35, 'f': 1.1, 'd': 0.05, 'phi0': 0.15,
        'k': 0.02, 'r': -0.01, 'C': 0.05
    }
    current5, _ = generate_synthetic_data(phi_ext5, Model2().function,
                                        params5, noise_level=0.05)
    
    data5 = np.column_stack((phi_ext5, current5))
    np.savetxt('/workspaces/Fit/data/noisy_data.txt', data5,
               header='phi_ext\tcurrent\nNoisy data: Model 2, noise=5%',
               fmt='%.6f', delimiter='\t')
    
    print("Created sample datasets:")
    print("  sample_model1.txt    - Clean Model 1 data")
    print("  sample_model2.txt    - Model 2 with harmonics")
    print("  sample_model3.txt    - Model 3 with strong higher-order effects")
    print("  high_frequency.txt   - High-frequency oscillations")
    print("  noisy_data.txt       - Challenging noisy data")


if __name__ == '__main__':
    create_sample_datasets()
