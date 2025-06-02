#!/usr/bin/env python
"""
Quick test to verify the MODEL_INFO KeyError fix works.
"""

import numpy as np
from josephson_fit import JosephsonTripleFitter
from josephson_fit.models import get_model

# Generate some test data
phi_ext = np.linspace(0, 4*np.pi, 100)
model_func = get_model(1)  # Model 1

# Create simple synthetic data
true_params = {'Ic': 0.005, 'T': 0.7, 'f': 1.0, 'd': 0.0, 'phi0': 0.0, 'k': 0.0, 'r': 0.0, 'C': 0.6}
current = model_func.evaluate(phi_ext, true_params) + np.random.normal(0, 0.01, len(phi_ext))
current_err = np.full_like(current, 0.01)

print("Testing Josephson Triple Fitter...")

# Create fitter instance
fitter = JosephsonTripleFitter(phi_ext, current, current_err)

try:
    # Test individual model fitting (the previous KeyError case)
    print("Testing Model 1 fitting...")
    result1 = fitter.fit_model('model1')
    print(f"✓ Model 1 fit successful: {result1.model_name}")
    
    print("Testing Model 2 fitting...")
    result2 = fitter.fit_model('model2')
    print(f"✓ Model 2 fit successful: {result2.model_name}")
    
    print("Testing Model 3 fitting...")
    result3 = fitter.fit_model('model3')
    print(f"✓ Model 3 fit successful: {result3.model_name}")
    
    # Test fit_all_models (which also had the KeyError)
    print("\nTesting fit_all_models...")
    results = fitter.fit_all_models()
    print(f"✓ fit_all_models successful: {list(results.keys())}")
    
    # Test model comparison
    print("\nTesting model comparison...")
    comparison = fitter.compare_models()
    print(f"✓ Model comparison successful")
    print(f"  Recommended model: {comparison['model_selection']['recommended_model']}")
    
    print("\n✅ All tests passed! The MODEL_INFO KeyError issue has been fixed.")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()
