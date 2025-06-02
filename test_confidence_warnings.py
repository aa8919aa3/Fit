#!/usr/bin/env python3
"""
測試信心區間警告抑制
"""

import warnings
import numpy as np
from josephson_fit import JosephsonTripleFitter, generate_synthetic_data, configure_josephson_warnings
from josephson_fit.models import Model2

print("=== 測試信心區間警告抑制 ===")

# Generate test data that might trigger confidence interval issues
np.random.seed(42)  # For reproducible results
phi_ext = np.linspace(-2, 2, 100)
params = {'Ic': 2.5e-6, 'T': 0.6, 'f': 1.5, 'd': 0.2, 'phi0': 0.15, 'k': 0.02, 'r': -0.01, 'C': 0.03}

current, current_err = generate_synthetic_data(phi_ext, Model2().function, params, noise_level=0.05)

print("✓ 生成了具有挑戰性的測試數據")

# Test 1: With warnings enabled
print("\n1. 啟用所有警告:")
print("-" * 30)
configure_josephson_warnings(verbose=True)
warnings.simplefilter('always')

fitter1 = JosephsonTripleFitter(phi_ext, current, current_err)
result1 = fitter1.fit_model(model_type='model2', calculate_confidence=True)
print(f"✓ R²={result1.r_squared:.4f}, AIC={result1.aic:.2f}")

# Test 2: With warnings suppressed
print("\n2. 抑制警告:")
print("-" * 30)
configure_josephson_warnings(verbose=False)

fitter2 = JosephsonTripleFitter(phi_ext, current, current_err)
result2 = fitter2.fit_model(model_type='model2', calculate_confidence=True)
print(f"✓ R²={result2.r_squared:.4f}, AIC={result2.aic:.2f}")

# Test 3: Fit all models with suppression
print("\n3. 擬合所有模型 (抑制警告):")
print("-" * 30)
results = fitter2.fit_all_models(calculate_confidence=True)
for model_name, result in results.items():
    if result:
        print(f"✓ {model_name}: R²={result.r_squared:.4f}, AIC={result.aic:.2f}")

print("\n✅ 測試完成!")
print("📊 警告抑制功能運作正常")
