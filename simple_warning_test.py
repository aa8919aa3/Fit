#!/usr/bin/env python3
"""
簡單的警告測試腳本
"""

import warnings
import numpy as np
from josephson_fit import JosephsonTripleFitter, generate_synthetic_data, configure_josephson_warnings
from josephson_fit.models import Model2

# Generate simple test data
phi_ext = np.linspace(-1, 1, 50)
params = {'Ic': 1.0e-6, 'T': 0.5, 'f': 1.0, 'd': 0.1, 'phi0': 0.1, 'k': 0.01, 'r': 0.0, 'C': 0.01}

print("=== 測試1: 無警告抑制 ===")
warnings.simplefilter('always')

try:
    current, _ = generate_synthetic_data(phi_ext, Model2().function, params, noise_level=0.01)
    print("✓ 數據生成成功")
    
    # Initialize fitter with data
    fitter = JosephsonTripleFitter(phi_ext, current)
    
    # Test without confidence calculation first
    result1 = fitter.fit_model(model_type='model1', calculate_confidence=False)
    print(f"✓ 擬合完成 (無信心區間): R²={result1.r_squared:.4f}")
    
    # Now test with confidence calculation (this should trigger warnings)
    print("\n--- 計算信心區間 (可能產生警告) ---")
    result2 = fitter.fit_model(model_type='model1', calculate_confidence=True)
    print(f"✓ 擬合完成 (含信心區間): R²={result2.r_squared:.4f}")
    
except Exception as e:
    print(f"❌ 錯誤: {e}")

print("\n=== 測試2: 啟用警告抑制 ===")
configure_josephson_warnings(verbose=False)

try:
    # Test with confidence calculation with suppression
    fitter2 = JosephsonTripleFitter(phi_ext, current)
    result3 = fitter2.fit_model(model_type='model1', calculate_confidence=True)
    print(f"✓ 擬合完成 (抑制警告): R²={result3.r_squared:.4f}")
    
except Exception as e:
    print(f"❌ 錯誤: {e}")

print("\n測試完成!")
