#!/usr/bin/env python3

import numpy as np
import sys
import os
sys.path.insert(0, '/workspaces/Fit')

from josephson_fit.utils import lombscargle_frequency_detection

# 建立測試資料
phi_ext = np.linspace(-2, 2, 100)
frequency = 1.5
np.random.seed(42)
current = np.sin(2 * np.pi * frequency * phi_ext) + 0.1 * np.random.randn(len(phi_ext))

print('phi_ext range:', phi_ext.min(), 'to', phi_ext.max())
print('phi_ext step size (median):', np.median(np.diff(np.sort(phi_ext))))
print('Expected frequency:', frequency)

# 檢查頻率範圍計算
phi_range = np.ptp(phi_ext)
min_freq = 0.1 / phi_range
max_freq = min(10.0 / phi_range, 10.0)

print('phi_range:', phi_range)
print('min_freq:', min_freq)
print('max_freq (new calculation):', max_freq)

# 測試頻率檢測
result = lombscargle_frequency_detection(phi_ext, current)
print('Detected frequency:', result['fundamental_frequency'])
print('Power at detected frequency:', result['fundamental_power'])
