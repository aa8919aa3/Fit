"""
Mathematical models for Josephson junction current-phase relationships.

This module contains the three Josephson junction models with increasing complexity:
- Model1: Standard Josephson junction with transparency effects
- Model2: Second-order harmonics
- Model3: Third-order harmonics
"""

import numpy as np
from typing import Union


class BaseJosephsonModel:
    """Base class for Josephson junction models."""
    
    def __init__(self, name: str, n_params: int):
        self.name = name
        self.n_params = n_params
    
    def evaluate(self, phi_ext: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate the model at given phi_ext values with parameters."""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def get_param_names(self) -> list:
        """Get list of parameter names."""
        return ['Ic', 'T', 'f', 'd', 'phi0', 'k', 'r', 'C']


class Model1(BaseJosephsonModel):
    """
    Standard Josephson Junction with Transparency Effects
    
    I_s(Φ_ext) = I_c·sin(2πf(Φ_ext-d)-φ₀) / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) 
                 + k(Φ_ext-d)² + r(Φ_ext-d) + C
    """
    
    def __init__(self):
        super().__init__("Model 1: Standard Josephson", 8)
    
    def evaluate(self, phi_ext: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate Model 1."""
        Ic = params['Ic']
        T = params['T']
        f = params['f']
        d = params['d']
        phi0 = params['phi0']
        k = params['k']
        r = params['r']
        C = params['C']
        
        # Calculate phase
        phase = 2 * np.pi * f * (phi_ext - d) - phi0
        
        # Josephson component with transparency correction
        sin_half_phase = np.sin(phase / 2)
        denominator = np.sqrt(1 - T * sin_half_phase**2)
        josephson_current = Ic * np.sin(phase) / denominator
        
        # Background polynomial
        x_shifted = phi_ext - d
        background = k * x_shifted**2 + r * x_shifted + C
        
        return josephson_current + background


class Model2(BaseJosephsonModel):
    """
    Second-Order Josephson Junction with Harmonic Terms
    
    I_s(Φ_ext) = I_c·[sin(2πf(Φ_ext-d)-φ₀) + sin²(2×(2πf(Φ_ext-d)-φ₀))/2] 
                 / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) + k(Φ_ext-d)² + r(Φ_ext-d) + C
    """
    
    def __init__(self):
        super().__init__("Model 2: Second-Order Harmonics", 8)
    
    def evaluate(self, phi_ext: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate Model 2."""
        Ic = params['Ic']
        T = params['T']
        f = params['f']
        d = params['d']
        phi0 = params['phi0']
        k = params['k']
        r = params['r']
        C = params['C']
        
        # Calculate phase
        phase = 2 * np.pi * f * (phi_ext - d) - phi0
        
        # Josephson component with second-order harmonics
        sin_phase = np.sin(phase)
        sin2_2phase = np.sin(2 * phase)**2
        numerator = sin_phase + sin2_2phase / 2
        
        # Transparency correction
        sin_half_phase = np.sin(phase / 2)
        denominator = np.sqrt(1 - T * sin_half_phase**2)
        josephson_current = Ic * numerator / denominator
        
        # Background polynomial
        x_shifted = phi_ext - d
        background = k * x_shifted**2 + r * x_shifted + C
        
        return josephson_current + background


class Model3(BaseJosephsonModel):
    """
    Third-Order Josephson Junction with Extended Harmonic Series
    
    I_s(Φ_ext) = I_c·[sin(2πf(Φ_ext-d)-φ₀) + sin²(2×(2πf(Φ_ext-d)-φ₀))/2 
                      + sin³(3×(2πf(Φ_ext-d)-φ₀))/3] 
                 / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) + k(Φ_ext-d)² + r(Φ_ext-d) + C
    """
    
    def __init__(self):
        super().__init__("Model 3: Third-Order Harmonics", 8)
    
    def evaluate(self, phi_ext: np.ndarray, params: dict) -> np.ndarray:
        """Evaluate Model 3."""
        Ic = params['Ic']
        T = params['T']
        f = params['f']
        d = params['d']
        phi0 = params['phi0']
        k = params['k']
        r = params['r']
        C = params['C']
        
        # Calculate phase
        phase = 2 * np.pi * f * (phi_ext - d) - phi0
        
        # Josephson component with third-order harmonics
        sin_phase = np.sin(phase)
        sin2_2phase = np.sin(2 * phase)**2
        sin3_3phase = np.sin(3 * phase)**3
        numerator = sin_phase + sin2_2phase / 2 + sin3_3phase / 3
        
        # Transparency correction
        sin_half_phase = np.sin(phase / 2)
        denominator = np.sqrt(1 - T * sin_half_phase**2)
        josephson_current = Ic * numerator / denominator
        
        # Background polynomial
        x_shifted = phi_ext - d
        background = k * x_shifted**2 + r * x_shifted + C
        
        return josephson_current + background


# Dictionary for easy model access
MODELS = {
    1: Model1,
    2: Model2,
    3: Model3
}
