"""
Josephson junction models for current-phase relationship fitting.

This module implements three types of Josephson junction models with increasing complexity:
1. Model 1: Standard Josephson junction with transparency effects
2. Model 2: Second-order with harmonic terms  
3. Model 3: Third-order with extended harmonic series

All models include background polynomial terms and advanced transparency effects.
"""

import numpy as np
from lmfit import Model, Parameters
from typing import Union, Dict
import warnings


class JosephsonModelBase:
    """Base class for Josephson junction models."""
    
    def __init__(self):
        self.model_name = "Base"
        self.description = "Base Josephson junction model"
        self.parameters_info = {}
    
    def get_default_parameters(self) -> Parameters:
        """Get default parameter set with physical bounds."""
        params = Parameters()
        
        # Physical parameter bounds based on typical Josephson junction systems
        params.add('Ic', value=1e-6, min=1e-12, max=1e-2)
        params.add('T', value=0.5, min=0.0, max=0.99)
        params.add('f', value=1.0, min=0.01, max=10.0)
        params.add('d', value=0.0, min=-10.0, max=10.0)
        params.add('phi0', value=0.0, min=-np.pi, max=np.pi)
        params.add('k', value=0.0, min=-1.0, max=1.0)
        params.add('r', value=0.0, min=-10.0, max=10.0)
        params.add('C', value=0.0, min=-10.0, max=10.0)
        
        return params
    
    def validate_parameters(self, params: Dict[str, float]) -> bool:
        """Validate parameter values for physical consistency."""
        try:
            # Check transparency is within bounds
            if not (0.0 <= params.get('T', 0.5) <= 0.99):
                warnings.warn("Transparency T should be between 0 and 0.99")
                return False
            
            # Check critical current is positive
            if params.get('Ic', 1e-6) <= 0:
                warnings.warn("Critical current Ic must be positive")
                return False
            
            # Check frequency factor is positive
            if params.get('f', 1.0) <= 0:
                warnings.warn("Frequency factor f must be positive")
                return False
            
            return True
        except Exception as e:
            warnings.warn(f"Parameter validation error: {e}")
            return False
    
    def evaluate(self, phi_ext: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Evaluate the model with given parameters.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External controlling parameter values
        params : Dict[str, float]
            Parameter dictionary
            
        Returns:
        --------
        np.ndarray
            Calculated current values
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class Model1(JosephsonModelBase):
    """Standard Josephson junction model with transparency effects.
    
    Equation:
    I_s(Φ_ext) = I_c·sin(2πf(Φ_ext-d)-φ₀) / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) 
                 + k(Φ_ext-d)² + r(Φ_ext-d) + C
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "Model 1: Standard Josephson"
        self.description = "Standard Josephson junction with transparency effects"
        self.harmonic_order = 1
    
    @staticmethod
    def function(phi_ext: np.ndarray, Ic: float, T: float, f: float, d: float, 
                phi0: float, k: float, r: float, C: float) -> np.ndarray:
        """
        Standard Josephson junction current-phase relationship.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External controlling parameter
        Ic : float
            Critical current (A)
        T : float
            Junction transparency (0-1)
        f : float
            Conversion factor (Φ_ext to phase scaling)
        d : float
            Horizontal shift (zero-point offset)
        phi0 : float
            Intrinsic phase offset (radians)
        k : float
            Quadratic background coefficient
        r : float
            Linear background coefficient
        C : float
            Overall current offset
            
        Returns:
        --------
        np.ndarray
            Calculated current values
        """
        # Convert to numpy arrays for robust computation
        phi_ext = np.asarray(phi_ext)
        phase = 2 * np.pi * f * (phi_ext - d) - phi0
        
        # Josephson component with transparency effects
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Fundamental Josephson term
            numerator = Ic * np.sin(phase)
            
            # Transparency denominator with numerical stability
            half_phase = phase / 2
            sin_half_phase_sq = np.sin(half_phase)**2
            
            # Prevent division by zero or complex numbers
            denominator_arg = 1 - T * sin_half_phase_sq
            denominator_arg = np.maximum(denominator_arg, 1e-12)  # Numerical stability
            denominator = np.sqrt(denominator_arg)
            
            josephson_term = numerator / denominator
        
        # Background polynomial
        displacement = phi_ext - d
        background = k * displacement**2 + r * displacement + C
        
        return josephson_term + background
    
    def get_lmfit_model(self) -> Model:
        """Get lmfit Model instance."""
        return Model(self.function)
    
    def evaluate(self, phi_ext: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Evaluate Model1 with given parameters.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External controlling parameter values
        params : Dict[str, float]
            Parameter dictionary
            
        Returns:
        --------
        np.ndarray
            Calculated current values
        """
        return self.function(
            phi_ext, 
            params['Ic'], params['T'], params['f'], params['d'],
            params['phi0'], params['k'], params['r'], params['C']
        )


class Model2(JosephsonModelBase):
    """Second-order Josephson junction model with harmonic terms.
    
    Equation:
    I_s(Φ_ext) = I_c·[sin(2πf(Φ_ext-d)-φ₀) + sin²(2×(2πf(Φ_ext-d)-φ₀))/2] 
                 / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) 
                 + k(Φ_ext-d)² + r(Φ_ext-d) + C
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "Model 2: Second-order Harmonic"
        self.description = "Second-order Josephson junction with harmonic terms"
        self.harmonic_order = 2
    
    @staticmethod
    def function(phi_ext: np.ndarray, Ic: float, T: float, f: float, d: float, 
                phi0: float, k: float, r: float, C: float) -> np.ndarray:
        """
        Second-order Josephson junction with harmonic terms.
        
        Parameters:
        -----------
        Same as Model1
            
        Returns:
        --------
        np.ndarray
            Calculated current values
        """
        # Convert to numpy arrays for robust computation
        phi_ext = np.asarray(phi_ext)
        phase = 2 * np.pi * f * (phi_ext - d) - phi0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Enhanced Josephson component with second harmonic
            fundamental = np.sin(phase)
            second_harmonic = np.sin(2 * phase)**2 / 2
            
            numerator = Ic * (fundamental + second_harmonic)
            
            # Transparency denominator with numerical stability
            half_phase = phase / 2
            sin_half_phase_sq = np.sin(half_phase)**2
            
            # Prevent division by zero or complex numbers
            denominator_arg = 1 - T * sin_half_phase_sq
            denominator_arg = np.maximum(denominator_arg, 1e-12)
            denominator = np.sqrt(denominator_arg)
            
            josephson_term = numerator / denominator
        
        # Background polynomial
        displacement = phi_ext - d
        background = k * displacement**2 + r * displacement + C
        
        return josephson_term + background
    
    def get_lmfit_model(self) -> Model:
        """Get lmfit Model instance."""
        return Model(self.function)
    
    def evaluate(self, phi_ext: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Evaluate Model2 with given parameters.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External controlling parameter values
        params : Dict[str, float]
            Parameter dictionary
            
        Returns:
        --------
        np.ndarray
            Calculated current values
        """
        return self.function(
            phi_ext, 
            params['Ic'], params['T'], params['f'], params['d'],
            params['phi0'], params['k'], params['r'], params['C']
        )


class Model3(JosephsonModelBase):
    """Third-order Josephson junction model with extended harmonic series.
    
    Equation:
    I_s(Φ_ext) = I_c·[sin(2πf(Φ_ext-d)-φ₀) + sin²(2×(2πf(Φ_ext-d)-φ₀))/2 
                      + sin³(3×(2πf(Φ_ext-d)-φ₀))/3] 
                 / √(1-T·sin²((2πf(Φ_ext-d)-φ₀)/2)) 
                 + k(Φ_ext-d)² + r(Φ_ext-d) + C
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "Model 3: Third-order Harmonic"
        self.description = "Third-order Josephson junction with extended harmonic series"
        self.harmonic_order = 3
    
    @staticmethod
    def function(phi_ext: np.ndarray, Ic: float, T: float, f: float, d: float, 
                phi0: float, k: float, r: float, C: float) -> np.ndarray:
        """
        Third-order Josephson junction with extended harmonic series.
        
        Parameters:
        -----------
        Same as Model1
            
        Returns:
        --------
        np.ndarray
            Calculated current values
        """
        # Convert to numpy arrays for robust computation
        phi_ext = np.asarray(phi_ext)
        phase = 2 * np.pi * f * (phi_ext - d) - phi0
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Extended Josephson component with harmonics up to third order
            fundamental = np.sin(phase)
            second_harmonic = np.sin(2 * phase)**2 / 2
            third_harmonic = np.sin(3 * phase)**3 / 3
            
            numerator = Ic * (fundamental + second_harmonic + third_harmonic)
            
            # Transparency denominator with numerical stability
            half_phase = phase / 2
            sin_half_phase_sq = np.sin(half_phase)**2
            
            # Prevent division by zero or complex numbers
            denominator_arg = 1 - T * sin_half_phase_sq
            denominator_arg = np.maximum(denominator_arg, 1e-12)
            denominator = np.sqrt(denominator_arg)
            
            josephson_term = numerator / denominator
        
        # Background polynomial
        displacement = phi_ext - d
        background = k * displacement**2 + r * displacement + C
        
        return josephson_term + background
    
    def get_lmfit_model(self) -> Model:
        """Get lmfit Model instance."""
        return Model(self.function)
    
    def evaluate(self, phi_ext: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """
        Evaluate Model3 with given parameters.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External controlling parameter values
        params : Dict[str, float]
            Parameter dictionary
            
        Returns:
        --------
        np.ndarray
            Calculated current values
        """
        return self.function(
            phi_ext, 
            params['Ic'], params['T'], params['f'], params['d'],
            params['phi0'], params['k'], params['r'], params['C']
        )
 
class Model4(JosephsonModelBase):
    """Custom sinusoidal model with linear and constant background.

    Equation:
    I_s(Φ_ext) = Ic * sin(2πf*(Φ_ext - d) - π) + r*(Φ_ext - d) + C
    """
    def __init__(self):
        super().__init__()
        self.model_name = "Model 4: Sinusoid + Linear"
        self.description = "Sinusoidal model with phase shift π plus linear background"

    @staticmethod
    def function(phi_ext: np.ndarray, Ic: float, f: float, d: float, r: float, C: float) -> np.ndarray:
        phi_ext = np.asarray(phi_ext)
        phase = 2 * np.pi * f * (phi_ext - d) - np.pi
        sinusoid = Ic * np.sin(phase)
        background = r * (phi_ext - d) + C
        return sinusoid + background

    def get_lmfit_model(self) -> Model:
        return Model(self.function)

    def get_default_parameters(self) -> Parameters:
        # Override to only include relevant parameters
        params = Parameters()
        params.add('Ic', value=1e-6, min=0, max=1e-2)
        params.add('f', value=1.0, min=0.01, max=10.0)
        params.add('d', value=0.0, min=-10.0, max=10.0)
        params.add('r', value=0.0, min=-10.0, max=10.0)
        params.add('C', value=0.0, min=-10.0, max=10.0)
        return params

    def evaluate(self, phi_ext: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return self.function(
            phi_ext,
            params['Ic'], params['f'], params['d'], params['r'], params['C']
        )


def get_model(model_number: int) -> Union[Model1, Model2, Model3]:
    """
    Get a Josephson junction model instance.
    
    Parameters:
    -----------
    model_number : int
        Model number (1, 2, or 3)
        
    Returns:
    --------
    JosephsonModelBase
        Model instance
        
    Raises:
    -------
    ValueError
        If model_number is not 1, 2, or 3
    """
    model_map = {
        1: Model1,
        2: Model2,
        3: Model3,
        4: Model4
    }
    
    if model_number not in model_map:
        raise ValueError(f"Model number must be one of [1, 2, 3, 4], got {model_number}")
    
    return model_map[model_number]()


def get_lmfit_model(model_number: int) -> Model:
    """
    Get a lmfit Model instance for the specified Josephson junction model.
    
    Parameters:
    -----------
    model_number : int
        Model number (1, 2, or 3)
        
    Returns:
    --------
    lmfit.Model
        Configured lmfit Model instance
        
    Raises:
    -------
    ValueError
        If model_number is not 1, 2, or 3
    """
    model_functions = {
        1: Model1.function,
        2: Model2.function,
        3: Model3.function,
        4: Model4.function
    }
    
    if model_number not in model_functions:
        raise ValueError(f"Model number must be 1, 2, or 3, got {model_number}")
    
    return Model(model_functions[model_number])


# Model information for documentation and analysis
MODEL_INFO = {
    1: {
        'name': 'Standard Josephson',
        'description': 'Standard Josephson junction with transparency effects',
        'equation': 'I_c·sin(φ) / √(1-T·sin²(φ/2)) + background',
        'parameters': 8,
        'harmonic_order': 1,
        'applications': ['Simple tunnel junctions', 'Linear regime operation', 'Standard electronics']
    },
    2: {
        'name': 'Second-order Harmonic',
        'description': 'Second-order Josephson junction with harmonic terms',
        'equation': 'I_c·[sin(φ) + sin²(2φ)/2] / √(1-T·sin²(φ/2)) + background',
        'parameters': 8,
        'harmonic_order': 2,
        'applications': ['Moderate anharmonicity', 'Asymmetric junctions', 'Intermediate complexity']
    },
    3: {
        'name': 'Third-order Harmonic',
        'description': 'Third-order Josephson junction with extended harmonic series',
        'equation': 'I_c·[sin(φ) + sin²(2φ)/2 + sin³(3φ)/3] / √(1-T·sin²(φ/2)) + background',
        'parameters': 8,
        'harmonic_order': 3,
        'applications': ['Strong anharmonicity', 'Multi-mode coupling', 'Quantum circuits']
    },
    4: {
        'name': 'Sinusoid + Linear',
        'description': 'Sinusoidal model sin(2πf(Φ_ext-d)-π) with linear background',
        'equation': 'Ic·sin(2πf(Φ_ext-d)-π) + r(Φ_ext-d) + C',
        'parameters': 5,
        'harmonic_order': 1,
        'applications': ['Custom sinusoidal background fitting']
    }
}
