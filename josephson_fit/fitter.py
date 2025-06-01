"""
Main fitting class for Josephson junction current-phase relationships.

This module provides the JosephsonFitter class which handles model fitting,
parameter optimization, and statistical analysis for all three Josephson models.
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from typing import Dict, List, Optional, Tuple, Any
import warnings

from .models import Model1, Model2, Model3, MODELS
from .utils import (
    estimate_initial_parameters,
    calculate_aic,
    calculate_bic,
    plot_periodogram
)


class FitResult:
    """Container for fitting results."""
    
    def __init__(self, lmfit_result, model_instance, phi_ext, current):
        self.lmfit_result = lmfit_result
        self.model = model_instance
        self.phi_ext = phi_ext
        self.current = current
        
        # Extract key metrics
        self.params = {name: param.value for name, param in lmfit_result.params.items()}
        self.param_errors = {name: param.stderr for name, param in lmfit_result.params.items()}
        self.chi_squared = lmfit_result.chisqr
        self.reduced_chi_squared = lmfit_result.redchi
        self.r_squared = 1 - lmfit_result.residual.var() / np.var(current)
        
        # Information criteria
        n_data = len(current)
        n_params = len(self.params)
        self.aic = calculate_aic(n_params, n_data, self.chi_squared)
        self.bic = calculate_bic(n_params, n_data, self.chi_squared)
        
        # Best fit curve
        self.fit_curve = self.model.evaluate(phi_ext, self.params)
        self.residuals = current - self.fit_curve
    
    def summary(self) -> str:
        """Generate a summary string of the fit results."""
        summary = f"\n{self.model.name} Fit Results:\n"
        summary += "=" * 50 + "\n"
        summary += f"Chi-squared: {self.chi_squared:.6f}\n"
        summary += f"Reduced Chi-squared: {self.reduced_chi_squared:.6f}\n"
        summary += f"R-squared: {self.r_squared:.6f}\n"
        summary += f"AIC: {self.aic:.2f}\n"
        summary += f"BIC: {self.bic:.2f}\n\n"
        
        summary += "Parameters:\n"
        summary += "-" * 30 + "\n"
        for name, value in self.params.items():
            error = self.param_errors[name]
            if error is not None:
                summary += f"{name:>6}: {value:8.4f} ± {error:8.4f}\n"
            else:
                summary += f"{name:>6}: {value:8.4f} ± {'N/A':>8}\n"
        
        return summary


class JosephsonFitter:
    """
    Main fitting class for Josephson junction current-phase relationships.
    
    This class provides methods to fit experimental data to the three different
    Josephson junction models using Lomb-Scargle frequency detection and lmfit
    optimization.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the fitter.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print fitting progress and results
        """
        self.verbose = verbose
        self.models = {1: Model1(), 2: Model2(), 3: Model3()}
    
    def _create_lmfit_model(self, model_instance) -> Model:
        """Create an lmfit Model from a Josephson model instance."""
        def model_func(phi_ext, Ic, T, f, d, phi0, k, r, C):
            params_dict = {
                'Ic': Ic, 'T': T, 'f': f, 'd': d,
                'phi0': phi0, 'k': k, 'r': r, 'C': C
            }
            return model_instance.evaluate(phi_ext, params_dict)
        
        return Model(model_func)
    
    def _create_parameters(self, initial_params: Dict[str, float]) -> Parameters:
        """Create lmfit Parameters object with bounds."""
        params = Parameters()
        
        # Parameter bounds based on physical constraints
        bounds = {
            'Ic': (1e-6, 1e6),     # Critical current: 1 µA to 1 A
            'T': (0.001, 0.999),   # Transparency: nearly 0 to nearly 1
            'f': (0.001, 100.0),   # Frequency: positive
            'd': (-100.0, 100.0),  # Shift: reasonable range
            'phi0': (-np.pi, np.pi), # Phase: one period
            'k': (-10.0, 10.0),    # Quadratic coeff
            'r': (-100.0, 100.0),  # Linear coeff
            'C': (-1000.0, 1000.0) # Offset
        }
        
        for name, value in initial_params.items():
            min_val, max_val = bounds[name]
            params.add(name, value=value, min=min_val, max=max_val)
        
        return params
    
    def fit_model(self, phi_ext: np.ndarray, 
                  current: np.ndarray,
                  model_type: int,
                  initial_params: Optional[Dict[str, float]] = None) -> FitResult:
        """
        Fit a specific Josephson model to data.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External parameter values
        current : np.ndarray
            Measured current values
        model_type : int
            Model type (1, 2, or 3)
        initial_params : dict, optional
            Initial parameter guesses. If None, auto-estimated.
            
        Returns:
        --------
        FitResult
            Object containing fit results and statistics
        """
        if model_type not in [1, 2, 3]:
            raise ValueError("model_type must be 1, 2, or 3")
        
        model_instance = self.models[model_type]
        
        if self.verbose:
            print(f"Fitting {model_instance.name}...")
        
        # Estimate initial parameters if not provided
        if initial_params is None:
            initial_params = estimate_initial_parameters(phi_ext, current)
            if self.verbose:
                print("Auto-estimated initial parameters:")
                for name, value in initial_params.items():
                    print(f"  {name}: {value:.4f}")
        
        # Create lmfit model and parameters
        lmfit_model = self._create_lmfit_model(model_instance)
        params = self._create_parameters(initial_params)
        
        # Perform fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = lmfit_model.fit(current, params, phi_ext=phi_ext)
            
            fit_result = FitResult(result, model_instance, phi_ext, current)
            
            if self.verbose:
                print(fit_result.summary())
            
            return fit_result
            
        except Exception as e:
            print(f"Fitting failed: {str(e)}")
            raise
    
    def fit_model1(self, phi_ext: np.ndarray, current: np.ndarray,
                   initial_params: Optional[Dict[str, float]] = None) -> FitResult:
        """Fit Model 1 (Standard Josephson Junction)."""
        return self.fit_model(phi_ext, current, 1, initial_params)
    
    def fit_model2(self, phi_ext: np.ndarray, current: np.ndarray,
                   initial_params: Optional[Dict[str, float]] = None) -> FitResult:
        """Fit Model 2 (Second-Order Harmonics)."""
        return self.fit_model(phi_ext, current, 2, initial_params)
    
    def fit_model3(self, phi_ext: np.ndarray, current: np.ndarray,
                   initial_params: Optional[Dict[str, float]] = None) -> FitResult:
        """Fit Model 3 (Third-Order Harmonics)."""
        return self.fit_model(phi_ext, current, 3, initial_params)
    
    def fit_all_models(self, phi_ext: np.ndarray, current: np.ndarray) -> List[FitResult]:
        """
        Fit all three models and return results.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External parameter values
        current : np.ndarray
            Measured current values
            
        Returns:
        --------
        list
            List of FitResult objects for models 1, 2, and 3
        """
        results = []
        
        # Use same initial parameters for all models
        initial_params = estimate_initial_parameters(phi_ext, current)
        
        for model_type in [1, 2, 3]:
            try:
                result = self.fit_model(phi_ext, current, model_type, initial_params)
                results.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"Model {model_type} fitting failed: {str(e)}")
                results.append(None)
        
        return results
    
    def compare_models(self, results: List[FitResult]) -> FitResult:
        """
        Compare models using information criteria and return the best one.
        
        Parameters:
        -----------
        results : list
            List of FitResult objects
            
        Returns:
        --------
        FitResult
            Best model based on AIC
        """
        valid_results = [r for r in results if r is not None]
        
        if not valid_results:
            raise ValueError("No valid fit results to compare")
        
        # Find best model by AIC (lower is better)
        best_result = min(valid_results, key=lambda r: r.aic)
        
        if self.verbose:
            print("\nModel Comparison:")
            print("=" * 60)
            print(f"{'Model':<25} {'AIC':<12} {'BIC':<12} {'R²':<12}")
            print("-" * 60)
            
            for result in valid_results:
                print(f"{result.model.name:<25} {result.aic:<12.2f} "
                      f"{result.bic:<12.2f} {result.r_squared:<12.4f}")
            
            print("-" * 60)
            print(f"Best model: {best_result.model.name} (lowest AIC)")
        
        return best_result
    
    def plot_fit_comparison(self, phi_ext: np.ndarray, current: np.ndarray,
                           results: List[FitResult], 
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot data and fit comparisons for multiple models.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External parameter values
        current : np.ndarray
            Measured current values
        results : list
            List of FitResult objects
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        plt.Figure
            The matplotlib figure object
        """
        valid_results = [r for r in results if r is not None]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Main plot: data and fits
        ax1 = axes[0, 0]
        ax1.scatter(phi_ext, current, alpha=0.7, s=20, color='black', 
                   label='Data', zorder=1)
        
        colors = ['red', 'blue', 'green']
        for i, result in enumerate(valid_results):
            ax1.plot(phi_ext, result.fit_curve, 
                    color=colors[i % len(colors)], linewidth=2,
                    label=f'{result.model.name} (AIC: {result.aic:.1f})')
        
        ax1.set_xlabel('Φ_ext')
        ax1.set_ylabel('Current')
        ax1.set_title('Data and Model Fits')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        ax2 = axes[0, 1]
        for i, result in enumerate(valid_results):
            ax2.scatter(phi_ext, result.residuals, 
                       color=colors[i % len(colors)], alpha=0.7, s=15,
                       label=result.model.name)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Φ_ext')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Fit Residuals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Model comparison bar chart
        ax3 = axes[1, 0]
        model_names = [r.model.name.split(':')[0] for r in valid_results]
        aics = [r.aic for r in valid_results]
        bics = [r.bic for r in valid_results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax3.bar(x - width/2, aics, width, label='AIC', alpha=0.8)
        ax3.bar(x + width/2, bics, width, label='BIC', alpha=0.8)
        
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Information Criterion')
        ax3.set_title('Model Selection Criteria')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # R² comparison
        ax4 = axes[1, 1]
        r_squareds = [r.r_squared for r in valid_results]
        bars = ax4.bar(model_names, r_squareds, alpha=0.8, color=colors[:len(valid_results)])
        
        # Add value labels on bars
        for i, (bar, r2) in enumerate(zip(bars, r_squareds)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.4f}', ha='center', va='bottom')
        
        ax4.set_xlabel('Model')
        ax4.set_ylabel('R²')
        ax4.set_title('Goodness of Fit (R²)')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_frequency(self, phi_ext: np.ndarray, current: np.ndarray,
                         save_path: Optional[str] = None) -> Tuple[float, plt.Figure]:
        """
        Perform frequency analysis using Lomb-Scargle periodogram.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External parameter values
        current : np.ndarray
            Measured current values
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        tuple
            (dominant_frequency, figure) - detected frequency and plot
        """
        from .utils import lombscargle_frequency_detection
        
        freq, power = lombscargle_frequency_detection(phi_ext, current)
        fig = plot_periodogram(phi_ext, current, save_path=save_path)
        
        if self.verbose:
            print(f"Detected dominant frequency: {freq:.6f}")
            print(f"Lomb-Scargle power: {power:.4f}")
        
        return freq, fig
