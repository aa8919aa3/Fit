"""
Main fitting class for Josephson junction current-phase relationships.

This module provides the JosephsonFitter class which handles model fitting,
parameter optimization, and statistical analysis for all three Josephson models.
"""

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from typing import Dict, List, Optional, Tuple
import warnings
from scipy import stats
from astropy.timeseries import LombScargle
from datetime import datetime

from .models import Model1, Model2, Model3
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
    
    def fit_all_models(self):
        """
        Fit all three Josephson models and return comparison results
        """
        from .models import Model1, Model2, Model3
        
        print("Fitting all Josephson models...")
        print("=" * 40)
        
        # Fit each model
        result1 = self.fit_model(Model1)
        result2 = self.fit_model(Model2)
        result3 = self.fit_model(Model3)
        
        # Store results
        self.results = {
            'model1': result1,
            'model2': result2,
            'model3': result3
        }
        
        return [result1, result2, result3]
    
    def analyze_harmonic_content(self):
        """
        Analyze harmonic content to guide model selection
        """
        print("Analyzing harmonic content...")
        
        # First fit Model 1 to get basic parameters
        from .models import Model1
        result1 = self.fit_model(Model1)
        f_fundamental = result1.params['f'].value
        
        # Analyze residuals from Model 1
        residuals = self.current - result1.best_fit
        
        if self.current_err is not None:
            ls_residuals = LombScargle(self.phi_ext, residuals, dy=self.current_err)
        else:
            ls_residuals = LombScargle(self.phi_ext, residuals)
        
        # Check power at different harmonics
        freq_fund = np.linspace(f_fundamental * 0.8, f_fundamental * 1.2, 1000)
        freq_2nd = np.linspace(f_fundamental * 1.8, f_fundamental * 2.2, 1000)
        freq_3rd = np.linspace(f_fundamental * 2.8, f_fundamental * 3.2, 1000)
        
        power_fund = ls_residuals.power(freq_fund)
        power_2nd = ls_residuals.power(freq_2nd)
        power_3rd = ls_residuals.power(freq_3rd)
        
        # Get maximum powers
        fundamental_power = np.max(power_fund)
        second_harmonic_power = np.max(power_2nd)
        third_harmonic_power = np.max(power_3rd)
        
        # Significance thresholds (empirical)
        sig_fundamental = fundamental_power > 0.05
        sig_second = second_harmonic_power > 0.03
        sig_third = third_harmonic_power > 0.02
        
        # Recommend model based on harmonic content
        if sig_third:
            recommended_model = "Model 3"
        elif sig_second:
            recommended_model = "Model 2"
        else:
            recommended_model = "Model 1"
        
        return {
            'fundamental_power': fundamental_power,
            'second_harmonic_power': second_harmonic_power,
            'third_harmonic_power': third_harmonic_power,
            'significant_fundamental': sig_fundamental,
            'significant_second': sig_second,
            'significant_third': sig_third,
            'recommended_model': recommended_model,
            'residuals': residuals
        }
    
    def perform_nested_f_tests(self, result1, result2, result3):
        """
        Perform F-tests for nested model comparisons
        """
        def f_test_pair(simple_result, complex_result):
            dof_simple = len(simple_result.data) - simple_result.nvarys
            dof_complex = len(complex_result.data) - complex_result.nvarys
            
            f_stat = ((simple_result.chisqr - complex_result.chisqr) / (dof_simple - dof_complex)) / (complex_result.chisqr / dof_complex)
            p_value = 1 - stats.f.cdf(f_stat, dof_simple - dof_complex, dof_complex)
            
            return f_stat, p_value
        
        f12, p12 = f_test_pair(result1, result2)  # Model 1 vs 2
        f23, p23 = f_test_pair(result2, result3)  # Model 2 vs 3
        f13, p13 = f_test_pair(result1, result3)  # Model 1 vs 3
        
        return {
            'f12': f12, 'p12': p12,
            'f23': f23, 'p23': p23,
            'f13': f13, 'p13': p13
        }
    
    def select_best_model(self, selection_criterion='aic', evidence_threshold=2.0):
        """
        Automatically select the best model based on statistical criteria
        """
        print("Performing automated model selection...")
        print("=" * 45)
        
        # Fit all models
        results = self.fit_all_models()
        
        # Calculate selection criteria
        if selection_criterion.lower() == 'aic':
            criteria = [r.aic for r in results]
        elif selection_criterion.lower() == 'bic':
            criteria = [r.bic for r in results]
        else:
            raise ValueError("selection_criterion must be 'aic' or 'bic'")
        
        best_idx = np.argmin(criteria)
        best_result = results[best_idx]
        
        # Calculate evidence strength
        criteria_diff = np.array(criteria) - criteria[best_idx]
        evidence_ratios = np.exp(criteria_diff / 2)
        
        # Determine evidence strength
        if len(criteria_diff[criteria_diff > 7]) > 0:
            evidence_strength = "Very Strong"
        elif len(criteria_diff[criteria_diff > 4]) > 0:
            evidence_strength = "Strong"
        elif len(criteria_diff[criteria_diff > evidence_threshold]) > 0:
            evidence_strength = "Moderate"
        else:
            evidence_strength = "Weak"
        
        best_result.model_name = f"Model {best_idx + 1}"
        best_result.evidence_strength = evidence_strength
        best_result.evidence_ratios = evidence_ratios
        
        print(f"Selected Model {best_idx + 1} with {evidence_strength.lower()} evidence")
        print(f"{selection_criterion.upper()}: {criteria[best_idx]:.2f}")
        
        return best_result
    
    def compare_all_models(self):
        """
        Comprehensive comparison of all three models
        """
        print("Fitting and comparing all Josephson models...")
        
        # Fit all models
        results = self.fit_all_models()
        result1, result2, result3 = results
        
        # Calculate information criteria
        aics = [r.aic for r in results]
        bics = [r.bic for r in results]
        
        # Find best models
        best_aic_idx = np.argmin(aics)
        best_bic_idx = np.argmin(bics)
        
        # Calculate weights and ratios
        ic_weights = self.calculate_information_criteria_weights(results)
        
        # Determine evidence strength
        aic_min = min(aics)
        aic_diffs = [aic - aic_min for aic in aics]
        max_diff = max([diff for diff in aic_diffs if diff > 0]) if any(diff > 0 for diff in aic_diffs) else 0
        
        if max_diff >= 7:
            evidence_strength = "Very Strong"
        elif max_diff >= 4:
            evidence_strength = "Strong"
        elif max_diff >= 2:
            evidence_strength = "Moderate"
        else:
            evidence_strength = "Weak"
        
        comparison = {
            'model1': {'result': result1, 'aic': aics[0], 'bic': bics[0], 'redchi': result1.redchi},
            'model2': {'result': result2, 'aic': aics[1], 'bic': bics[1], 'redchi': result2.redchi},
            'model3': {'result': result3, 'aic': aics[2], 'bic': bics[2], 'redchi': result3.redchi},
            'best_model': f"Model {best_aic_idx + 1}",
            'best_result': results[best_aic_idx],
            'akaike_weights': ic_weights['akaike_weights'],
            'evidence_ratios': ic_weights['evidence_ratios'],
            'evidence_strength': evidence_strength,
            'all_results': results,
            'aic_differences': aic_diffs
        }
        
        self.comparison_results = comparison
        return comparison
    
    def calculate_information_criteria_weights(self, results):
        """
        Calculate Akaike weights for model comparison
        """
        aics = [r.aic for r in results]
        aic_min = min(aics)
        aic_differences = [aic - aic_min for aic in aics]
        
        # Akaike weights
        raw_weights = [np.exp(-0.5 * diff) for diff in aic_differences]
        weight_sum = sum(raw_weights)
        akaike_weights = [w / weight_sum for w in raw_weights]
        
        # Evidence ratios
        evidence_ratios = [np.exp(0.5 * diff) for diff in aic_differences]
        
        return {
            'akaike_weights': akaike_weights,
            'evidence_ratios': evidence_ratios,
            'aic_differences': aic_differences
        }
    
    def plot_triple_model_comparison(self):
        """
        Create comprehensive comparison plot for all three models
        """
        if not hasattr(self, 'comparison_results'):
            comparison = self.compare_all_models()
        else:
            comparison = self.comparison_results
        
        results = comparison['all_results']
        result1, result2, result3 = results
        
        fig = plt.figure(figsize=(18, 14))
        
        # Create grid layout
        gs = fig.add_gridspec(5, 3, height_ratios=[3, 1, 1, 1, 1], hspace=0.4, wspace=0.3)
        
        # Main data and fits
        ax_main = fig.add_subplot(gs[0, :])
        if self.current_err is not None:
            ax_main.errorbar(self.phi_ext, self.current, yerr=self.current_err, fmt='o', alpha=0.6, 
                            label='Data', markersize=4, color='black')
        else:
            ax_main.plot(self.phi_ext, self.current, 'o', alpha=0.6, label='Data', markersize=4, color='black')
        
        ax_main.plot(self.phi_ext, result1.best_fit, 'r-', linewidth=2, label='Model 1 (Standard)')
        ax_main.plot(self.phi_ext, result2.best_fit, 'g-', linewidth=2, label='Model 2 (Second-order)')
        ax_main.plot(self.phi_ext, result3.best_fit, 'b-', linewidth=2, label='Model 3 (Third-order)')
        
        ax_main.set_xlabel('Φ_ext')
        ax_main.set_ylabel('Current (A)')
        ax_main.legend()
        ax_main.set_title('Triple Model Comparison - Josephson Junction Analysis')
        ax_main.grid(True, alpha=0.3)
        
        # Individual residual plots
        colors = ['red', 'green', 'blue']
        model_names = ['Model 1', 'Model 2', 'Model 3']
        
        for i, (result, color, name) in enumerate(zip(results, colors, model_names)):
            ax_res = fig.add_subplot(gs[1, i])
            residuals = self.current - result.best_fit
            
            if self.current_err is not None:
                ax_res.errorbar(self.phi_ext, residuals, yerr=self.current_err, fmt='o', 
                              alpha=0.6, markersize=3, color=color)
            else:
                ax_res.plot(self.phi_ext, residuals, 'o', alpha=0.6, markersize=3, color=color)
            
            ax_res.axhline(y=0, color=color, linestyle='-', alpha=0.5)
            ax_res.set_xlabel('Φ_ext')
            ax_res.set_ylabel('Residuals')
            ax_res.set_title(f'{name} (χ²ᵣ = {result.redchi:.3f})')
            ax_res.grid(True, alpha=0.3)
        
        # Information criteria comparison
        ax_ic = fig.add_subplot(gs[2, :])
        criteria = ['AIC', 'BIC', 'Reduced χ²']
        model1_values = [result1.aic, result1.bic, result1.redchi]
        model2_values = [result2.aic, result2.bic, result2.redchi]
        model3_values = [result3.aic, result3.bic, result3.redchi]
        
        x = np.arange(len(criteria))
        width = 0.25
        
        ax_ic.bar(x - width, model1_values, width, label='Model 1', alpha=0.7, color='red')
        ax_ic.bar(x, model2_values, width, label='Model 2', alpha=0.7, color='green')
        ax_ic.bar(x + width, model3_values, width, label='Model 3', alpha=0.7, color='blue')
        
        ax_ic.set_xlabel('Criteria')
        ax_ic.set_ylabel('Value')
        ax_ic.set_title('Model Comparison Criteria')
        ax_ic.set_xticks(x)
        ax_ic.set_xticklabels(criteria)
        ax_ic.legend()
        ax_ic.grid(True, alpha=0.3)
        
        # Akaike weights
        ax_weights = fig.add_subplot(gs[3, 0])
        
        models = ['Model 1', 'Model 2', 'Model 3']
        ax_weights.bar(models, comparison['akaike_weights'], color=['red', 'green', 'blue'], alpha=0.7)
        ax_weights.set_ylabel('Akaike Weight')
        ax_weights.set_title('Model Weights')
        ax_weights.grid(True, alpha=0.3)
        
        # Parameter comparison for critical current
        ax_params = fig.add_subplot(gs[3, 1])
        Ic_values = [r.params['Ic'].value for r in results]
        Ic_errors = [r.params['Ic'].stderr or 0 for r in results]
        
        ax_params.errorbar(models, Ic_values, yerr=Ic_errors, fmt='o', capsize=5)
        ax_params.set_ylabel('Critical Current (A)')
        ax_params.set_title('Critical Current Comparison')
        ax_params.grid(True, alpha=0.3)
        
        # Transparency comparison
        ax_trans = fig.add_subplot(gs[3, 2])
        T_values = [r.params['T'].value for r in results]
        T_errors = [r.params['T'].stderr or 0 for r in results]
        
        ax_trans.errorbar(models, T_values, yerr=T_errors, fmt='s', capsize=5, color='orange')
        ax_trans.set_ylabel('Transparency T')
        ax_trans.set_title('Junction Transparency')
        ax_trans.grid(True, alpha=0.3)
        
        # Model selection summary
        ax_summary = fig.add_subplot(gs[4, :])
        ax_summary.axis('off')
        
        # Find best model
        aics = [r.aic for r in results]
        best_idx = np.argmin(aics)
        aic_diffs = [aic - aics[best_idx] for aic in aics]
        
        summary_text = f"""
Model Selection Summary:
Best Model: {model_names[best_idx]} (AIC = {aics[best_idx]:.2f})
AIC Differences: Model 1: {aic_diffs[0]:.2f}, Model 2: {aic_diffs[1]:.2f}, Model 3: {aic_diffs[2]:.2f}
Evidence Strength: {"Strong" if max(aic_diffs) > 4 else "Moderate" if max(aic_diffs) > 2 else "Weak"}
Akaike Weights: Model 1: {comparison['akaike_weights'][0]:.3f}, Model 2: {comparison['akaike_weights'][1]:.3f}, Model 3: {comparison['akaike_weights'][2]:.3f}
        """
        
        ax_summary.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.suptitle('Comprehensive Josephson Junction Model Analysis', fontsize=16, y=0.95)
        
        return fig
    
    def generate_analysis_report(self):
        """
        Generate comprehensive analysis report for all three models
        """
        if not hasattr(self, 'comparison_results'):
            comparison = self.compare_all_models()
        else:
            comparison = self.comparison_results
        
        results = comparison['all_results']
        result1, result2, result3 = results
        
        from datetime import datetime
        
        report = f"""
Josephson Junction Analysis Report - Triple Model Comparison
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

================================================================================
DATA SUMMARY
================================================================================
- Data points: {len(self.phi_ext)}
- Φ_ext range: {self.phi_ext.min():.4f} to {self.phi_ext.max():.4f}
- Current range: {self.current.min():.3e} to {self.current.max():.3e} A
- Mean uncertainty: {np.mean(self.current_err) if self.current_err is not None else "N/A"} A

================================================================================
MODEL COMPARISON RESULTS
================================================================================

Model 1 (Standard Josephson):
- Critical Current: {result1.params['Ic'].value:.3e} ± {result1.params['Ic'].stderr:.3e} A
- Junction Transparency: {result1.params['T'].value:.4f} ± {result1.params['T'].stderr:.4f}
- Frequency: {result1.params['f'].value:.6f} ± {result1.params['f'].stderr:.6f}
- Reduced χ²: {result1.redchi:.4f}
- AIC: {result1.aic:.2f}
- BIC: {result1.bic:.2f}
- Akaike Weight: {comparison['akaike_weights'][0]:.3f}

Model 2 (Second-order Josephson):
- Critical Current: {result2.params['Ic'].value:.3e} ± {result2.params['Ic'].stderr:.3e} A
- Junction Transparency: {result2.params['T'].value:.4f} ± {result2.params['T'].stderr:.4f}
- Frequency: {result2.params['f'].value:.6f} ± {result2.params['f'].stderr:.6f}
- Reduced χ²: {result2.redchi:.4f}
- AIC: {result2.aic:.2f}
- BIC: {result2.bic:.2f}
- Akaike Weight: {comparison['akaike_weights'][1]:.3f}

Model 3 (Third-order Josephson):
- Critical Current: {result3.params['Ic'].value:.3e} ± {result3.params['Ic'].stderr:.3e} A
- Junction Transparency: {result3.params['T'].value:.4f} ± {result3.params['T'].stderr:.4f}
- Frequency: {result3.params['f'].value:.6f} ± {result3.params['f'].stderr:.6f}
- Reduced χ²: {result3.redchi:.4f}
- AIC: {result3.aic:.2f}
- BIC: {result3.bic:.2f}
- Akaike Weight: {comparison['akaike_weights'][2]:.3f}

================================================================================
MODEL SELECTION
================================================================================
Best model (AIC): {comparison['best_model']}
Evidence strength: {comparison['evidence_strength']}
Evidence ratios: Model 1: {comparison['evidence_ratios'][0]:.1f}, Model 2: {comparison['evidence_ratios'][1]:.1f}, Model 3: {comparison['evidence_ratios'][2]:.1f}

================================================================================
RECOMMENDATIONS
================================================================================
1. Use {comparison['best_model']} for analysis and parameter extraction
2. Generated by: Josephson Junction Analysis Toolkit v1.0

================================================================================
        """
        
        return report
