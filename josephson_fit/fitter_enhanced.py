"""
Enhanced Josephson junction fitting module for triple model system.

This module provides comprehensive fitting capabilities for three Josephson junction models
with sophisticated model selection, statistical analysis, and visualization tools.
"""

import numpy as np
import warnings
from lmfit import Parameters
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from .models import get_model, get_lmfit_model, MODEL_INFO
from .utils import (
    lombscargle_frequency_detection,
    estimate_initial_parameters,
    analyze_harmonic_content,
    calculate_information_criteria,
    perform_f_test,
    calculate_akaike_weights,
    validate_fit_quality,
    interpret_physical_parameters
)


@dataclass
class JosephsonFitResult:
    """
    Comprehensive container for Josephson junction fitting results.
    """
    # Core fit data
    model_name: str
    model_type: str
    lmfit_result: Any
    phi_ext: np.ndarray
    current: np.ndarray
    current_err: Optional[np.ndarray]
    
    # Fitted parameters
    params: Dict[str, float]
    param_errors: Dict[str, Optional[float]]
    param_bounds: Dict[str, Tuple[float, float]]
    
    # Statistical metrics
    chi_square: float
    reduced_chi_square: float
    r_squared: float
    aic: float
    bic: float
    aicc: float
    
    # Quality assessment
    fit_quality: Dict[str, Any]
    parameter_correlations: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    # Physical interpretation
    physical_interpretation: Dict[str, Any]
    
    # Frequency analysis
    frequency_analysis: Dict[str, Any]
    
    # Model predictions
    fitted_curve: np.ndarray
    residuals: np.ndarray
    prediction_intervals: Optional[Tuple[np.ndarray, np.ndarray]]
    
    @property
    def critical_current(self) -> float:
        """Get critical current in Amperes."""
        return self.params['Ic']
    
    @property
    def critical_current_uA(self) -> float:
        """Get critical current in microamperes."""
        return self.params['Ic'] * 1e6
    
    @property
    def transparency(self) -> float:
        """Get junction transparency."""
        return self.params['T']
    
    @property
    def is_good_fit(self) -> bool:
        """Quick assessment if fit is of good quality."""
        return (self.fit_quality.get('quality_rating', 'Poor') in ['Good', 'Excellent'] and
                self.reduced_chi_square < 3.0 and
                self.r_squared > 0.7)


class JosephsonTripleFitter:
    """
    Enhanced Josephson junction fitting class for triple model system.
    
    This class provides comprehensive fitting capabilities including:
    - Automatic frequency detection using Lomb-Scargle periodogram
    - Intelligent parameter initialization
    - Model selection and comparison
    - Statistical validation and quality assessment
    - Physical interpretation of results
    - Advanced visualization tools
    """
    
    def __init__(self, phi_ext: np.ndarray, current: np.ndarray, 
                 current_err: Optional[np.ndarray] = None):
        """
        Initialize the fitter with experimental data.
        
        Parameters:
        -----------
        phi_ext : np.ndarray
            External parameter values (e.g., flux, voltage)
        current : np.ndarray
            Measured current values
        current_err : np.ndarray, optional
            Current measurement uncertainties
        """
        self.phi_ext = np.asarray(phi_ext)
        self.current = np.asarray(current)
        self.current_err = np.asarray(current_err) if current_err is not None else None
        
        # Validate input data
        self._validate_input_data()
        
        # Initialize storage for results
        self.results = {}
        self.frequency_analysis = None
        self.harmonic_analysis = None
        
        # Configuration
        self.fit_config = {
            'method': 'leastsq',
            'max_nfev': 2000,
            'ftol': 1e-12,
            'xtol': 1e-12,
            'gtol': 1e-12,
            'verbose': False
        }
    
    def _validate_input_data(self):
        """Validate input data arrays."""
        if len(self.phi_ext) != len(self.current):
            raise ValueError("phi_ext and current must have the same length")
        
        if len(self.phi_ext) < 10:
            raise ValueError("Need at least 10 data points for reliable fitting")
        
        if self.current_err is not None and len(self.current_err) != len(self.current):
            raise ValueError("current_err must have the same length as current")
        
        if np.any(~np.isfinite(self.phi_ext)) or np.any(~np.isfinite(self.current)):
            raise ValueError("Input data contains non-finite values")
        
        if self.current_err is not None and (np.any(self.current_err <= 0) or np.any(~np.isfinite(self.current_err))):
            raise ValueError("current_err must be positive and finite")
    
    def analyze_frequency_content(self, **kwargs) -> Dict[str, Any]:
        """
        Perform Lomb-Scargle frequency analysis of the data.
        
        Parameters:
        -----------
        **kwargs : dict
            Additional parameters for frequency detection
            
        Returns:
        --------
        dict
            Comprehensive frequency analysis results
        """
        self.frequency_analysis = lombscargle_frequency_detection(
            self.phi_ext, self.current, **kwargs
        )
        return self.frequency_analysis
    
    def analyze_harmonic_content(self) -> Dict[str, Any]:
        """
        Analyze harmonic content to guide model selection.
        
        Returns:
        --------
        dict
            Harmonic analysis results with model recommendations
        """
        self.harmonic_analysis = analyze_harmonic_content(
            self.phi_ext, self.current, self.current_err
        )
        return self.harmonic_analysis
    
    def fit_model(self, model_type: str, 
                  custom_params: Optional[Parameters] = None,
                  **fit_kwargs) -> JosephsonFitResult:
        """
        Fit a specific Josephson model to the data.
        
        Parameters:
        -----------
        model_type : str
            Model type ('model1', 'model2', or 'model3')
        custom_params : Parameters, optional
            Custom initial parameters (if None, auto-estimated)
        **fit_kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        JosephsonFitResult
            Comprehensive fitting results
        """
        if model_type not in ['model1', 'model2', 'model3']:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Convert model type string to integer for model functions
        model_number_map = {'model1': 1, 'model2': 2, 'model3': 3}
        model_number = model_number_map[model_type]
        
        # Ensure frequency analysis is done
        if self.frequency_analysis is None:
            self.analyze_frequency_content()
        
        # Get model function and lmfit model
        model_func = get_model(model_number)
        lmfit_model = get_lmfit_model(model_number)
        
        # Estimate initial parameters if not provided
        if custom_params is None:
            initial_params = estimate_initial_parameters(
                self.phi_ext, self.current, self.frequency_analysis or {}
            )
        else:
            initial_params = custom_params
        
        # Configure fitting options
        fit_options = {**self.fit_config, **fit_kwargs}
        
        # Perform fit
        if self.current_err is not None:
            weights = 1.0 / self.current_err
        else:
            weights = None
        
        result = lmfit_model.fit(
            self.current, 
            initial_params, 
            phi_ext=self.phi_ext,
            weights=weights,
            **fit_options
        )
        
        # Create comprehensive result object
        fit_result = self._create_fit_result(result, model_type, model_func)
        
        # Store result
        self.results[model_type] = fit_result
        
        return fit_result
    
    def _create_fit_result(self, lmfit_result, model_type: str, model_func) -> JosephsonFitResult:
        """Create comprehensive fit result object."""
        
        # Convert model_type string to integer for MODEL_INFO access
        model_number_map = {'model1': 1, 'model2': 2, 'model3': 3}
        
        # Extract basic parameters
        params = {name: param.value for name, param in lmfit_result.params.items()}
        param_errors = {name: param.stderr for name, param in lmfit_result.params.items()}
        param_bounds = {name: (param.min, param.max) for name, param in lmfit_result.params.items()}
        
        # Calculate fitted curve and residuals
        fitted_curve = model_func(self.phi_ext, **params)
        residuals = self.current - fitted_curve
        
        # Calculate R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((self.current - np.mean(self.current))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate information criteria
        ic_results = calculate_information_criteria(lmfit_result, len(self.current))
        
        # Validate fit quality
        fit_quality = validate_fit_quality(lmfit_result, self.phi_ext, self.current, self.current_err)
        
        # Physical interpretation
        physical_interpretation = interpret_physical_parameters(params, model_type)
        
        # Parameter correlations
        try:
            param_correlations = lmfit_result.covar
        except Exception:
            param_correlations = np.array([])
        
        # Confidence intervals (if available)
        confidence_intervals = {}
        try:
            # Suppress lmfit confidence interval warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", 
                                      message=".*rel_change.*", 
                                      category=UserWarning,
                                      module="lmfit.confidence")
                warnings.filterwarnings("ignore", 
                                      message=".*prob.*", 
                                      category=UserWarning,
                                      module="lmfit.confidence")
                
                # Calculate confidence intervals with improved parameters
                ci = lmfit_result.conf_interval(
                    maxiter=200,        # Increase max iterations
                    prob_func=None,     # Use default probability function
                    verbose=False       # Reduce verbosity
                )
                
                for param_name, intervals in ci.items():
                    if len(intervals) >= 2:
                        # Extract 95% confidence interval
                        confidence_intervals[param_name] = (intervals[0][1], intervals[-1][1])
        except Exception:
            confidence_intervals = {}
        
        # Prediction intervals (simplified)
        prediction_intervals = None
        if self.current_err is not None:
            std_residual = np.std(residuals)
            prediction_intervals = (
                fitted_curve - 2 * std_residual,
                fitted_curve + 2 * std_residual
            )
        
        return JosephsonFitResult(
            model_name=MODEL_INFO[model_number_map[model_type]]['name'],
            model_type=model_type,
            lmfit_result=lmfit_result,
            phi_ext=self.phi_ext,
            current=self.current,
            current_err=self.current_err,
            params=params,
            param_errors=param_errors,
            param_bounds=param_bounds,
            chi_square=lmfit_result.chisqr,
            reduced_chi_square=lmfit_result.redchi,
            r_squared=r_squared,
            aic=ic_results['aic'],
            bic=ic_results['bic'],
            aicc=ic_results['aicc'],
            fit_quality=fit_quality,
            parameter_correlations=param_correlations,
            confidence_intervals=confidence_intervals,
            physical_interpretation=physical_interpretation,
            frequency_analysis=self.frequency_analysis or {},
            fitted_curve=fitted_curve,
            residuals=residuals,
            prediction_intervals=prediction_intervals
        )
    
    def fit_all_models(self, **fit_kwargs) -> Dict[str, JosephsonFitResult]:
        """
        Fit all three Josephson models and return results.
        
        Parameters:
        -----------
        **fit_kwargs : dict
            Additional fitting parameters
            
        Returns:
        --------
        dict
            Dictionary with results for all three models
        """
        models = ['model1', 'model2', 'model3']
        results = {}
        
        for model_type in models:
            try:
                result = self.fit_model(model_type, **fit_kwargs)
                results[model_type] = result
                print(f"✓ {model_type.capitalize()} fit completed successfully")
            except Exception as e:
                print(f"✗ {model_type.capitalize()} fit failed: {str(e)}")
                results[model_type] = None
        
        return results
    
    def compare_models(self, results: Optional[Dict[str, JosephsonFitResult]] = None) -> Dict[str, Any]:
        """
        Comprehensive comparison of all fitted models.
        
        Parameters:
        -----------
        results : dict, optional
            Model results (if None, uses stored results)
            
        Returns:
        --------
        dict
            Comprehensive model comparison
        """
        if results is None:
            results = self.results
        
        # Filter out failed fits
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) == 0:
            raise ValueError("No valid fit results available for comparison")
        
        # Extract comparison metrics
        comparison = {
            'model_summary': {},
            'statistical_tests': {},
            'information_criteria': {},
            'model_selection': {}
        }
        
        # Model summary
        for model_type, result in valid_results.items():
            comparison['model_summary'][model_type] = {
                'name': result.model_name,
                'n_params': len(result.params),
                'chi_square': result.chi_square,
                'reduced_chi_square': result.reduced_chi_square,
                'r_squared': result.r_squared,
                'aic': result.aic,
                'bic': result.bic,
                'quality_rating': result.fit_quality['quality_rating'],
                'critical_current_uA': result.critical_current_uA,
                'transparency': result.transparency
            }
        
        # Information criteria comparison
        aic_values = [result.aic for result in valid_results.values()]
        bic_values = [result.bic for result in valid_results.values()]
        model_names = list(valid_results.keys())
        
        # Calculate Akaike weights
        akaike_weights = calculate_akaike_weights(aic_values)
        
        comparison['information_criteria'] = {
            'aic_values': dict(zip(model_names, aic_values)),
            'bic_values': dict(zip(model_names, bic_values)),
            'akaike_weights': dict(zip(model_names, akaike_weights)),
            'best_aic': model_names[np.argmin(aic_values)],
            'best_bic': model_names[np.argmin(bic_values)]
        }
        
        # F-tests for nested models
        f_tests = {}
        model_types = ['model1', 'model2', 'model3']
        for i in range(len(model_types)):
            for j in range(i+1, len(model_types)):
                model1, model2 = model_types[i], model_types[j]
                if model1 in valid_results and model2 in valid_results:
                    f_test_result = perform_f_test(
                        valid_results[model1].lmfit_result,
                        valid_results[model2].lmfit_result
                    )
                    f_tests[f'{model1}_vs_{model2}'] = f_test_result
        
        comparison['statistical_tests']['f_tests'] = f_tests
        
        # Model selection recommendation
        if len(valid_results) >= 2:
            # Use multiple criteria for recommendation
            best_aic = model_names[np.argmin(aic_values)]
            best_bic = model_names[np.argmin(bic_values)]
            best_weight = model_names[np.argmax(akaike_weights)]
            
            # Quality-weighted recommendation
            quality_scores = []
            for model_type in model_names:
                result = valid_results[model_type]
                quality_score = result.fit_quality['quality_percentage']
                
                # Penalize overfitting (too many parameters for improvement)
                aic_penalty = (result.aic - min(aic_values)) / 10.0
                final_score = quality_score - aic_penalty
                quality_scores.append(final_score)
            
            recommended_model = model_names[np.argmax(quality_scores)]
            
            # Determine confidence in recommendation
            max_weight = max(akaike_weights)
            if max_weight > 0.9:
                confidence = "Very High"
            elif max_weight > 0.7:
                confidence = "High"
            elif max_weight > 0.5:
                confidence = "Moderate"
            else:
                confidence = "Low"
            
            comparison['model_selection'] = {
                'recommended_model': recommended_model,
                'confidence': confidence,
                'best_aic': best_aic,
                'best_bic': best_bic,
                'best_akaike_weight': best_weight,
                'quality_based_recommendation': recommended_model
            }
        
        return comparison
    
    def select_best_model(self, criterion: str = 'combined') -> JosephsonFitResult:
        """
        Select the best model based on specified criterion.
        
        Parameters:
        -----------
        criterion : str
            Selection criterion ('aic', 'bic', 'quality', 'combined')
            
        Returns:
        --------
        JosephsonFitResult
            Best model result
        """
        if not self.results:
            raise ValueError("No fit results available. Run fit_all_models() first.")
        
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if criterion == 'aic':
            best_model = min(valid_results.keys(), 
                           key=lambda k: valid_results[k].aic)
        elif criterion == 'bic':
            best_model = min(valid_results.keys(), 
                           key=lambda k: valid_results[k].bic)
        elif criterion == 'quality':
            best_model = max(valid_results.keys(), 
                           key=lambda k: valid_results[k].fit_quality['quality_percentage'])
        elif criterion == 'combined':
            comparison = self.compare_models(valid_results)
            best_model = comparison['model_selection']['recommended_model']
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        return valid_results[best_model]
    
    def generate_report(self, include_plots: bool = True) -> str:
        """
        Generate comprehensive analysis report.
        
        Parameters:
        -----------
        include_plots : bool
            Whether to generate and save plots
            
        Returns:
        --------
        str
            Formatted analysis report
        """
        if not self.results:
            return "No fit results available. Run fitting first."
        
        comparison = self.compare_models()
        
        report = []
        report.append("JOSEPHSON JUNCTION TRIPLE MODEL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data summary
        report.append("DATA SUMMARY:")
        report.append(f"  Number of data points: {len(self.phi_ext)}")
        report.append(f"  Φ_ext range: {np.min(self.phi_ext):.3f} to {np.max(self.phi_ext):.3f}")
        report.append(f"  Current range: {np.min(self.current)*1e6:.2f} to {np.max(self.current)*1e6:.2f} µA")
        report.append(f"  Uncertainties provided: {'Yes' if self.current_err is not None else 'No'}")
        report.append("")
        
        # Frequency analysis
        if self.frequency_analysis:
            freq = self.frequency_analysis
            report.append("FREQUENCY ANALYSIS:")
            report.append(f"  Fundamental frequency: {freq['fundamental_frequency']:.4f}")
            report.append(f"  Signal-to-noise ratio: {freq['snr']:.2f}")
            if freq['harmonics']:
                report.append("  Detected harmonics:")
                for n, harmonic in freq['harmonics'].items():
                    report.append(f"    {n}nd harmonic: {harmonic['relative_power']:.3f} relative power")
            report.append("")
        
        # Model comparison
        report.append("MODEL COMPARISON:")
        report.append("-" * 30)
        for model_type, summary in comparison['model_summary'].items():
            report.append(f"{summary['name']}:")
            report.append(f"  Critical current: {summary['critical_current_uA']:.2f} µA")
            report.append(f"  Transparency: {summary['transparency']:.3f}")
            report.append(f"  Reduced χ²: {summary['reduced_chi_square']:.4f}")
            report.append(f"  R²: {summary['r_squared']:.4f}")
            report.append(f"  AIC: {summary['aic']:.2f}")
            report.append(f"  BIC: {summary['bic']:.2f}")
            report.append(f"  Quality: {summary['quality_rating']}")
            report.append("")
        
        # Model selection
        selection = comparison['model_selection']
        report.append("RECOMMENDED MODEL:")
        report.append(f"  Best model: {selection['recommended_model']}")
        report.append(f"  Confidence: {selection['confidence']}")
        report.append("  Based on: Combined statistical criteria")
        report.append("")
        
        # Statistical tests
        if 'f_tests' in comparison['statistical_tests']:
            report.append("STATISTICAL TESTS:")
            for test_name, f_result in comparison['statistical_tests']['f_tests'].items():
                models = test_name.replace('_vs_', ' vs ')
                report.append(f"  F-test ({models}):")
                report.append(f"    F-statistic: {f_result['f_statistic']:.3f}")
                report.append(f"    p-value: {f_result['p_value']:.6f}")
                report.append(f"    Significant: {'Yes' if f_result['significant'] else 'No'}")
            report.append("")
        
        # Physical interpretation for best model
        best_result = self.select_best_model()
        interp = best_result.physical_interpretation
        report.append("PHYSICAL INTERPRETATION (Best Model):")
        report.append(f"  Critical current: {interp['critical_current']['value_uA']:.2f} µA")
        report.append(f"  Current regime: {interp['critical_current']['regime']}")
        report.append(f"  Junction transparency: {interp['transparency']['value']:.3f}")
        report.append(f"  Transport regime: {interp['transparency']['regime']}")
        if 'josephson_energy' in interp:
            report.append(f"  Josephson energy: {interp['josephson_energy']['value_GHz']:.2f} GHz")
        report.append(f"  Background dominance: {interp['background']['dominance']}")
        report.append("")
        
        return '\n'.join(report)


# Convenience functions for direct use
def fit_josephson_model1(phi_ext: np.ndarray, current: np.ndarray, 
                        current_err: Optional[np.ndarray] = None, **kwargs) -> JosephsonFitResult:
    """Fit Model 1 (standard Josephson) directly."""
    fitter = JosephsonTripleFitter(phi_ext, current, current_err)
    return fitter.fit_model('model1', **kwargs)


def fit_josephson_model2(phi_ext: np.ndarray, current: np.ndarray, 
                        current_err: Optional[np.ndarray] = None, **kwargs) -> JosephsonFitResult:
    """Fit Model 2 (second-order Josephson) directly."""
    fitter = JosephsonTripleFitter(phi_ext, current, current_err)
    return fitter.fit_model('model2', **kwargs)


def fit_josephson_model3(phi_ext: np.ndarray, current: np.ndarray, 
                        current_err: Optional[np.ndarray] = None, **kwargs) -> JosephsonFitResult:
    """Fit Model 3 (third-order Josephson) directly."""
    fitter = JosephsonTripleFitter(phi_ext, current, current_err)
    return fitter.fit_model('model3', **kwargs)


def fit_all_josephson_models(phi_ext: np.ndarray, current: np.ndarray, 
                           current_err: Optional[np.ndarray] = None, **kwargs) -> Dict[str, JosephsonFitResult]:
    """Fit all three Josephson models and return results."""
    fitter = JosephsonTripleFitter(phi_ext, current, current_err)
    return fitter.fit_all_models(**kwargs)


def select_best_model(phi_ext: np.ndarray, current: np.ndarray, 
                     current_err: Optional[np.ndarray] = None, 
                     criterion: str = 'combined', **kwargs) -> JosephsonFitResult:
    """Automatically select best Josephson model."""
    fitter = JosephsonTripleFitter(phi_ext, current, current_err)
    fitter.fit_all_models(**kwargs)
    return fitter.select_best_model(criterion)


def compare_all_models(phi_ext: np.ndarray, current: np.ndarray, 
                      current_err: Optional[np.ndarray] = None, **kwargs) -> Dict[str, Any]:
    """Compare all three Josephson models."""
    fitter = JosephsonTripleFitter(phi_ext, current, current_err)
    fitter.fit_all_models(**kwargs)
    return fitter.compare_models()
