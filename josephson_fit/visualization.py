"""
Comprehensive visualization tools for Josephson junction fitting analysis.

This module provides plotting functions for current-phase relationship data,
fitting results, model comparisons, and advanced analysis visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from .fitter import JosephsonFitResult

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
except Exception:
    plt.style.use('default')


@dataclass
class PlotConfig:
    """Configuration for plot appearance and style."""
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 100
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.7


def plot_current_phase_data(
    phase: np.ndarray,
    current: np.ndarray,
    title: str = "Current-Phase Relationship",
    xlabel: str = "Phase (rad)",
    ylabel: str = "Current (μA)",
    config: Optional[PlotConfig] = None,
    ax: Optional[Axes] = None,
    show_grid: bool = True,
    **kwargs
) -> Figure:
    """
    Plot current-phase relationship data.
    
    Parameters
    ----------
    phase : np.ndarray
        Phase values in radians
    current : np.ndarray
        Current values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    config : PlotConfig, optional
        Plot configuration
    ax : Axes, optional
        Existing axes to plot on
    show_grid : bool
        Whether to show grid
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.get_figure()
    
    # Plot data
    ax.scatter(phase, current, 
              s=config.marker_size**2, 
              alpha=config.alpha,
              label='Data',
              **kwargs)
    
    ax.set_xlabel(xlabel, fontsize=config.label_size)
    ax.set_ylabel(ylabel, fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size)
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    return fig


def plot_fit_result(
    result: JosephsonFitResult,
    phase_range: Optional[Tuple[float, float]] = None,
    n_points: int = 1000,
    config: Optional[PlotConfig] = None,
    show_residuals: bool = True,
    **kwargs
) -> Figure:
    """
    Plot fitting results with data, model, and residuals.
    
    Parameters
    ----------
    result : JosephsonFitResult
        Fitting result to plot
    phase_range : tuple, optional
        Phase range for model curve (min, max)
    n_points : int
        Number of points for model curve
    config : PlotConfig, optional
        Plot configuration
    show_residuals : bool
        Whether to show residuals subplot
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    # Determine phase range
    if phase_range is None:
        phase_range = (result.phi_ext.min(), result.phi_ext.max())
    
    # Create figure layout
    if show_residuals:
        fig, (ax_main, ax_residuals) = plt.subplots(2, 1, 
                                                   figsize=(config.figsize[0], config.figsize[1] * 1.2),
                                                   height_ratios=[3, 1])
    else:
        fig, ax_main = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot data
    ax_main.scatter(result.phi_ext, result.current, 
                   s=config.marker_size**2, 
                   alpha=config.alpha,
                   label='Data', 
                   color='blue')
    
    # Generate model curve from fitted parameters
    phase_model = np.linspace(phase_range[0], phase_range[1], n_points)
    
    # Get model prediction (this will need to be implemented based on the actual model structure)
    try:
        from .models import get_model
        model_func = get_model(result.model_name)
        if model_func:
            current_model = model_func(phase_model, **result.params)
            ax_main.plot(phase_model, current_model, 
                        color='red', 
                        linewidth=config.line_width,
                        label=f'{result.model_name} Model')
    except Exception as e:
        print(f"Warning: Could not generate model curve: {e}")
    
    ax_main.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax_main.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax_main.set_title(f'{result.model_name} Model Fit', fontsize=config.title_size)
    ax_main.legend(fontsize=config.legend_size)
    ax_main.grid(True, alpha=0.3)
    
    # Plot residuals if requested
    if show_residuals:
        try:
            # Calculate residuals from fit result
            residuals = result.lmfit_result.residual
            ax_residuals.scatter(result.phi_ext, residuals, 
                               s=config.marker_size**2/2, 
                               alpha=config.alpha,
                               color='green')
            ax_residuals.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax_residuals.set_xlabel('Phase (rad)', fontsize=config.label_size)
            ax_residuals.set_ylabel('Residuals', fontsize=config.label_size)
            ax_residuals.set_title('Residuals', fontsize=config.title_size-2)
            ax_residuals.grid(True, alpha=0.3)
        except Exception as e:
            print(f"Warning: Could not plot residuals: {e}")
    
    plt.tight_layout()
    return fig


def plot_model_comparison(
    results: List[JosephsonFitResult],
    phase_range: Optional[Tuple[float, float]] = None,
    n_points: int = 1000,
    config: Optional[PlotConfig] = None,
    show_aic_weights: bool = True,
    **kwargs
) -> Figure:
    """
    Compare multiple model fits on the same plot.
    
    Parameters
    ----------
    results : List[JosephsonFitResult]
        List of fitting results to compare
    phase_range : tuple, optional
        Phase range for model curves
    n_points : int
        Number of points for model curves
    config : PlotConfig, optional
        Plot configuration
    show_aic_weights : bool
        Whether to show AIC weights in legend
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    if not results:
        raise ValueError("No results provided for comparison")
    
    # Use first result for data
    first_result = results[0]
    
    # Determine phase range
    if phase_range is None:
        phase_range = (first_result.phi_ext.min(), first_result.phi_ext.max())
    
    fig, (ax_main, ax_stats) = plt.subplots(2, 1, 
                                           figsize=(config.figsize[0], config.figsize[1] * 1.2),
                                           height_ratios=[3, 1])
    
    # Plot data once
    ax_main.scatter(first_result.phi_ext, first_result.current, 
                   s=config.marker_size**2, 
                   alpha=config.alpha,
                   label='Data', 
                   color='black', 
                   zorder=10)
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Generate model curves and calculate AIC weights if requested
    phase_model = np.linspace(phase_range[0], phase_range[1], n_points)
    aic_values = [result.aic for result in results]
    
    if show_aic_weights:
        # Calculate AIC weights
        aic_min = min(aic_values)
        delta_aic = [aic - aic_min for aic in aic_values]
        weights = [np.exp(-0.5 * delta) for delta in delta_aic]
        weight_sum = sum(weights)
        aic_weights = [w / weight_sum for w in weights]
    
    # Plot each model
    for i, result in enumerate(results):
        try:
            from .models import get_model
            model_func = get_model(result.model_name)
            if model_func:
                current_model = model_func(phase_model, **result.params)
                
                if show_aic_weights:
                    label = f'{result.model_name} (w={aic_weights[i]:.3f})'
                else:
                    label = result.model_name
                
                ax_main.plot(phase_model, current_model, 
                            color=colors[i % len(colors)], 
                            linewidth=config.line_width,
                            label=label)
        except Exception as e:
            print(f"Warning: Could not plot model {result.model_name}: {e}")
    
    ax_main.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax_main.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax_main.set_title('Model Comparison', fontsize=config.title_size)
    ax_main.legend(fontsize=config.legend_size)
    ax_main.grid(True, alpha=0.3)
    
    # Statistics comparison
    model_names = [result.model_name for result in results]
    r_squared_values = [result.r_squared for result in results]
    
    x_pos = np.arange(len(results))
    
    bars1 = ax_stats.bar(x_pos, r_squared_values, alpha=0.7, color='blue')
    
    ax_stats.set_xlabel('Model', fontsize=config.label_size)
    ax_stats.set_ylabel('R²', fontsize=config.label_size)
    ax_stats.set_xticks(x_pos)
    ax_stats.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars1, r_squared_values):
        height = bar.get_height()
        ax_stats.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_statistics_comparison(
    results: List[JosephsonFitResult],
    config: Optional[PlotConfig] = None,
    **kwargs
) -> Figure:
    """
    Plot comparison of statistical metrics across models.
    
    Parameters
    ----------
    results : List[JosephsonFitResult]
        List of fitting results to compare
    config : PlotConfig, optional
        Plot configuration
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    if not results:
        raise ValueError("No results provided for comparison")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(config.figsize[0] * 1.2, config.figsize[1]))
    
    model_names = [result.model_name for result in results]
    
    # R-squared comparison
    r_squared_values = [result.r_squared for result in results]
    ax1.bar(model_names, r_squared_values, alpha=0.7, color='blue')
    ax1.set_ylabel('R²', fontsize=config.label_size)
    ax1.set_title('R-squared', fontsize=config.title_size)
    ax1.tick_params(axis='x', rotation=45)
    
    # AIC comparison
    aic_values = [result.aic for result in results]
    ax2.bar(model_names, aic_values, alpha=0.7, color='red')
    ax2.set_ylabel('AIC', fontsize=config.label_size)
    ax2.set_title('Akaike Information Criterion', fontsize=config.title_size)
    ax2.tick_params(axis='x', rotation=45)
    
    # BIC comparison
    bic_values = [result.bic for result in results]
    ax3.bar(model_names, bic_values, alpha=0.7, color='green')
    ax3.set_ylabel('BIC', fontsize=config.label_size)
    ax3.set_title('Bayesian Information Criterion', fontsize=config.title_size)
    ax3.tick_params(axis='x', rotation=45)
    
    # Chi-square comparison
    chi_square_values = [result.reduced_chi_square for result in results]
    ax4.bar(model_names, chi_square_values, alpha=0.7, color='orange')
    ax4.set_ylabel('χ²/ν', fontsize=config.label_size)
    ax4.set_title('Reduced Chi-square', fontsize=config.title_size)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig


def plot_parameter_comparison(
    results: List[JosephsonFitResult],
    parameter_name: str,
    config: Optional[PlotConfig] = None,
    **kwargs
) -> Figure:
    """
    Plot parameter values and uncertainties across multiple models.
    
    Parameters
    ----------
    results : List[JosephsonFitResult]
        List of fitting results
    parameter_name : str
        Name of parameter to analyze
    config : PlotConfig, optional
        Plot configuration
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    model_names = []
    values = []
    errors = []
    
    for result in results:
        if parameter_name in result.params:
            model_names.append(result.model_name)
            values.append(result.params[parameter_name])
            if result.param_errors[parameter_name] is not None:
                errors.append(result.param_errors[parameter_name])
            else:
                errors.append(0)
    
    if not values:
        raise ValueError(f"Parameter '{parameter_name}' not found in any results")
    
    x_pos = np.arange(len(model_names))
    
    # Bar plot with error bars
    bars = ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)
    ax.set_xlabel('Model', fontsize=config.label_size)
    ax.set_ylabel(f'{parameter_name}', fontsize=config.label_size)
    ax.set_title(f'Parameter: {parameter_name}', fontsize=config.title_size)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, value, error in zip(bars, values, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + error + max(values)*0.01,
                f'{value:.3f}±{error:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig


def create_comprehensive_report(
    results: List[JosephsonFitResult],
    output_file: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    **kwargs
) -> Figure:
    """
    Create a comprehensive analysis report with multiple subplots.
    
    Parameters
    ----------
    results : List[JosephsonFitResult]
        List of fitting results
    output_file : str, optional
        File path to save the report
    config : PlotConfig, optional
        Plot configuration
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Model comparison
    ax1 = fig.add_subplot(gs[0, :])
    _plot_comparison_subplot(results, ax1, config)
    
    # 2. Best fit detailed view
    ax2 = fig.add_subplot(gs[1, 0])
    best_result = min(results, key=lambda r: r.aic)
    _plot_detailed_fit(best_result, ax2, config)
    
    # 3. Statistics comparison
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_statistics_subplot(results, ax3, config)
    
    # 4. Model selection criteria
    ax4 = fig.add_subplot(gs[2, :])
    _plot_model_selection(results, ax4, config)
    
    plt.suptitle('Josephson Junction Fitting - Comprehensive Analysis Report', 
                fontsize=config.title_size + 2, y=0.98)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_comparison_subplot(results: List[JosephsonFitResult], ax: Axes, config: PlotConfig):
    """Helper function for model comparison subplot."""
    first_result = results[0]
    ax.scatter(first_result.phi_ext, first_result.current, 
              s=config.marker_size**2, alpha=config.alpha, 
              label='Data', color='black', zorder=10)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    phase_model = np.linspace(first_result.phi_ext.min(), first_result.phi_ext.max(), 1000)
    
    for i, result in enumerate(results):
        try:
            from .models import get_model
            model_func = get_model(result.model_name)
            if model_func:
                current_model = model_func(phase_model, **result.params)
                ax.plot(phase_model, current_model, color=colors[i % len(colors)], 
                       linewidth=config.line_width, label=result.model_name)
        except Exception as e:
            print(f"Warning: Could not plot model {result.model_name}: {e}")
    
    ax.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax.set_title('Model Comparison', fontsize=config.title_size)
    ax.legend(fontsize=config.legend_size)
    ax.grid(True, alpha=0.3)


def _plot_detailed_fit(result: JosephsonFitResult, ax: Axes, config: PlotConfig):
    """Helper function for detailed fit subplot."""
    ax.scatter(result.phi_ext, result.current, s=config.marker_size**2, 
              alpha=config.alpha, label='Data', color='blue')
    
    try:
        from .models import get_model
        model_func = get_model(result.model_name)
        if model_func:
            current_fit = model_func(result.phi_ext, **result.params)
            ax.plot(result.phi_ext, current_fit, color='red', 
                   linewidth=config.line_width, label='Best Fit')
    except Exception as e:
        print(f"Warning: Could not plot fit for {result.model_name}: {e}")
    
    ax.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax.set_title(f'Best Model: {result.model_name}', fontsize=config.title_size)
    ax.legend(fontsize=config.legend_size)
    ax.grid(True, alpha=0.3)


def _plot_statistics_subplot(results: List[JosephsonFitResult], ax: Axes, config: PlotConfig):
    """Helper function for statistics comparison subplot."""
    model_names = [result.model_name for result in results]
    aic_values = [result.aic for result in results]
    bic_values = [result.bic for result in results]
    
    x_pos = np.arange(len(results))
    width = 0.35
    
    ax.bar(x_pos - width/2, aic_values, width, label='AIC', alpha=0.7)
    ax.bar(x_pos + width/2, bic_values, width, label='BIC', alpha=0.7)
    
    ax.set_xlabel('Model', fontsize=config.label_size)
    ax.set_ylabel('Information Criterion', fontsize=config.label_size)
    ax.set_title('Model Selection Criteria', fontsize=config.title_size)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=config.legend_size)


def _plot_model_selection(results: List[JosephsonFitResult], ax: Axes, config: PlotConfig):
    """Helper function for model selection subplot."""
    # Calculate AIC weights
    aic_values = [result.aic for result in results]
    aic_min = min(aic_values)
    delta_aic = [aic - aic_min for aic in aic_values]
    weights = [np.exp(-0.5 * delta) for delta in delta_aic]
    weight_sum = sum(weights)
    aic_weights = [w / weight_sum for w in weights]
    
    model_names = [result.model_name for result in results]
    
    # Create pie chart of AIC weights
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    wedges, texts, autotexts = ax.pie(aic_weights, labels=model_names, autopct='%1.1f%%',
                                     colors=colors[:len(results)], startangle=90)
    
    ax.set_title('Model Selection (AIC Weights)', fontsize=config.title_size)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')


# Convenience function for quick plotting
def quick_plot(phase: np.ndarray, current: np.ndarray, 
               result: Optional[JosephsonFitResult] = None,
               **kwargs) -> Figure:
    """
    Quick plotting function for immediate visualization.
    
    Parameters
    ----------
    phase : np.ndarray
        Phase values
    current : np.ndarray
        Current values
    result : JosephsonFitResult, optional
        Fitting result to overlay
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    Figure
        The created figure
    """
    if result is None:
        return plot_current_phase_data(phase, current, **kwargs)
    else:
        return plot_fit_result(result, **kwargs)


@dataclass
class PlotConfig:
    """Configuration for plot appearance and style."""
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 100
    style: str = 'seaborn-v0_8-darkgrid'
    color_palette: str = 'husl'
    font_size: int = 12
    title_size: int = 14
    label_size: int = 12
    legend_size: int = 10
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.7


def plot_current_phase_data(
    phase: np.ndarray,
    current: np.ndarray,
    title: str = "Current-Phase Relationship",
    xlabel: str = "Phase (rad)",
    ylabel: str = "Current (μA)",
    config: Optional[PlotConfig] = None,
    ax: Optional[plt.Axes] = None,
    show_grid: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot current-phase relationship data.
    
    Parameters
    ----------
    phase : np.ndarray
        Phase values in radians
    current : np.ndarray
        Current values
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    config : PlotConfig, optional
        Plot configuration
    ax : plt.Axes, optional
        Existing axes to plot on
    show_grid : bool
        Whether to show grid
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    else:
        fig = ax.get_figure()
    
    # Plot data
    ax.scatter(phase, current, 
              s=config.marker_size**2, 
              alpha=config.alpha,
              label='Data',
              **kwargs)
    
    ax.set_xlabel(xlabel, fontsize=config.label_size)
    ax.set_ylabel(ylabel, fontsize=config.label_size)
    ax.set_title(title, fontsize=config.title_size)
    
    if show_grid:
        ax.grid(True, alpha=0.3)
    
    return fig


def plot_fit_result(
    result: JosephsonFitResult,
    phase_range: Optional[Tuple[float, float]] = None,
    n_points: int = 1000,
    config: Optional[PlotConfig] = None,
    show_residuals: bool = True,
    show_confidence: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Plot fitting results with data, model, residuals, and confidence intervals.
    
    Parameters
    ----------
    result : JosephsonFitResult
        Fitting result to plot
    phase_range : tuple, optional
        Phase range for model curve (min, max)
    n_points : int
        Number of points for model curve
    config : PlotConfig, optional
        Plot configuration
    show_residuals : bool
        Whether to show residuals subplot
    show_confidence : bool
        Whether to show confidence intervals
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    # Determine phase range
    if phase_range is None:
        phase_range = (result.phase.min(), result.phase.max())
    
    # Create figure layout
    if show_residuals:
        fig = plt.figure(figsize=(config.figsize[0], config.figsize[1] * 1.2))
        gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
        ax_main = fig.add_subplot(gs[0])
        ax_residuals = fig.add_subplot(gs[1])
        ax_info = fig.add_subplot(gs[2])
        ax_info.axis('off')
    else:
        fig, ax_main = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Plot data
    ax_main.scatter(result.phase, result.current, 
                   s=config.marker_size**2, 
                   alpha=config.alpha,
                   label='Data', 
                   color='blue')
    
    # Generate model curve
    phase_model = np.linspace(phase_range[0], phase_range[1], n_points)
    current_model = result.model.eval(result.params, phase=phase_model)
    
    # Plot model
    ax_main.plot(phase_model, current_model, 
                color='red', 
                linewidth=config.line_width,
                label=f'{result.model_name} Model')
    
    # Plot confidence intervals if available and requested
    if show_confidence and hasattr(result, 'confidence_interval') and result.confidence_interval is not None:
        current_lower = result.model.eval(result.confidence_interval['lower'], phase=phase_model)
        current_upper = result.model.eval(result.confidence_interval['upper'], phase=phase_model)
        ax_main.fill_between(phase_model, current_lower, current_upper,
                           alpha=0.2, color='red', label='95% Confidence')
    
    ax_main.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax_main.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax_main.set_title(f'{result.model_name} Model Fit', fontsize=config.title_size)
    ax_main.legend(fontsize=config.legend_size)
    ax_main.grid(True, alpha=0.3)
    
    # Plot residuals if requested
    if show_residuals:
        residuals = result.current - result.best_fit
        ax_residuals.scatter(result.phase, residuals, 
                           s=config.marker_size**2/2, 
                           alpha=config.alpha,
                           color='green')
        ax_residuals.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax_residuals.set_xlabel('Phase (rad)', fontsize=config.label_size)
        ax_residuals.set_ylabel('Residuals', fontsize=config.label_size)
        ax_residuals.set_title('Residuals', fontsize=config.title_size-2)
        ax_residuals.grid(True, alpha=0.3)
        
        # Add fit statistics
        info_text = f"""
Fit Statistics:
R² = {result.r_squared:.4f}
RMSE = {result.rmse:.4f}
χ²/ν = {result.reduced_chi_square:.4f}
AIC = {result.aic:.2f}
BIC = {result.bic:.2f}
        """
        ax_info.text(0.05, 0.95, info_text, transform=ax_info.transAxes,
                    fontsize=config.legend_size, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        # Add parameter values
        param_text = "Parameters:\n"
        for name, value in result.params.valuesdict().items():
            param_text += f"{name} = {value:.4f}\n"
        
        ax_info.text(0.55, 0.95, param_text, transform=ax_info.transAxes,
                    fontsize=config.legend_size, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    return fig


def plot_model_comparison(
    results: List[JosephsonFitResult],
    phase_range: Optional[Tuple[float, float]] = None,
    n_points: int = 1000,
    config: Optional[PlotConfig] = None,
    show_aic_weights: bool = True,
    **kwargs
) -> plt.Figure:
    """
    Compare multiple model fits on the same plot.
    
    Parameters
    ----------
    results : List[JosephsonFitResult]
        List of fitting results to compare
    phase_range : tuple, optional
        Phase range for model curves
    n_points : int
        Number of points for model curves
    config : PlotConfig, optional
        Plot configuration
    show_aic_weights : bool
        Whether to show AIC weights in legend
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    if not results:
        raise ValueError("No results provided for comparison")
    
    # Use first result for data
    first_result = results[0]
    
    # Determine phase range
    if phase_range is None:
        phase_range = (first_result.phase.min(), first_result.phase.max())
    
    fig, (ax_main, ax_stats) = plt.subplots(2, 1, 
                                           figsize=(config.figsize[0], config.figsize[1] * 1.2),
                                           height_ratios=[3, 1])
    
    # Plot data once
    ax_main.scatter(first_result.phase, first_result.current, 
                   s=config.marker_size**2, 
                   alpha=config.alpha,
                   label='Data', 
                   color='black', 
                   zorder=10)
    
    # Colors for different models
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    # Generate model curves and calculate AIC weights if requested
    phase_model = np.linspace(phase_range[0], phase_range[1], n_points)
    aic_values = [result.aic for result in results]
    
    if show_aic_weights:
        # Calculate AIC weights
        aic_min = min(aic_values)
        delta_aic = [aic - aic_min for aic in aic_values]
        weights = [np.exp(-0.5 * delta) for delta in delta_aic]
        weight_sum = sum(weights)
        aic_weights = [w / weight_sum for w in weights]
    
    # Plot each model
    for i, result in enumerate(results):
        current_model = result.model.eval(result.params, phase=phase_model)
        
        if show_aic_weights:
            label = f'{result.model_name} (w={aic_weights[i]:.3f})'
        else:
            label = result.model_name
        
        ax_main.plot(phase_model, current_model, 
                    color=colors[i], 
                    linewidth=config.line_width,
                    label=label)
    
    ax_main.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax_main.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax_main.set_title('Model Comparison', fontsize=config.title_size)
    ax_main.legend(fontsize=config.legend_size)
    ax_main.grid(True, alpha=0.3)
    
    # Statistics comparison
    model_names = [result.model_name for result in results]
    r_squared_values = [result.r_squared for result in results]
    rmse_values = [result.rmse for result in results]
    
    x_pos = np.arange(len(results))
    
    # Create twin axis for RMSE
    ax_stats2 = ax_stats.twinx()
    
    bars1 = ax_stats.bar(x_pos - 0.2, r_squared_values, 0.4, 
                        label='R²', alpha=0.7, color='blue')
    bars2 = ax_stats2.bar(x_pos + 0.2, rmse_values, 0.4, 
                         label='RMSE', alpha=0.7, color='red')
    
    ax_stats.set_xlabel('Model', fontsize=config.label_size)
    ax_stats.set_ylabel('R²', fontsize=config.label_size, color='blue')
    ax_stats2.set_ylabel('RMSE', fontsize=config.label_size, color='red')
    ax_stats.set_xticks(x_pos)
    ax_stats.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars1, r_squared_values):
        height = bar.get_height()
        ax_stats.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    for bar, value in zip(bars2, rmse_values):
        height = bar.get_height()
        ax_stats2.text(bar.get_x() + bar.get_width()/2., height + max(rmse_values)*0.01,
                      f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_harmonic_analysis(
    phase: np.ndarray,
    current: np.ndarray,
    frequencies: Optional[np.ndarray] = None,
    power_spectrum: Optional[np.ndarray] = None,
    significant_harmonics: Optional[List[int]] = None,
    config: Optional[PlotConfig] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot harmonic analysis of current-phase data.
    
    Parameters
    ----------
    phase : np.ndarray
        Phase values
    current : np.ndarray
        Current values
    frequencies : np.ndarray, optional
        Frequency values for power spectrum
    power_spectrum : np.ndarray, optional
        Power spectrum values
    significant_harmonics : List[int], optional
        List of significant harmonic numbers
    config : PlotConfig, optional
        Plot configuration
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    # If no frequency analysis provided, compute it
    if frequencies is None or power_spectrum is None:
        from .utils import lombscargle_frequency_detection
        result = lombscargle_frequency_detection(phase, current, return_spectrum=True)
        frequencies = result.get('frequencies', np.array([]))
        power_spectrum = result.get('power_spectrum', np.array([]))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                 figsize=(config.figsize[0] * 1.2, config.figsize[1]))
    
    # Original data
    ax1.scatter(phase, current, s=config.marker_size**2, alpha=config.alpha)
    ax1.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax1.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax1.set_title('Original Data', fontsize=config.title_size)
    ax1.grid(True, alpha=0.3)
    
    # Power spectrum
    if len(frequencies) > 0 and len(power_spectrum) > 0:
        ax2.plot(frequencies, power_spectrum, linewidth=config.line_width)
        ax2.set_xlabel('Frequency', fontsize=config.label_size)
        ax2.set_ylabel('Power', fontsize=config.label_size)
        ax2.set_title('Lomb-Scargle Periodogram', fontsize=config.title_size)
        ax2.grid(True, alpha=0.3)
        
        # Mark significant harmonics
        if significant_harmonics:
            for harmonic in significant_harmonics:
                if harmonic < len(frequencies):
                    ax2.axvline(frequencies[harmonic], color='red', linestyle='--', alpha=0.7)
                    ax2.text(frequencies[harmonic], max(power_spectrum) * 0.9, 
                           f'H{harmonic}', rotation=90, ha='right')
    
    # Phase-folded data (if fundamental frequency detected)
    if len(frequencies) > 0:
        try:
            # Assume fundamental frequency is the first significant peak
            fundamental_freq = frequencies[np.argmax(power_spectrum)]
            if fundamental_freq > 0:
                phase_folded = np.mod(phase * fundamental_freq, 2 * np.pi)
                sorted_indices = np.argsort(phase_folded)
                ax3.plot(phase_folded[sorted_indices], current[sorted_indices], 
                        'o-', markersize=config.marker_size/2, linewidth=1)
                ax3.set_xlabel('Phase (folded)', fontsize=config.label_size)
                ax3.set_ylabel('Current (μA)', fontsize=config.label_size)
                ax3.set_title('Phase-Folded Data', fontsize=config.title_size)
                ax3.grid(True, alpha=0.3)
        except:
            ax3.text(0.5, 0.5, 'Phase folding not available', 
                    transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('Phase-Folded Data', fontsize=config.title_size)
    
    # Harmonic content summary
    ax4.axis('off')
    if significant_harmonics:
        harmonic_text = "Significant Harmonics:\n"
        for i, harmonic in enumerate(significant_harmonics[:5]):  # Show first 5
            harmonic_text += f"H{harmonic}: f = {frequencies[harmonic]:.3f}\n"
        
        ax4.text(0.1, 0.9, harmonic_text, transform=ax4.transAxes,
                fontsize=config.legend_size, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    else:
        ax4.text(0.5, 0.5, 'No significant harmonics detected', 
                transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    return fig


def plot_parameter_correlations(
    result: JosephsonFitResult,
    config: Optional[PlotConfig] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot parameter correlation matrix from fit result.
    
    Parameters
    ----------
    result : JosephsonFitResult
        Fitting result with correlation matrix
    config : PlotConfig, optional
        Plot configuration
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    if not hasattr(result, 'correlation_matrix') or result.correlation_matrix is None:
        raise ValueError("No correlation matrix available in fit result")
    
    fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
    
    # Get parameter names
    param_names = list(result.params.valuesdict().keys())
    
    # Create heatmap
    im = ax.imshow(result.correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_xticklabels(param_names)
    ax.set_yticklabels(param_names)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add correlation values as text
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{result.correlation_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black" if abs(result.correlation_matrix[i, j]) < 0.5 else "white")
    
    ax.set_title("Parameter Correlation Matrix", fontsize=config.title_size)
    fig.colorbar(im, ax=ax, label='Correlation Coefficient')
    
    plt.tight_layout()
    return fig


def plot_uncertainty_analysis(
    results: List[JosephsonFitResult],
    parameter_name: str,
    config: Optional[PlotConfig] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot parameter uncertainty analysis across multiple models.
    
    Parameters
    ----------
    results : List[JosephsonFitResult]
        List of fitting results
    parameter_name : str
        Name of parameter to analyze
    config : PlotConfig, optional
        Plot configuration
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(config.figsize[0] * 1.2, config.figsize[1]))
    
    model_names = []
    values = []
    errors = []
    
    for result in results:
        if parameter_name in result.params.valuesdict():
            model_names.append(result.model_name)
            values.append(result.params[parameter_name].value)
            if result.params[parameter_name].stderr is not None:
                errors.append(result.params[parameter_name].stderr)
            else:
                errors.append(0)
    
    if not values:
        raise ValueError(f"Parameter '{parameter_name}' not found in any results")
    
    x_pos = np.arange(len(model_names))
    
    # Bar plot with error bars
    bars = ax1.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7)
    ax1.set_xlabel('Model', fontsize=config.label_size)
    ax1.set_ylabel(f'{parameter_name}', fontsize=config.label_size)
    ax1.set_title(f'Parameter: {parameter_name}', fontsize=config.title_size)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Add value labels
    for bar, value, error in zip(bars, values, errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + error + max(values)*0.01,
                f'{value:.3f}±{error:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Box plot for uncertainty distribution
    if len(set(errors)) > 1:  # Only if we have varying uncertainties
        uncertainty_data = [np.random.normal(val, err, 1000) for val, err in zip(values, errors)]
        ax2.boxplot(uncertainty_data, labels=model_names)
        ax2.set_ylabel(f'{parameter_name}', fontsize=config.label_size)
        ax2.set_title('Uncertainty Distribution', fontsize=config.title_size)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'Insufficient uncertainty data', 
                transform=ax2.transAxes, ha='center', va='center')
        ax2.set_title('Uncertainty Distribution', fontsize=config.title_size)
    
    plt.tight_layout()
    return fig


def create_comprehensive_report(
    results: List[JosephsonFitResult],
    phase: np.ndarray,
    current: np.ndarray,
    output_file: Optional[str] = None,
    config: Optional[PlotConfig] = None,
    **kwargs
) -> plt.Figure:
    """
    Create a comprehensive analysis report with multiple subplots.
    
    Parameters
    ----------
    results : List[JosephsonFitResult]
        List of fitting results
    phase : np.ndarray
        Phase data
    current : np.ndarray
        Current data
    output_file : str, optional
        File path to save the report
    config : PlotConfig, optional
        Plot configuration
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if config is None:
        config = PlotConfig()
    
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Model comparison
    ax1 = fig.add_subplot(gs[0, :])
    _plot_comparison_subplot(results, ax1, config)
    
    # 2. Best fit detailed view
    ax2 = fig.add_subplot(gs[1, 0])
    best_result = min(results, key=lambda r: r.aic)
    _plot_detailed_fit(best_result, ax2, config)
    
    # 3. Residuals analysis
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_residuals_analysis(best_result, ax3, config)
    
    # 4. Statistics comparison
    ax4 = fig.add_subplot(gs[2, 0])
    _plot_statistics_comparison(results, ax4, config)
    
    # 5. Parameter comparison (if common parameters exist)
    ax5 = fig.add_subplot(gs[2, 1])
    _plot_parameter_comparison(results, ax5, config)
    
    # 6. Model selection criteria
    ax6 = fig.add_subplot(gs[3, :])
    _plot_model_selection(results, ax6, config)
    
    plt.suptitle('Josephson Junction Fitting - Comprehensive Analysis Report', 
                fontsize=config.title_size + 2, y=0.98)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def _plot_comparison_subplot(results: List[JosephsonFitResult], ax: plt.Axes, config: PlotConfig):
    """Helper function for model comparison subplot."""
    first_result = results[0]
    ax.scatter(first_result.phase, first_result.current, 
              s=config.marker_size**2, alpha=config.alpha, 
              label='Data', color='black', zorder=10)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    phase_model = np.linspace(first_result.phase.min(), first_result.phase.max(), 1000)
    
    for i, result in enumerate(results):
        current_model = result.model.eval(result.params, phase=phase_model)
        ax.plot(phase_model, current_model, color=colors[i], 
               linewidth=config.line_width, label=result.model_name)
    
    ax.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax.set_title('Model Comparison', fontsize=config.title_size)
    ax.legend(fontsize=config.legend_size)
    ax.grid(True, alpha=0.3)


def _plot_detailed_fit(result: JosephsonFitResult, ax: plt.Axes, config: PlotConfig):
    """Helper function for detailed fit subplot."""
    ax.scatter(result.phase, result.current, s=config.marker_size**2, 
              alpha=config.alpha, label='Data', color='blue')
    ax.plot(result.phase, result.best_fit, color='red', 
           linewidth=config.line_width, label='Best Fit')
    
    ax.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax.set_ylabel('Current (μA)', fontsize=config.label_size)
    ax.set_title(f'Best Model: {result.model_name}', fontsize=config.title_size)
    ax.legend(fontsize=config.legend_size)
    ax.grid(True, alpha=0.3)


def _plot_residuals_analysis(result: JosephsonFitResult, ax: plt.Axes, config: PlotConfig):
    """Helper function for residuals analysis subplot."""
    residuals = result.current - result.best_fit
    ax.scatter(result.phase, residuals, s=config.marker_size**2, 
              alpha=config.alpha, color='green')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Phase (rad)', fontsize=config.label_size)
    ax.set_ylabel('Residuals', fontsize=config.label_size)
    ax.set_title('Residuals Analysis', fontsize=config.title_size)
    ax.grid(True, alpha=0.3)


def _plot_statistics_comparison(results: List[JosephsonFitResult], ax: plt.Axes, config: PlotConfig):
    """Helper function for statistics comparison subplot."""
    model_names = [result.model_name for result in results]
    aic_values = [result.aic for result in results]
    bic_values = [result.bic for result in results]
    
    x_pos = np.arange(len(results))
    width = 0.35
    
    ax.bar(x_pos - width/2, aic_values, width, label='AIC', alpha=0.7)
    ax.bar(x_pos + width/2, bic_values, width, label='BIC', alpha=0.7)
    
    ax.set_xlabel('Model', fontsize=config.label_size)
    ax.set_ylabel('Information Criterion', fontsize=config.label_size)
    ax.set_title('Model Selection Criteria', fontsize=config.title_size)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=config.legend_size)


def _plot_parameter_comparison(results: List[JosephsonFitResult], ax: plt.Axes, config: PlotConfig):
    """Helper function for parameter comparison subplot."""
    # Find common parameters
    all_params = set()
    for result in results:
        all_params.update(result.params.valuesdict().keys())
    
    common_params = []
    for param in all_params:
        if all(param in result.params.valuesdict() for result in results):
            common_params.append(param)
    
    if common_params:
        # Plot first common parameter
        param_name = common_params[0]
        model_names = [result.model_name for result in results]
        values = [result.params[param_name].value for result in results]
        
        ax.bar(model_names, values, alpha=0.7)
        ax.set_xlabel('Model', fontsize=config.label_size)
        ax.set_ylabel(param_name, fontsize=config.label_size)
        ax.set_title(f'Parameter: {param_name}', fontsize=config.title_size)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No common parameters', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Parameter Comparison', fontsize=config.title_size)


def _plot_model_selection(results: List[JosephsonFitResult], ax: plt.Axes, config: PlotConfig):
    """Helper function for model selection subplot."""
    # Calculate AIC weights
    aic_values = [result.aic for result in results]
    aic_min = min(aic_values)
    delta_aic = [aic - aic_min for aic in aic_values]
    weights = [np.exp(-0.5 * delta) for delta in delta_aic]
    weight_sum = sum(weights)
    aic_weights = [w / weight_sum for w in weights]
    
    model_names = [result.model_name for result in results]
    
    # Create pie chart of AIC weights
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    wedges, texts, autotexts = ax.pie(aic_weights, labels=model_names, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
    
    ax.set_title('Model Selection (AIC Weights)', fontsize=config.title_size)
    
    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')


# Convenience function for quick plotting
def quick_plot(phase: np.ndarray, current: np.ndarray, 
               result: Optional[JosephsonFitResult] = None,
               **kwargs) -> plt.Figure:
    """
    Quick plotting function for immediate visualization.
    
    Parameters
    ----------
    phase : np.ndarray
        Phase values
    current : np.ndarray
        Current values
    result : JosephsonFitResult, optional
        Fitting result to overlay
    **kwargs
        Additional plotting arguments
    
    Returns
    -------
    plt.Figure
        The created figure
    """
    if result is None:
        return plot_current_phase_data(phase, current, **kwargs)
    else:
        return plot_fit_result(result, **kwargs)
