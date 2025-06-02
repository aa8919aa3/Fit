#!/usr/bin/env python3
"""
Enhanced Command Line Interface for Josephson Junction Triple Model Fitting Toolkit

This module provides a comprehensive command-line interface for fitting and analyzing
Josephson junction current-phase relationships using the triple model system.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Optional, Tuple

from josephson_fit import (
    JosephsonTripleFitter,
    fit_all_josephson_models,
    select_best_model,
    compare_all_models,
    generate_synthetic_data,
    get_model
)


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Load experimental data from file.
    
    Parameters:
    -----------
    filepath : str
        Path to data file
        
    Returns:
    --------
    tuple
        (phi_ext, current, current_err) arrays
    """
    try:
        data = np.loadtxt(filepath)
        
        if data.ndim == 1:
            raise ValueError("Data file must have at least 2 columns")
        
        if data.shape[1] < 2:
            raise ValueError("Data file must have at least 2 columns (phi_ext, current)")
        
        phi_ext = data[:, 0]
        current = data[:, 1]
        
        # Check for error column
        current_err = None
        if data.shape[1] >= 3:
            current_err = data[:, 2]
            print(f"✓ Loaded {len(phi_ext)} data points with uncertainties from {filepath}")
        else:
            print(f"✓ Loaded {len(phi_ext)} data points from {filepath}")
        
        return phi_ext, current, current_err
        
    except Exception as e:
        print(f"✗ Error loading data from {filepath}: {e}")
        sys.exit(1)


def save_results(results, output_dir: str, prefix: str = "josephson_fit"):
    """Save fitting results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save numerical results
    results_file = output_path / f"{prefix}_results.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for model_type, result in results.items():
        if result is not None:
            json_results[model_type] = {
                'model_name': result.model_name,
                'params': result.params,
                'param_errors': {k: v for k, v in result.param_errors.items() if v is not None},
                'chi_square': result.chi_square,
                'reduced_chi_square': result.reduced_chi_square,
                'r_squared': result.r_squared,
                'aic': result.aic,
                'bic': result.bic,
                'quality_rating': result.fit_quality['quality_rating'],
                'critical_current_uA': result.critical_current_uA,
                'transparency': result.transparency
            }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"✓ Results saved to {results_file}")


def create_summary_plot(results, phi_ext, current, current_err, output_dir: str, prefix: str = "josephson_fit"):
    """Create and save summary plots."""
    output_path = Path(output_dir)
    
    # Filter successful fits
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("✗ No valid results to plot")
        return
    
    # Create figure with subplots
    n_models = len(valid_results)
    fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
    
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    colors = ['blue', 'red', 'green']
    model_names = ['Model 1', 'Model 2', 'Model 3']
    
    for i, (model_type, result) in enumerate(valid_results.items()):
        color = colors[int(model_type[-1]) - 1]
        model_name = model_names[int(model_type[-1]) - 1]
        
        # Data and fit plot
        ax1 = axes[0, i]
        if current_err is not None:
            ax1.errorbar(phi_ext, current*1e6, yerr=current_err*1e6, 
                        fmt='o', color='gray', alpha=0.6, markersize=3, label='Data')
        else:
            ax1.plot(phi_ext, current*1e6, 'o', color='gray', alpha=0.6, markersize=3, label='Data')
        
        ax1.plot(phi_ext, result.fitted_curve*1e6, '-', color=color, linewidth=2, label=f'{model_name} Fit')
        ax1.set_xlabel('Φ_ext')
        ax1.set_ylabel('Current (µA)')
        ax1.set_title(f'{model_name}\nIc = {result.critical_current_uA:.2f} µA, T = {result.transparency:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        ax2 = axes[1, i]
        residuals_uA = result.residuals * 1e6
        ax2.plot(phi_ext, residuals_uA, 'o', color=color, alpha=0.7, markersize=4)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Φ_ext')
        ax2.set_ylabel('Residuals (µA)')
        ax2.set_title(f'Residuals (σ = {np.std(residuals_uA):.3f} µA)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_path / f"{prefix}_summary.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Summary plot saved to {plot_file}")


def create_comparison_plot(comparison_results, output_dir: str, prefix: str = "josephson_fit"):
    """Create model comparison visualization."""
    output_path = Path(output_dir)
    
    model_summary = comparison_results['model_summary']
    if len(model_summary) < 2:
        print("⚠ Need at least 2 models for comparison plot")
        return
    
    # Extract data for plotting
    models = list(model_summary.keys())
    model_names = [model_summary[m]['name'] for m in models]
    aic_values = [model_summary[m]['aic'] for m in models]
    bic_values = [model_summary[m]['bic'] for m in models]
    r_squared = [model_summary[m]['r_squared'] for m in models]
    redchi = [model_summary[m]['reduced_chi_square'] for m in models]
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # AIC comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(model_names, aic_values, color=['blue', 'red', 'green'][:len(models)])
    ax1.set_ylabel('AIC')
    ax1.set_title('Akaike Information Criterion')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars1, aic_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(aic_values),
                f'{val:.1f}', ha='center', va='bottom')
    
    # BIC comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(model_names, bic_values, color=['blue', 'red', 'green'][:len(models)])
    ax2.set_ylabel('BIC')
    ax2.set_title('Bayesian Information Criterion')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars2, bic_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(bic_values),
                f'{val:.1f}', ha='center', va='bottom')
    
    # R-squared comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(model_names, r_squared, color=['blue', 'red', 'green'][:len(models)])
    ax3.set_ylabel('R²')
    ax3.set_title('Coefficient of Determination')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, 1)
    
    for bar, val in zip(bars3, r_squared):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom')
    
    # Reduced chi-square comparison
    ax4 = axes[1, 1]
    bars4 = ax4.bar(model_names, redchi, color=['blue', 'red', 'green'][:len(models)])
    ax4.set_ylabel('Reduced χ²')
    ax4.set_title('Reduced Chi-Square')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Ideal = 1')
    ax4.legend()
    
    for bar, val in zip(bars4, redchi):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(redchi),
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_file = output_path / f"{prefix}_comparison.png"
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved to {comparison_file}")


def generate_synthetic_data_cmd(args):
    """Generate synthetic data based on command line arguments."""
    print("Generating synthetic Josephson junction data...")
    
    # Create parameter grid
    phi_ext = np.linspace(args.phi_min, args.phi_max, args.n_points)
    
    # Define model parameters
    model_params = {
        'Ic': args.Ic * 1e-6,  # Convert µA to A
        'T': args.T,
        'f': args.f,
        'd': args.d,
        'phi0': args.phi0,
        'k': args.k,
        'r': args.r,
        'C': args.C * 1e-6  # Convert µA to A
    }
    
    # Get model function
    model_func = get_model(f'model{args.model}')
    
    # Generate synthetic data
    current, current_err = generate_synthetic_data(
        phi_ext, model_func, model_params,
        noise_level=args.noise_level,
        noise_type=args.noise_type
    )
    
    # Save to file
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if args.include_errors:
        data = np.column_stack([phi_ext, current, current_err])
        header = "phi_ext current(A) current_err(A)"
    else:
        data = np.column_stack([phi_ext, current])
        header = "phi_ext current(A)"
    
    np.savetxt(output_file, data, header=header, fmt='%.6e')
    
    print(f"✓ Synthetic data saved to {output_file}")
    print(f"  Model type: Model {args.model}")
    print(f"  Data points: {args.n_points}")
    print(f"  Critical current: {args.Ic} µA")
    print(f"  Transparency: {args.T}")
    print(f"  Noise level: {args.noise_level * 100}%")
    print(f"  Noise type: {args.noise_type}")


def fit_data_cmd(args):
    """Perform fitting based on command line arguments."""
    print("Josephson Junction Triple Model Fitting")
    print("=" * 40)
    
    # Load data
    phi_ext, current, current_err = load_data(args.input)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model == 'all':
        print("\nFitting all three models...")
        print("-" * 30)
        
        # Fit all models
        results = fit_all_josephson_models(phi_ext, current, current_err)
        
        # Display results summary
        for model_type, result in results.items():
            if result is not None:
                print(f"\n{result.model_name}:")
                print(f"  Critical current: {result.critical_current_uA:.2f} µA")
                print(f"  Transparency: {result.transparency:.3f}")
                print(f"  Reduced χ²: {result.reduced_chi_square:.4f}")
                print(f"  R²: {result.r_squared:.4f}")
                print(f"  AIC: {result.aic:.2f}")
                print(f"  Quality: {result.fit_quality['quality_rating']}")
            else:
                print(f"\n{model_type}: FAILED")
        
        # Model comparison
        print("\nModel Comparison:")
        print("-" * 20)
        comparison = compare_all_models(phi_ext, current, current_err)
        
        if 'model_selection' in comparison:
            selection = comparison['model_selection']
            print(f"Recommended model: {selection['recommended_model']}")
            print(f"Confidence: {selection['confidence']}")
            
            # Display information criteria
            ic = comparison['information_criteria']
            print(f"Best AIC: {ic['best_aic']}")
            print(f"Best BIC: {ic['best_bic']}")
            
            print("\nAkaike weights:")
            for model, weight in ic['akaike_weights'].items():
                print(f"  {model}: {weight:.3f}")
        
        # Generate comprehensive report
        if args.generate_report:
            fitter = JosephsonTripleFitter(phi_ext, current, current_err)
            fitter.results = results
            report = fitter.generate_report()
            
            report_file = output_dir / f"{args.prefix}_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"\n✓ Comprehensive report saved to {report_file}")
        
        # Save results
        save_results(results, str(output_dir), args.prefix)
        
        # Create plots
        if args.create_plots:
            create_summary_plot(results, phi_ext, current, current_err, str(output_dir), args.prefix)
            create_comparison_plot(comparison, str(output_dir), args.prefix)
    
    else:
        # Fit single model
        model_num = int(args.model)
        model_type = f'model{model_num}'
        
        print(f"\nFitting Model {model_num}...")
        
        fitter = JosephsonTripleFitter(phi_ext, current, current_err)
        result = fitter.fit_model(model_type)
        
        print(f"\n{result.model_name} Results:")
        print(f"  Critical current: {result.critical_current_uA:.2f} µA")
        print(f"  Transparency: {result.transparency:.3f}")
        print(f"  Reduced χ²: {result.reduced_chi_square:.4f}")
        print(f"  R²: {result.r_squared:.4f}")
        print(f"  Quality: {result.fit_quality['quality_rating']}")
        
        # Save single model result
        single_result = {model_type: result}
        save_results(single_result, str(output_dir), args.prefix)
        
        if args.create_plots:
            create_summary_plot(single_result, phi_ext, current, current_err, str(output_dir), args.prefix)


def main():
    """Main command line interface."""
    parser = argparse.ArgumentParser(
        description='Enhanced Josephson Junction Triple Model Fitting Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Fit all models to experimental data
  josephson-fit fit data.txt --model all --output-dir results/
  
  # Fit only Model 2 with plots and report
  josephson-fit fit data.txt --model 2 --create-plots --generate-report
  
  # Generate synthetic Model 3 data
  josephson-fit generate --model 3 --Ic 10.5 --T 0.6 --output synthetic_data.txt
  
  # Quick analysis with automatic model selection
  josephson-fit fit data.txt --model all --create-plots --generate-report
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fitting command
    fit_parser = subparsers.add_parser('fit', help='Fit Josephson models to data')
    fit_parser.add_argument('input', help='Input data file (phi_ext, current, [current_err])')
    fit_parser.add_argument('--model', choices=['1', '2', '3', 'all'], default='all',
                           help='Model to fit (1, 2, 3, or all)')
    fit_parser.add_argument('--output-dir', default='josephson_results',
                           help='Output directory for results')
    fit_parser.add_argument('--prefix', default='josephson_fit',
                           help='Prefix for output files')
    fit_parser.add_argument('--create-plots', action='store_true',
                           help='Generate visualization plots')
    fit_parser.add_argument('--generate-report', action='store_true',
                           help='Generate comprehensive analysis report')
    
    # Data generation command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--model', type=int, choices=[1, 2, 3], default=1,
                           help='Model type for synthetic data')
    gen_parser.add_argument('--output', default='synthetic_data.txt',
                           help='Output file for synthetic data')
    gen_parser.add_argument('--n-points', type=int, default=100,
                           help='Number of data points')
    gen_parser.add_argument('--phi-min', type=float, default=-2.0,
                           help='Minimum phi_ext value')
    gen_parser.add_argument('--phi-max', type=float, default=2.0,
                           help='Maximum phi_ext value')
    gen_parser.add_argument('--Ic', type=float, default=5.0,
                           help='Critical current (µA)')
    gen_parser.add_argument('--T', type=float, default=0.3,
                           help='Junction transparency')
    gen_parser.add_argument('--f', type=float, default=1.0,
                           help='Frequency factor')
    gen_parser.add_argument('--d', type=float, default=0.0,
                           help='Horizontal shift')
    gen_parser.add_argument('--phi0', type=float, default=0.0,
                           help='Phase offset (radians)')
    gen_parser.add_argument('--k', type=float, default=0.0,
                           help='Quadratic background coefficient')
    gen_parser.add_argument('--r', type=float, default=0.0,
                           help='Linear background coefficient')
    gen_parser.add_argument('--C', type=float, default=0.0,
                           help='Current offset (µA)')
    gen_parser.add_argument('--noise-level', type=float, default=0.02,
                           help='Noise level (fraction)')
    gen_parser.add_argument('--noise-type', choices=['gaussian', 'poisson', 'mixed'],
                           default='gaussian', help='Type of noise')
    gen_parser.add_argument('--include-errors', action='store_true',
                           help='Include error column in output')
    
    args = parser.parse_args()
    
    if args.command == 'fit':
        fit_data_cmd(args)
    elif args.command == 'generate':
        generate_synthetic_data_cmd(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
