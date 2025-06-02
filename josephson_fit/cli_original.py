#!/usr/bin/env python3
"""
Command Line Interface for Josephson Junction Fitting Toolkit

This module provides a command-line interface for fitting Josephson junction
current-phase relationships.
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from josephson_fit import JosephsonTripleFitter, generate_synthetic_data, Model1, Model2, Model3, fit_all_josephson_models, compare_all_models


def load_data(filepath):
    """Load data from file."""
    try:
        data = np.loadtxt(filepath)
        if data.shape[1] < 2:
            raise ValueError("Data file must have at least 2 columns (phi_ext, current)")
        
        phi_ext = data[:, 0]
        current = data[:, 1]
        
        print(f"Loaded {len(phi_ext)} data points from {filepath}")
        return phi_ext, current
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def fit_data(args):
    """Perform fitting based on command line arguments."""
    # Load data
    phi_ext, current = load_data(args.input)
    
    print(f"Fitting data with {len(phi_ext)} points...")
    
    if args.model == 'all':
        # Fit all models
        print("Fitting all models...")
        results = fit_all_josephson_models(phi_ext, current)
        
        # Compare models
        best_model = compare_all_models(results)
        print(f"\nBest model: {best_model['best_model']}")
        
        # Plot comparison
        if args.plot:
            # Create plots manually since we don't have the old comparison method
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            
            model_names = ['Model 1', 'Model 2', 'Model 3']
            colors = ['blue', 'red', 'green']
            
            for i, (model_key, result) in enumerate(results.items()):
                if result is not None:
                    model_idx = int(model_key[-1]) - 1
                    
                    # Data and fit plot
                    ax1 = axes[0, model_idx]
                    ax1.scatter(phi_ext, current, alpha=0.7, s=20, color='gray', label='Data')
                    ax1.plot(phi_ext, result.fitted_curve, color=colors[model_idx], linewidth=2, label='Fit')
                    ax1.set_xlabel('Φ_ext')
                    ax1.set_ylabel('Current')
                    ax1.set_title(f'{model_names[model_idx]}\nR² = {result.r_squared:.4f}')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Residuals
                    ax2 = axes[1, model_idx]
                    ax2.scatter(phi_ext, result.residuals, alpha=0.7, s=20, color=colors[model_idx])
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax2.set_xlabel('Φ_ext')
                    ax2.set_ylabel('Residuals')
                    ax2.set_title('Fit Residuals')
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if args.output:
                plot_path = Path(args.output).with_suffix('.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {plot_path}")
            
            if not args.no_display:
                plt.show()
        
        return results
    
    else:
        # Fit specific model
        model_num = int(args.model)
        if model_num not in [1, 2, 3]:
            print("Error: Model must be 1, 2, 3, or 'all'")
            sys.exit(1)
        
        print(f"Fitting Model {model_num}...")
        
        # Create fitter and fit specific model
        fitter = JosephsonTripleFitter(phi_ext, current)
        result = fitter.fit_model(f'model{model_num}')
        
        # Plot if requested
        if args.plot:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Data and fit
            ax1.scatter(phi_ext, current, alpha=0.7, s=20, color='blue', label='Data')
            ax1.plot(phi_ext, result.fitted_curve, 'red', linewidth=2, label='Fit')
            ax1.set_xlabel('Φ_ext')
            ax1.set_ylabel('Current')
            ax1.set_title(f'{result.model_name}\nR² = {result.r_squared:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Residuals
            ax2.scatter(phi_ext, result.residuals, alpha=0.7, s=20, color='red')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Φ_ext')
            ax2.set_ylabel('Residuals')
            ax2.set_title('Fit Residuals')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if args.output:
                plot_path = Path(args.output).with_suffix('.png')
                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {plot_path}")
            
            if not args.no_display:
                plt.show()
        
        return result


def generate_synthetic(args):
    """Generate synthetic data."""
    phi_ext = np.linspace(args.phi_min, args.phi_max, args.n_points)
    
    # Default parameters
    params = {
        'Ic': args.Ic,
        'T': args.T,
        'f': args.f,
        'd': args.d,
        'phi0': args.phi0,
        'k': args.k,
        'r': args.r,
        'C': args.C
    }
    
    print(f"Generating synthetic data for Model {args.model}")
    print("Parameters:")
    for name, value in params.items():
        print(f"  {name}: {value}")
    
    # Get the appropriate model function
    if args.model == 1:
        model_func = Model1().function
    elif args.model == 2:
        model_func = Model2().function
    else:  # args.model == 3
        model_func = Model3().function
    
    current, _ = generate_synthetic_data(phi_ext, model_func, params, args.noise)
    
    # Save data
    data = np.column_stack((phi_ext, current))
    np.savetxt(args.output, data, header='phi_ext current', 
               fmt='%.6f', delimiter='\t')
    
    print(f"Synthetic data saved to {args.output}")
    
    # Plot if requested
    if args.plot:
        plt.figure(figsize=(10, 6))
        plt.plot(phi_ext, current, 'b-', linewidth=1.5)
        plt.xlabel('Φ_ext')
        plt.ylabel('Current')
        plt.title(f'Synthetic Model {args.model} Data (noise: {args.noise:.1%})')
        plt.grid(True, alpha=0.3)
        
        if not args.no_display:
            plt.show()


def analyze_frequency(args):
    """Perform frequency analysis."""
    phi_ext, current = load_data(args.input)
    
    fitter = JosephsonFitter(verbose=args.verbose)
    
    print("Performing frequency analysis...")
    detected_freq, fig = fitter.analyze_frequency(phi_ext, current)
    
    print(f"Detected frequency: {detected_freq:.6f}")
    
    if args.output:
        plot_path = Path(args.output).with_suffix('.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Periodogram saved to {plot_path}")
    
    if not args.no_display:
        plt.show()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Josephson Junction Current-Phase Relationship Fitting Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fit all models to data
  josephson-fit fit data.txt --model all --plot
  
  # Fit specific model
  josephson-fit fit data.txt --model 2 --plot --output results
  
  # Generate synthetic data
  josephson-fit generate --model 2 --output synthetic.txt --plot
  
  # Frequency analysis
  josephson-fit frequency data.txt --output freq_analysis
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Fit command
    fit_parser = subparsers.add_parser('fit', help='Fit Josephson models to data')
    fit_parser.add_argument('input', help='Input data file (2 columns: phi_ext, current)')
    fit_parser.add_argument('--model', default='all', 
                           help='Model to fit (1, 2, 3, or "all")')
    fit_parser.add_argument('--plot', action='store_true', 
                           help='Generate plots')
    fit_parser.add_argument('--output', help='Output file prefix for plots/results')
    fit_parser.add_argument('--no-display', action='store_true',
                           help='Don\'t display plots (save only)')
    fit_parser.add_argument('--verbose', action='store_true',
                           help='Verbose output')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate synthetic data')
    gen_parser.add_argument('--model', type=int, default=2, choices=[1, 2, 3],
                           help='Model type (1, 2, or 3)')
    gen_parser.add_argument('--output', required=True,
                           help='Output data file')
    gen_parser.add_argument('--phi-min', type=float, default=-2.0,
                           help='Minimum phi_ext value')
    gen_parser.add_argument('--phi-max', type=float, default=2.0,
                           help='Maximum phi_ext value')
    gen_parser.add_argument('--n-points', type=int, default=200,
                           help='Number of data points')
    gen_parser.add_argument('--noise', type=float, default=0.02,
                           help='Noise level (fraction)')
    gen_parser.add_argument('--plot', action='store_true',
                           help='Plot generated data')
    gen_parser.add_argument('--no-display', action='store_true',
                           help='Don\'t display plots')
    
    # Model parameters
    gen_parser.add_argument('--Ic', type=float, default=2.0,
                           help='Critical current')
    gen_parser.add_argument('--T', type=float, default=0.3,
                           help='Transparency')
    gen_parser.add_argument('--f', type=float, default=1.0,
                           help='Frequency')
    gen_parser.add_argument('--d', type=float, default=0.0,
                           help='Horizontal shift')
    gen_parser.add_argument('--phi0', type=float, default=0.0,
                           help='Phase offset')
    gen_parser.add_argument('--k', type=float, default=0.0,
                           help='Quadratic coefficient')
    gen_parser.add_argument('--r', type=float, default=0.0,
                           help='Linear coefficient')
    gen_parser.add_argument('--C', type=float, default=0.0,
                           help='Constant offset')
    
    # Frequency analysis command
    freq_parser = subparsers.add_parser('frequency', help='Analyze frequency content')
    freq_parser.add_argument('input', help='Input data file')
    freq_parser.add_argument('--output', help='Output file for periodogram plot')
    freq_parser.add_argument('--no-display', action='store_true',
                            help='Don\'t display plots')
    freq_parser.add_argument('--verbose', action='store_true',
                            help='Verbose output')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    if args.command == 'fit':
        fit_data(args)
    elif args.command == 'generate':
        generate_synthetic(args)
    elif args.command == 'frequency':
        analyze_frequency(args)


if __name__ == '__main__':
    main()
