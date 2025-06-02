#!/usr/bin/env python3
"""
Model Comparison Example

This example demonstrates how to systematically compare the three Josephson
junction models and select the optimal one based on information criteria.
"""

import numpy as np
import matplotlib.pyplot as plt
from josephson_fit import (
    JosephsonTripleFitter, 
    generate_synthetic_data,
    Model1, Model2, Model3
)


def generate_test_datasets():
    """Generate test datasets for each model type."""
    phi_ext = np.linspace(-2.5, 2.5, 200)
    
    datasets = {}
    
    # Dataset 1: Simple Model 1 data
    params1 = {
        'Ic': 2.0, 'T': 0.3, 'f': 1.0, 'd': 0.0, 'phi0': 0.0,
        'k': 0.02, 'r': 0.0, 'C': 0.0
    }
    current1 = generate_synthetic_data(phi_ext, Model1().function, 
                                     params1, noise_level=0.02)
    datasets['Model 1 Data'] = (phi_ext, current1, params1, 1)
    
    # Dataset 2: Model 2 with significant second-order effects
    params2 = {
        'Ic': 2.5, 'T': 0.4, 'f': 1.2, 'd': 0.1, 'phi0': 0.3,
        'k': 0.03, 'r': -0.05, 'C': 0.1
    }
    current2 = generate_synthetic_data(phi_ext, Model2().function, 
                                     params2, noise_level=0.02)
    datasets['Model 2 Data'] = (phi_ext, current2, params2, 2)
    
    # Dataset 3: Model 3 with third-order harmonics
    params3 = {
        'Ic': 3.0, 'T': 0.5, 'f': 1.4, 'd': -0.1, 'phi0': 0.4,
        'k': 0.04, 'r': 0.02, 'C': -0.05
    }
    current3 = generate_synthetic_data(phi_ext, Model3().function, 
                                     params3, noise_level=0.025)
    datasets['Model 3 Data'] = (phi_ext, current3, params3, 3)
    
    return datasets


def analyze_dataset(name, phi_ext, current, true_params, true_model):
    """Analyze a single dataset with all three models."""
    print(f"\nAnalyzing {name}")
    print("=" * (len(name) + 10))
    
    print(f"True model: Model {true_model}")
    print("True parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value}")
    
    # Create fitter for this dataset
    fitter = JosephsonTripleFitter(phi_ext, current)
    
    # Fit all models
    results = fitter.fit_all_models()
    valid_results = [r for r in results.values() if r is not None]
    
    if not valid_results:
        print("No successful fits!")
        return None
    
    # Model selection
    best_model = fitter.select_best_model()
    
    # Calculate delta AIC and BIC
    print("\nModel Selection Metrics:")
    print("-" * 30)
    min_aic = min(r.aic for r in valid_results)
    min_bic = min(r.bic for r in valid_results)
    
    print(f"{'Model':<15} {'AIC':<10} {'ΔAIC':<10} {'BIC':<10} {'ΔBIC':<10} {'R²':<10}")
    print("-" * 70)
    
    for i, (model_type, result) in enumerate(results.items()):
        if result is not None:
            model_num = int(model_type[-1])  # Extract number from 'model1', 'model2', etc.
            delta_aic = result.aic - min_aic
            delta_bic = result.bic - min_bic
            
            marker = " *" if result == best_model else ""
            print(f"Model {model_num:<9} {result.aic:<10.2f} {delta_aic:<10.2f} "
                  f"{result.bic:<10.2f} {delta_bic:<10.2f} {result.r_squared:<10.4f}{marker}")
    
    # Check if correct model was selected
    best_model_num = int(best_model.model_type[-1])  # Extract from 'model1', etc.
    correct_selection = (best_model_num == true_model)
    
    print(f"\nSelected Model: {best_model_num}")
    print(f"Correct Selection: {'✓' if correct_selection else '✗'}")
    
    return {
        'name': name,
        'true_model': true_model,
        'selected_model': best_model_num,
        'correct': correct_selection,
        'results': results,
        'phi_ext': phi_ext,
        'current': current
    }


def plot_model_comparison_summary(analysis_results):
    """Plot summary of model comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Model selection accuracy
    ax1 = axes[0, 0]
    dataset_names = [r['name'] for r in analysis_results]
    correct_selections = [r['correct'] for r in analysis_results]
    
    colors = ['green' if correct else 'red' for correct in correct_selections]
    bars = ax1.bar(range(len(dataset_names)), [1]*len(dataset_names), 
                   color=colors, alpha=0.7)
    
    ax1.set_xticks(range(len(dataset_names)))
    ax1.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax1.set_ylabel('Selection Result')
    ax1.set_title('Model Selection Accuracy')
    ax1.set_ylim(0, 1.2)
    
    # Add text labels
    for i, (bar, correct) in enumerate(zip(bars, correct_selections)):
        text = "Correct" if correct else "Incorrect"
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                text, ha='center', va='center', fontweight='bold')
    
    # Plot 2: AIC differences for each dataset
    ax2 = axes[0, 1]
    for i, result in enumerate(analysis_results):
        results = result['results']
        valid_results = [r for r in results.values() if r is not None]
        
        if valid_results:
            aics = [r.aic for r in valid_results]
            min_aic = min(aics)
            delta_aics = [aic - min_aic for aic in aics]
            
            models = [f"M{j+1}" for j in range(len(delta_aics))]
            x_pos = [j + i*0.25 for j in range(len(models))]
            
            ax2.bar(x_pos, delta_aics, width=0.2, alpha=0.7, 
                   label=result['name'])
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('ΔAIC')
    ax2.set_title('AIC Differences by Dataset')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: R² comparison
    ax3 = axes[1, 0]
    model_names = ['Model 1', 'Model 2', 'Model 3']
    
    for i, result in enumerate(analysis_results):
        results = result['results']
        r_squareds = []
        
        for j, (model_type, res) in enumerate(results.items()):
            if res is not None:
                r_squareds.append(res.r_squared)
            else:
                r_squareds.append(0)
        
        x_pos = [j + i*0.25 for j in range(len(r_squareds))]
        ax3.bar(x_pos, r_squareds, width=0.2, alpha=0.7, 
               label=result['name'])
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('R²')
    ax3.set_title('R² Values by Model and Dataset')
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(model_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Parameter recovery (simplified placeholder)
    ax4 = axes[1, 1]
    
    ax4.text(0.5, 0.5, 'Parameter Recovery\nAnalysis\n(Implementation needed)', 
            ha='center', va='center', transform=ax4.transAxes,
            fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat'))
    ax4.set_title('Parameter Recovery Analysis')
    
    plt.tight_layout()
    return fig


def main():
    """Run model comparison example."""
    print("Josephson Junction Fitting - Model Comparison")
    print("=" * 50)
    
    # Generate test datasets
    print("Generating test datasets...")
    datasets = generate_test_datasets()
    
    # Analyze each dataset
    analysis_results = []
    
    for name, (phi_ext, current, true_params, true_model) in datasets.items():
        result = analyze_dataset(name, phi_ext, current, true_params, true_model)
        if result:
            analysis_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    correct_count = sum(1 for r in analysis_results if r['correct'])
    total_count = len(analysis_results)
    
    print(f"Model Selection Accuracy: {correct_count}/{total_count} "
          f"({100*correct_count/total_count:.1f}%)")
    
    print("\nDataset Results:")
    for result in analysis_results:
        status = "✓" if result['correct'] else "✗"
        print(f"  {result['name']:<15}: True={result['true_model']}, "
              f"Selected={result['selected_model']} {status}")
    
    # Plot individual fits for each dataset
    for result in analysis_results:
        # Create fitter for plotting and recreate the fits
        fitter = JosephsonTripleFitter(result['phi_ext'], result['current'])
        fitter_results = fitter.fit_all_models()
        
        # Convert dict of results to list for visualization function
        results_list = [res for res in fitter_results.values() if res is not None]
        
        # Use visualization module for plotting
        from josephson_fit.visualization import plot_model_comparison
        if results_list:
            fig = plot_model_comparison(results_list)
            fig.suptitle(f"{result['name']} - Model Comparison", fontsize=16)
    
    # Plot summary
    plot_model_comparison_summary(analysis_results)
    
    plt.show()
    
    print("\nModel comparison analysis complete!")


if __name__ == "__main__":
    main()
