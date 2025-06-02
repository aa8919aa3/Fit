#!/usr/bin/env python3
import pytest
# Only skip when collected by pytest (imported), not when run as script
if __name__ != "__main__":
    pytest.skip("Skipping real data test script in pytest", allow_module_level=True)
"""
Real data testing script for Josephson Junction Fitting Toolkit.
Tests the toolkit with real experimental data from 164_ic.csv.
"""

__test__ = False  # prevent pytest from collecting this script as tests
import numpy as np
__test__ = False  # Prevent pytest from collecting this script as a test module
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_real_data():
    """Load and preprocess the real experimental data."""
    print("="*60)
    print("LOADING REAL EXPERIMENTAL DATA")
    print("="*60)
    
    try:
        # Load the CSV data
        data_path = '/workspaces/Fit/data/317_ic.csv'
        df = pd.read_csv(data_path)
        
        print(f"✓ Data loaded successfully from {data_path}")
        print(f"✓ Data shape: {df.shape}")
        print(f"✓ Columns: {list(df.columns)}")
        
        # Extract flux and current data
        flux = df['Flux'].values
        current = df['Ic'].values
        
        # Convert flux to phi_ext (external flux in units of flux quantum)
        # Assuming flux is in Tesla and we need to normalize
        phi_ext = flux * 1000  # Scale factor for better fitting
        
        # Generate error estimates (typically 1-5% of signal)
        current_err = np.maximum(current * 0.02, np.min(current) * 0.1)
        
        print(f"✓ Flux range: {np.min(flux):.6f} to {np.max(flux):.6f}")
        print(f"✓ Current range: {np.min(current):.2e} to {np.max(current):.2e} A")
        print(f"✓ Phi_ext range: {np.min(phi_ext):.3f} to {np.max(phi_ext):.3f}")
        print(f"✓ Estimated uncertainties: {np.min(current_err):.2e} to {np.max(current_err):.2e} A")
        
        return phi_ext, current, current_err, flux
        
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        raise

def run_model_fitting(phi_ext, current, current_err):
    """Test all three models with the real data."""
    print("\n" + "="*60)
    print("TESTING ALL THREE MODELS WITH REAL DATA")
    print("="*60)
    
    try:
        from josephson_fit import JosephsonTripleFitter
        from josephson_fit.warnings_config import configure_josephson_warnings
        
        # Suppress warnings for cleaner output
        configure_josephson_warnings(verbose=False)
        
        # Create fitter instance
        fitter = JosephsonTripleFitter(phi_ext, current, current_err)
        print("✓ JosephsonTripleFitter created successfully")
        
        models = ['model1', 'model2', 'model3']
        results = {}
        
        for model_name in models:
            print(f"\nTesting {model_name.upper()}:")
            
            try:
                import time
                start_time = time.time()
                
                # Fit the model
                result = fitter.fit_model(model_name)
                
                fit_time = time.time() - start_time
                
                # Store results
                results[model_name] = result
                
                print(f"  ✓ Fit completed in {fit_time:.3f} seconds")
                print(f"  ✓ R² = {result.r_squared:.4f}")
                print(f"  ✓ Reduced χ² = {result.reduced_chi_square:.4f}")
                print(f"  ✓ AIC = {result.aic:.2f}")
                print(f"  ✓ BIC = {result.bic:.2f}")
                
                # Print key parameters
                print("  Key parameters:")
                if hasattr(result, 'params'):
                    for param_name in ['Ic', 'T', 'f']:
                        if param_name in result.params:
                            param = result.params[param_name]
                            if hasattr(param, 'value') and hasattr(param, 'stderr'):
                                print(f"    {param_name} = {param.value:.4e} ± {param.stderr:.4e}")
                            else:
                                print(f"    {param_name} = {param:.4e}")
                
            except Exception as e:
                print(f"  ✗ {model_name} fitting failed: {e}")
                results[model_name] = None
                continue
                
        # Mark successful results
        successful_count = sum(1 for v in results.values() if v is not None)
        print(f"\n✓ Successfully fitted {successful_count}/{len(models)} models")
        
        return results
        
    except Exception as e:
        print(f"✗ Model fitting test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(results):
    """Compare the fitting results of all models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    if not results:
        print("✗ No results to compare")
        return
    
    # Filter successful fits
    successful_results = {k: v for k, v in results.items() if v is not None}
    
    if not successful_results:
        print("✗ No successful fits to compare")
        return
    
    print(f"Comparing {len(successful_results)} successful model fits:")
    print()
    print(f"{'Model':<10} {'R²':<10} {'Reduced χ²':<12} {'AIC':<10} {'BIC':<10}")
    print("-" * 52)
    
    best_r2 = 0
    best_chi2 = float('inf')
    best_aic = float('inf')
    best_model = None
    
    for model_name, result in successful_results.items():
        r2 = result.r_squared
        chi2 = result.reduced_chi_square
        aic = result.aic
        bic = result.bic
        
        print(f"{model_name:<10} {r2:<10.4f} {chi2:<12.4f} {aic:<10.2f} {bic:<10.2f}")
        
        # Track best model by AIC (lower is better)
        if aic < best_aic:
            best_aic = aic
            best_model = model_name
            best_r2 = r2
            best_chi2 = chi2
    
    print()
    print(f"✓ Best model by AIC: {best_model}")
    print(f"  - R² = {best_r2:.4f}")
    print(f"  - Reduced χ² = {best_chi2:.4f}")
    print(f"  - AIC = {best_aic:.2f}")

def create_visualization(phi_ext, current, current_err, flux, results):
    """Create visualization plots of the data and fits."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Data vs Flux
        ax1.errorbar(flux, current*1e6, yerr=current_err*1e6, 
                    fmt='o', markersize=2, alpha=0.6, label='Experimental Data')
        ax1.set_xlabel('Flux (T)')
        ax1.set_ylabel('Critical Current (μA)')
        ax1.set_title('Experimental Data: Critical Current vs Flux')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Data vs Phi_ext with model fits
        ax2.errorbar(phi_ext, current*1e6, yerr=current_err*1e6,
                    fmt='o', markersize=2, alpha=0.6, label='Data', color='black')
        
        # Plot fitted models
        colors = ['red', 'blue', 'green']
        for i, (model_name, result) in enumerate(results.items()):
            if result is not None:
                # Generate smooth curve for plotting
                phi_smooth = np.linspace(np.min(phi_ext), np.max(phi_ext), 500)
                if hasattr(result, 'eval'):
                    fitted_curve = result.eval(phi_ext=phi_smooth)
                    ax2.plot(phi_smooth, fitted_curve*1e6, 
                            color=colors[i], linewidth=2,
                            label=f'{model_name} (R²={result.r_squared:.3f})')
        
        ax2.set_xlabel('Phi_ext (normalized)')
        ax2.set_ylabel('Critical Current (μA)')
        ax2.set_title('Model Fits to Real Data')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_path = '/workspaces/Fit/real_data_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")

def main():
    """Main function to run all tests."""
    print("JOSEPHSON JUNCTION FITTING TOOLKIT")
    print("REAL DATA TESTING WITH 164_ic.csv")
    print("="*60)
    
    try:
        # Load real data
        phi_ext, current, current_err, flux = load_real_data()
        
        # Test model fitting
        results = test_model_fitting(phi_ext, current, current_err)
        
        if results:
            # Compare models
            compare_models(results)
            
            # Create visualization
            create_visualization(phi_ext, current, current_err, flux, results)
            
            print("\n" + "="*60)
            print("REAL DATA TESTING COMPLETED SUCCESSFULLY")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("REAL DATA TESTING FAILED")
            print("="*60)
            return False
            
    except Exception as e:
        print(f"\n✗ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
