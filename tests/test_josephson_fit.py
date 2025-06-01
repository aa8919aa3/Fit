"""
Unit tests for Josephson junction fitting toolkit.
"""

import unittest
import numpy as np
from josephson_fit import JosephsonFitter, generate_synthetic_data
from josephson_fit.models import Model1, Model2, Model3


class TestModels(unittest.TestCase):
    """Test the Josephson junction models."""
    
    def setUp(self):
        """Set up test data."""
        self.phi_ext = np.linspace(-2, 2, 100)
        self.params = {
            'Ic': 2.0, 'T': 0.3, 'f': 1.0, 'd': 0.0, 'phi0': 0.0,
            'k': 0.0, 'r': 0.0, 'C': 0.0
        }
    
    def test_model1_evaluation(self):
        """Test Model 1 evaluation."""
        model = Model1()
        result = model.evaluate(self.phi_ext, self.params)
        
        self.assertEqual(len(result), len(self.phi_ext))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
    
    def test_model2_evaluation(self):
        """Test Model 2 evaluation."""
        model = Model2()
        result = model.evaluate(self.phi_ext, self.params)
        
        self.assertEqual(len(result), len(self.phi_ext))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
    
    def test_model3_evaluation(self):
        """Test Model 3 evaluation."""
        model = Model3()
        result = model.evaluate(self.phi_ext, self.params)
        
        self.assertEqual(len(result), len(self.phi_ext))
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
    
    def test_transparency_limit(self):
        """Test behavior at transparency limits."""
        model = Model1()
        
        # Low transparency
        params_low = self.params.copy()
        params_low['T'] = 0.001
        result_low = model.evaluate(self.phi_ext, params_low)
        
        # High transparency  
        params_high = self.params.copy()
        params_high['T'] = 0.999
        result_high = model.evaluate(self.phi_ext, params_high)
        
        self.assertFalse(np.any(np.isnan(result_low)))
        self.assertFalse(np.any(np.isnan(result_high)))


class TestSyntheticData(unittest.TestCase):
    """Test synthetic data generation."""
    
    def setUp(self):
        """Set up test parameters."""
        self.phi_ext = np.linspace(-1, 1, 50)
        self.params = {
            'Ic': 1.5, 'T': 0.4, 'f': 1.2, 'd': 0.0, 'phi0': 0.1,
            'k': 0.01, 'r': 0.0, 'C': 0.0
        }
    
    def test_generate_model1_data(self):
        """Test Model 1 data generation."""
        current = generate_synthetic_data(self.phi_ext, model_type=1, 
                                        params=self.params, noise_level=0.02)
        
        self.assertEqual(len(current), len(self.phi_ext))
        self.assertFalse(np.any(np.isnan(current)))
        self.assertFalse(np.any(np.isinf(current)))
    
    def test_generate_model2_data(self):
        """Test Model 2 data generation."""
        current = generate_synthetic_data(self.phi_ext, model_type=2, 
                                        params=self.params, noise_level=0.02)
        
        self.assertEqual(len(current), len(self.phi_ext))
        self.assertFalse(np.any(np.isnan(current)))
        self.assertFalse(np.any(np.isinf(current)))
    
    def test_generate_model3_data(self):
        """Test Model 3 data generation."""
        current = generate_synthetic_data(self.phi_ext, model_type=3, 
                                        params=self.params, noise_level=0.02)
        
        self.assertEqual(len(current), len(self.phi_ext))
        self.assertFalse(np.any(np.isnan(current)))
        self.assertFalse(np.any(np.isinf(current)))
    
    def test_noise_levels(self):
        """Test different noise levels."""
        for noise_level in [0.0, 0.01, 0.05, 0.1]:
            current = generate_synthetic_data(self.phi_ext, model_type=1,
                                            params=self.params, 
                                            noise_level=noise_level)
            
            self.assertEqual(len(current), len(self.phi_ext))
            self.assertFalse(np.any(np.isnan(current)))


class TestFitting(unittest.TestCase):
    """Test the fitting functionality."""
    
    def setUp(self):
        """Set up test data."""
        self.phi_ext = np.linspace(-2, 2, 100)
        self.params = {
            'Ic': 2.0, 'T': 0.35, 'f': 1.1, 'd': 0.05, 'phi0': 0.2,
            'k': 0.02, 'r': -0.01, 'C': 0.1
        }
        self.current = generate_synthetic_data(self.phi_ext, model_type=2,
                                             params=self.params, 
                                             noise_level=0.01)
        self.fitter = JosephsonFitter(verbose=False)
    
    def test_fit_model1(self):
        """Test Model 1 fitting."""
        result = self.fitter.fit_model1(self.phi_ext, self.current)
        
        self.assertIsNotNone(result)
        self.assertIn('Ic', result.params)
        self.assertGreater(result.r_squared, 0.5)  # Should have reasonable fit
    
    def test_fit_model2(self):
        """Test Model 2 fitting."""
        result = self.fitter.fit_model2(self.phi_ext, self.current)
        
        self.assertIsNotNone(result)
        self.assertIn('Ic', result.params)
        self.assertGreater(result.r_squared, 0.8)  # Should fit well (correct model)
    
    def test_fit_model3(self):
        """Test Model 3 fitting."""
        result = self.fitter.fit_model3(self.phi_ext, self.current)
        
        self.assertIsNotNone(result)
        self.assertIn('Ic', result.params)
        self.assertGreater(result.r_squared, 0.5)
    
    def test_fit_all_models(self):
        """Test fitting all models."""
        results = self.fitter.fit_all_models(self.phi_ext, self.current)
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r is not None for r in results))
    
    def test_model_comparison(self):
        """Test model comparison."""
        results = self.fitter.fit_all_models(self.phi_ext, self.current)
        best_model = self.fitter.compare_models(results)
        
        self.assertIsNotNone(best_model)
        self.assertIn('Ic', best_model.params)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def setUp(self):
        """Set up test data."""
        self.phi_ext = np.linspace(-2, 2, 100)
        # Generate data with known frequency
        self.frequency = 1.5
        self.current = np.sin(2 * np.pi * self.frequency * self.phi_ext) + \
                      0.1 * np.random.randn(len(self.phi_ext))
    
    def test_frequency_detection(self):
        """Test Lomb-Scargle frequency detection."""
        from josephson_fit.utils import lombscargle_frequency_detection
        
        detected_freq, power = lombscargle_frequency_detection(self.phi_ext, 
                                                              self.current)
        
        # Should detect frequency within 10% of true value
        self.assertAlmostEqual(detected_freq, self.frequency, delta=0.15)
        self.assertGreater(power, 0.1)
    
    def test_initial_parameter_estimation(self):
        """Test initial parameter estimation."""
        from josephson_fit.utils import estimate_initial_parameters
        
        params = estimate_initial_parameters(self.phi_ext, self.current)
        
        self.assertIn('Ic', params)
        self.assertIn('f', params)
        self.assertGreater(params['Ic'], 0)
        self.assertGreater(params['f'], 0)
        self.assertLessEqual(params['T'], 1.0)
        self.assertGreaterEqual(params['T'], 0.0)


if __name__ == '__main__':
    unittest.main()
