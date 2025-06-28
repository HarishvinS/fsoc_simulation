#!/usr/bin/env python3
"""
Simple test script to debug optimization issues.
"""

import sys
sys.path.append('.')

from backend.optimizer.models import ModelManager, PowerPredictorModel
from pathlib import Path
import pandas as pd

def test_model_loading():
    """Test if models can be loaded properly."""
    print("Testing model loading...")
    
    manager = ModelManager(models_dir='models')
    
    # Load XGBoost model
    model_file = Path('models/power_predictor_xgboost.pkl')
    if model_file.exists():
        print(f"Loading model from {model_file}")
        model = PowerPredictorModel('xgboost')
        model.load_model(str(model_file))
        manager.power_predictors['xgboost'] = model
        print(f"Model loaded successfully. R² Score: {model.metrics.r2_score:.4f}")
        return manager
    else:
        print("Model file not found")
        return None

def test_prediction(manager):
    """Test if prediction works."""
    print("\nTesting prediction...")
    
    if not manager.power_predictors:
        print("No models loaded")
        return False
    
    model = list(manager.power_predictors.values())[0]
    
    # Create test features with correct naming
    test_features = {
        'input_height_tx': 20.0,
        'input_height_rx': 15.0,
        'input_fog_density': 0.1,
        'input_rain_rate': 1.0,
        'input_surface_temp': 25.0,
        'input_ambient_temp': 20.0,
        'input_wavelength_nm': 1550.0,
        'input_tx_power_dbm': 20.0,
        'link_distance_km': 1.4,
        'elevation_angle_deg': 0.2,
        'input_material_tx': 'white_paint',
        'input_material_rx': 'white_paint',
        'thermal_gradient': 0.25,
        'atmospheric_loading': 0.15,
        'height_differential': 5.0,
        'scattering_potential': 0.02
    }
    
    try:
        prediction = model.predict(test_features)
        print(f"Prediction successful: {prediction:.2f} dBm")
        return True
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_optimization(manager):
    """Test a very simple optimization."""
    print("\nTesting simple optimization...")
    
    try:
        optimizer = manager.create_deployment_optimizer()
        print("Optimizer created successfully")
        
        # Very simple test case
        base_conditions = {
            'input_lat_tx': 37.7749, 'input_lon_tx': -122.4194,
            'input_lat_rx': 37.7849, 'input_lon_rx': -122.4094,
            'input_avg_fog_density': 0.1, 'input_avg_rain_rate': 1.0,
            'input_avg_surface_temp': 20, 'input_avg_ambient_temp': 18,
            'input_wavelength_nm': 1550, 'input_tx_power_dbm': 20,
            'reliability_target': 0.99
        }
        
        constraints = {
            'min_height': 10, 'max_height': 30,  # Small range
            'available_materials': ['white_paint'],  # Single material
            'min_received_power': -30
        }
        
        print("Running optimization with limited parameter space...")
        result = optimizer.optimize_deployment(base_conditions, constraints)
        print(f"Optimization completed successfully!")
        print(f"Result keys: {list(result.keys())}")
        print(f"TX Height: {result.get('height_tx', 'N/A')}")
        print(f"RX Height: {result.get('height_rx', 'N/A')}")
        print(f"Power: {result.get('predicted_power_dbm', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting optimization debugging...")
    
    # Test 1: Model loading
    manager = test_model_loading()
    if not manager:
        print("Model loading failed, exiting")
        sys.exit(1)
    
    # Test 2: Prediction
    if not test_prediction(manager):
        print("Prediction test failed, exiting")
        sys.exit(1)
    
    # Test 3: Simple optimization
    if not test_simple_optimization(manager):
        print("Optimization test failed")
        sys.exit(1)
    
    print("\nAll tests passed!")
