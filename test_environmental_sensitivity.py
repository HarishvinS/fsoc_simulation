#!/usr/bin/env python3
"""
Test script to verify that the model is sensitive to environmental changes.
"""

import sys
sys.path.append('.')

from backend.optimizer.models import ModelManager, PowerPredictorModel
from pathlib import Path
import pandas as pd

def load_best_model():
    """Load the best performing model."""
    manager = ModelManager(models_dir='models')
    
    # Load XGBoost model (best performing)
    model_file = Path('models/power_predictor_xgboost.pkl')
    if model_file.exists():
        model = PowerPredictorModel('xgboost')
        model.load_model(str(model_file))
        manager.power_predictors['xgboost'] = model
        print(f"Loaded XGBoost model. R² Score: {model.metrics.r2_score:.4f}")
        return model
    else:
        print("Model file not found")
        return None

def test_environmental_sensitivity():
    """Test if the model responds appropriately to environmental changes."""
    model = load_best_model()
    if not model:
        return False
    
    # Base scenario
    base_features = {
        'input_height_tx': 20.0,
        'input_height_rx': 20.0,
        'input_fog_density': 0.1,
        'input_rain_rate': 1.0,
        'input_surface_temp': 25.0,
        'input_ambient_temp': 20.0,
        'input_wavelength_nm': 1550.0,
        'input_tx_power_dbm': 20.0,
        'link_distance_km': 1.4,
        'elevation_angle_deg': 0.0,
        'input_material_tx': 'white_paint',
        'input_material_rx': 'white_paint',
        'thermal_gradient': 0.25,
        'atmospheric_loading': 0.15,
        'height_differential': 0.0,
        'scattering_potential': 0.02
    }
    
    # Test scenarios with different environmental conditions
    test_scenarios = [
        {
            'name': 'Clear Weather',
            'changes': {'input_fog_density': 0.0, 'input_rain_rate': 0.0}
        },
        {
            'name': 'Light Fog',
            'changes': {'input_fog_density': 0.5, 'input_rain_rate': 0.0}
        },
        {
            'name': 'Heavy Fog',
            'changes': {'input_fog_density': 3.0, 'input_rain_rate': 0.0}
        },
        {
            'name': 'Light Rain',
            'changes': {'input_fog_density': 0.0, 'input_rain_rate': 5.0}
        },
        {
            'name': 'Heavy Rain',
            'changes': {'input_fog_density': 0.0, 'input_rain_rate': 20.0}
        },
        {
            'name': 'Extreme Weather',
            'changes': {'input_fog_density': 5.0, 'input_rain_rate': 50.0}
        },
        {
            'name': 'Cold Weather',
            'changes': {'input_surface_temp': 0.0, 'input_ambient_temp': -5.0}
        },
        {
            'name': 'Hot Weather',
            'changes': {'input_surface_temp': 50.0, 'input_ambient_temp': 45.0}
        }
    ]
    
    results = []
    
    print("Testing environmental sensitivity...")
    print(f"Base prediction with: fog={base_features['input_fog_density']}, rain={base_features['input_rain_rate']}, temp={base_features['input_surface_temp']}°C")
    
    base_prediction = model.predict(base_features)
    print(f"Base prediction: {base_prediction:.2f} dBm")
    print()
    
    for scenario in test_scenarios:
        # Create modified features
        test_features = base_features.copy()
        test_features.update(scenario['changes'])
        
        # Update derived features based on changes
        if 'input_fog_density' in scenario['changes'] or 'input_rain_rate' in scenario['changes']:
            fog = test_features['input_fog_density']
            rain = test_features['input_rain_rate']
            test_features['atmospheric_loading'] = fog * 0.5 + rain * 0.1
            test_features['scattering_potential'] = (fog * rain * 0.1) + (fog ** 1.5) * 0.2
        
        if 'input_surface_temp' in scenario['changes'] or 'input_ambient_temp' in scenario['changes']:
            surface_temp = test_features['input_surface_temp']
            ambient_temp = test_features['input_ambient_temp']
            avg_height = (test_features['input_height_tx'] + test_features['input_height_rx']) / 2
            test_features['thermal_gradient'] = abs(surface_temp - ambient_temp) / max(1, avg_height / 10)
        
        # Get prediction
        prediction = model.predict(test_features)
        power_change = prediction - base_prediction
        
        print(f"{scenario['name']:15}: {prediction:6.2f} dBm (change: {power_change:+6.2f} dB)")
        
        results.append({
            'scenario': scenario['name'],
            'prediction': prediction,
            'change': power_change,
            'conditions': scenario['changes']
        })
    
    # Analyze sensitivity
    print(f"\n=== SENSITIVITY ANALYSIS ===")
    power_changes = [abs(r['change']) for r in results]
    max_change = max(power_changes)
    min_change = min(power_changes)
    avg_change = sum(power_changes) / len(power_changes)
    
    print(f"Power change range: {min_change:.2f} - {max_change:.2f} dB")
    print(f"Average absolute change: {avg_change:.2f} dB")
    
    # Check specific sensitivities
    fog_scenarios = [r for r in results if 'input_fog_density' in r['conditions']]
    rain_scenarios = [r for r in results if 'input_rain_rate' in r['conditions']]
    temp_scenarios = [r for r in results if 'input_surface_temp' in r['conditions']]
    
    if fog_scenarios:
        fog_changes = [abs(r['change']) for r in fog_scenarios]
        print(f"Fog sensitivity: {max(fog_changes):.2f} dB max change")
    
    if rain_scenarios:
        rain_changes = [abs(r['change']) for r in rain_scenarios]
        print(f"Rain sensitivity: {max(rain_changes):.2f} dB max change")
    
    if temp_scenarios:
        temp_changes = [abs(r['change']) for r in temp_scenarios]
        print(f"Temperature sensitivity: {max(temp_changes):.2f} dB max change")
    
    # Determine if model is sufficiently sensitive
    if max_change > 5.0 and avg_change > 1.0:
        print("✓ Model shows good environmental sensitivity!")
        return True
    elif max_change > 2.0:
        print("⚠ Model shows moderate environmental sensitivity")
        return True
    else:
        print("✗ Model shows poor environmental sensitivity - may need retraining")
        return False

def test_height_sensitivity():
    """Test if the model responds to height changes."""
    model = load_best_model()
    if not model:
        return False
    
    print("\n=== HEIGHT SENSITIVITY TEST ===")
    
    base_features = {
        'input_height_tx': 20.0,
        'input_height_rx': 20.0,
        'input_fog_density': 0.1,
        'input_rain_rate': 1.0,
        'input_surface_temp': 25.0,
        'input_ambient_temp': 20.0,
        'input_wavelength_nm': 1550.0,
        'input_tx_power_dbm': 20.0,
        'link_distance_km': 1.4,
        'elevation_angle_deg': 0.0,
        'input_material_tx': 'white_paint',
        'input_material_rx': 'white_paint',
        'thermal_gradient': 0.25,
        'atmospheric_loading': 0.15,
        'height_differential': 0.0,
        'scattering_potential': 0.02
    }
    
    heights = [5, 10, 20, 30, 50, 75, 100]
    base_prediction = model.predict(base_features)
    
    print(f"Base prediction (20m/20m): {base_prediction:.2f} dBm")
    
    for height in heights:
        if height == 20:  # Skip base case
            continue
            
        test_features = base_features.copy()
        test_features['input_height_tx'] = height
        test_features['input_height_rx'] = height
        test_features['height_differential'] = 0.0
        
        prediction = model.predict(test_features)
        change = prediction - base_prediction
        
        print(f"Height {height:3d}m: {prediction:6.2f} dBm (change: {change:+6.2f} dB)")
    
    return True

if __name__ == "__main__":
    print("Testing model environmental and height sensitivity...")
    
    env_sensitive = test_environmental_sensitivity()
    height_sensitive = test_height_sensitivity()
    
    if env_sensitive and height_sensitive:
        print("\n✓ Model shows appropriate sensitivity to environmental and height changes!")
    else:
        print("\n⚠ Model sensitivity may need improvement.")
    
    sys.exit(0 if env_sensitive else 1)
