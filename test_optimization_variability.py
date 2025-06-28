#!/usr/bin/env python3
"""
Test script to verify optimization produces varied results for different scenarios.
"""

import sys
sys.path.append('.')

from backend.optimizer.models import ModelManager, PowerPredictorModel
from pathlib import Path

def load_models():
    """Load trained models."""
    manager = ModelManager(models_dir='models')
    
    # Load XGBoost model (best performing)
    model_file = Path('models/power_predictor_xgboost.pkl')
    if model_file.exists():
        model = PowerPredictorModel('xgboost')
        model.load_model(str(model_file))
        manager.power_predictors['xgboost'] = model
        print(f"Loaded XGBoost model. R² Score: {model.metrics.r2_score:.4f}")
        return manager
    else:
        print("Model file not found")
        return None

def test_optimization_scenarios():
    """Test optimization with vastly different scenarios."""
    manager = load_models()
    if not manager:
        return False
    
    optimizer = manager.create_deployment_optimizer()
    
    # Define very different test scenarios
    scenarios = [
        {
            'name': 'Clear Weather - Short Distance',
            'conditions': {
                'input_lat_tx': 37.7749, 'input_lon_tx': -122.4194,
                'input_lat_rx': 37.7849, 'input_lon_rx': -122.4094,
                'input_avg_fog_density': 0.0, 'input_avg_rain_rate': 0.0,
                'input_avg_surface_temp': 25, 'input_avg_ambient_temp': 20,
                'input_wavelength_nm': 1550, 'input_tx_power_dbm': 20,
                'reliability_target': 0.95
            },
            'constraints': {
                'min_height': 5, 'max_height': 50,
                'available_materials': ['white_paint', 'aluminum', 'steel'],
                'min_received_power': -30
            }
        },
        {
            'name': 'Heavy Fog - Short Distance',
            'conditions': {
                'input_lat_tx': 37.7749, 'input_lon_tx': -122.4194,
                'input_lat_rx': 37.7849, 'input_lon_rx': -122.4094,
                'input_avg_fog_density': 3.0, 'input_avg_rain_rate': 0.5,
                'input_avg_surface_temp': 15, 'input_avg_ambient_temp': 12,
                'input_wavelength_nm': 1550, 'input_tx_power_dbm': 20,
                'reliability_target': 0.95
            },
            'constraints': {
                'min_height': 5, 'max_height': 50,
                'available_materials': ['white_paint', 'aluminum', 'steel'],
                'min_received_power': -30
            }
        },
        {
            'name': 'Heavy Rain - Long Distance',
            'conditions': {
                'input_lat_tx': 40.7128, 'input_lon_tx': -74.0060,
                'input_lat_rx': 40.7589, 'input_lon_rx': -73.9851,
                'input_avg_fog_density': 0.5, 'input_avg_rain_rate': 20.0,
                'input_avg_surface_temp': 30, 'input_avg_ambient_temp': 25,
                'input_wavelength_nm': 1550, 'input_tx_power_dbm': 25,
                'reliability_target': 0.90
            },
            'constraints': {
                'min_height': 10, 'max_height': 100,
                'available_materials': ['aluminum', 'steel'],
                'min_received_power': -25
            }
        },
        {
            'name': 'Extreme Weather - High Requirements',
            'conditions': {
                'input_lat_tx': 51.5074, 'input_lon_tx': -0.1278,
                'input_lat_rx': 51.5174, 'input_lon_rx': -0.1178,
                'input_avg_fog_density': 5.0, 'input_avg_rain_rate': 50.0,
                'input_avg_surface_temp': 5, 'input_avg_ambient_temp': 2,
                'input_wavelength_nm': 1310, 'input_tx_power_dbm': 30,
                'reliability_target': 0.999
            },
            'constraints': {
                'min_height': 20, 'max_height': 150,
                'available_materials': ['steel'],
                'min_received_power': -15
            }
        }
    ]
    
    results = []
    
    for scenario in scenarios:
        print(f"\n=== {scenario['name']} ===")
        print(f"Weather: fog={scenario['conditions']['input_avg_fog_density']}, rain={scenario['conditions']['input_avg_rain_rate']}")
        print(f"Temperature: {scenario['conditions']['input_avg_surface_temp']}°C")
        print(f"Requirements: power≥{scenario['constraints']['min_received_power']}dBm, reliability≥{scenario['conditions']['reliability_target']:.1%}")
        
        try:
            result = optimizer.optimize_deployment(scenario['conditions'], scenario['constraints'])
            
            print(f"✓ Optimization successful!")
            print(f"  TX Height: {result.get('height_tx', 'N/A'):.1f}m")
            print(f"  RX Height: {result.get('height_rx', 'N/A'):.1f}m")
            print(f"  TX Material: {result.get('material_tx', 'N/A')}")
            print(f"  RX Material: {result.get('material_rx', 'N/A')}")
            print(f"  Predicted Power: {result.get('predicted_power_dbm', 'N/A'):.1f} dBm")
            print(f"  Reliability: {result.get('estimated_reliability', 'N/A'):.1%}")
            print(f"  Score: {result.get('optimization_score', 'N/A'):.2f}")
            print(f"  Constraints Met: {result.get('constraints_met', 'N/A')}")
            print(f"  Configs Tested: {result.get('total_configurations_tested', 'N/A')}")
            
            results.append({
                'scenario': scenario['name'],
                'tx_height': result.get('height_tx', 0),
                'rx_height': result.get('height_rx', 0),
                'tx_material': result.get('material_tx', ''),
                'rx_material': result.get('material_rx', ''),
                'power': result.get('predicted_power_dbm', 0),
                'reliability': result.get('estimated_reliability', 0),
                'score': result.get('optimization_score', 0),
                'constraints_met': result.get('constraints_met', False)
            })
            
        except Exception as e:
            print(f"✗ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Analyze variability
    print(f"\n=== VARIABILITY ANALYSIS ===")
    if len(results) >= 2:
        tx_heights = [r['tx_height'] for r in results if r['constraints_met']]
        rx_heights = [r['rx_height'] for r in results if r['constraints_met']]
        powers = [r['power'] for r in results if r['constraints_met']]
        
        if tx_heights:
            print(f"TX Height range: {min(tx_heights):.1f}m - {max(tx_heights):.1f}m (variation: {max(tx_heights) - min(tx_heights):.1f}m)")
            print(f"RX Height range: {min(rx_heights):.1f}m - {max(rx_heights):.1f}m (variation: {max(rx_heights) - min(rx_heights):.1f}m)")
            print(f"Power range: {min(powers):.1f}dBm - {max(powers):.1f}dBm (variation: {max(powers) - min(powers):.1f}dB)")
            
            # Check if results are varied enough
            height_variation = max(tx_heights) - min(tx_heights) + max(rx_heights) - min(rx_heights)
            power_variation = max(powers) - min(powers)
            
            if height_variation > 10 and power_variation > 5:
                print("✓ Results show good variability - optimization is working correctly!")
                return True
            else:
                print("⚠ Results show limited variability - may indicate hardcoded values or poor model")
                return False
        else:
            print("✗ No valid results to analyze")
            return False
    else:
        print("✗ Insufficient results for variability analysis")
        return False

if __name__ == "__main__":
    print("Testing optimization result variability...")
    success = test_optimization_scenarios()
    if success:
        print("\n✓ Optimization system is working correctly with varied results!")
    else:
        print("\n✗ Optimization system may have issues with result variability.")
    sys.exit(0 if success else 1)
