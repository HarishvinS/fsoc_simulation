#!/usr/bin/env python3
"""
Test script to simulate the web interface optimization workflow.
This tests the same logic that the web interface uses.
"""

import sys
sys.path.append('.')

from backend.optimizer.models import ModelManager, PowerPredictorModel
from backend.ingest.input_schema import OptimizationRequest, MaterialType
from pathlib import Path
import time

def load_models():
    """Load trained models like the web interface does."""
    manager = ModelManager(models_dir='models')
    
    # Load all available models
    models_dir = Path('models')
    model_files = list(models_dir.glob('power_predictor_*.pkl'))
    
    for model_file in model_files:
        model_type = model_file.name.replace('power_predictor_', '').replace('.pkl', '')
        try:
            model = PowerPredictorModel(model_type)
            model.load_model(str(model_file))
            manager.power_predictors[model_type] = model
            print(f"Loaded {model_type} model (R² Score: {model.metrics.r2_score:.4f})")
        except Exception as e:
            print(f"Failed to load {model_type}: {e}")
    
    print(f"Total models loaded: {len(manager.power_predictors)}")
    return manager

def simulate_optimization_request(manager, request_data):
    """Simulate the optimization request processing like the web interface."""
    try:
        # Create optimization request (like the web form does)
        request = OptimizationRequest(**request_data)
        
        # Create deployment optimizer
        optimizer = manager.create_deployment_optimizer()
        
        # Prepare base conditions (like the API does)
        base_conditions = {
            'input_lat_tx': request.lat_tx,
            'input_lon_tx': request.lon_tx,
            'input_lat_rx': request.lat_rx,
            'input_lon_rx': request.lon_rx,
            'input_avg_fog_density': request.avg_fog_density,
            'input_avg_rain_rate': request.avg_rain_rate,
            'input_avg_surface_temp': request.avg_surface_temp,
            'input_avg_ambient_temp': request.avg_ambient_temp,
            'input_wavelength_nm': 1550,  # Default
            'input_tx_power_dbm': 20,     # Default
            'reliability_target': request.reliability_target
        }
        
        # Prepare constraints
        constraints = {
            'min_height': request.min_height,
            'max_height': request.max_height,
            'available_materials': [m.value for m in request.available_materials],
            'min_received_power': request.min_received_power_dbm
        }
        
        # Add constraints to base conditions for optimization algorithm
        base_conditions['min_received_power'] = request.min_received_power_dbm
        
        # Run optimization
        start_time = time.time()
        recommendations = optimizer.optimize_deployment(
            base_conditions, constraints, "max_power"
        )
        optimization_time = time.time() - start_time
        
        # Calculate confidence score (like the API does)
        constraints_met = recommendations.get('constraints_met', False)
        predicted_power = recommendations.get('predicted_power_dbm', -50)
        
        if not constraints_met:
            confidence_score = 0.2
        else:
            if 'confidence_level' in recommendations:
                confidence_score = float(recommendations['confidence_level'])
            else:
                # Fallback confidence calculation
                power_factor = max(0.0, min(1.0, (predicted_power + 50) / 40))
                reliability_factor = recommendations.get('estimated_reliability', 0.5)
                confidence_score = (power_factor * 0.6 + reliability_factor * 0.4)
        
        confidence_score = max(0.1, min(1.0, confidence_score))
        
        return {
            'success': True,
            'recommendations': recommendations,
            'confidence_score': confidence_score,
            'optimization_time': optimization_time
        }
        
    except Exception as e:
        return {
            'success': False,
            'error_message': str(e),
            'optimization_time': 0
        }

def test_web_interface_scenarios():
    """Test scenarios that users might submit through the web interface."""
    manager = load_models()
    if not manager.power_predictors:
        print("No models loaded - cannot test optimization")
        return False
    
    # Test scenarios representing different user inputs
    test_scenarios = [
        {
            'name': 'San Francisco - Good Weather',
            'data': {
                'lat_tx': 37.7749,
                'lon_tx': -122.4194,
                'lat_rx': 37.7849,
                'lon_rx': -122.4094,
                'avg_fog_density': 0.1,
                'avg_rain_rate': 2.0,
                'avg_surface_temp': 25.0,
                'avg_ambient_temp': 20.0,
                'min_height': 5.0,
                'max_height': 50.0,
                'available_materials': [MaterialType.WHITE_PAINT, MaterialType.ALUMINUM, MaterialType.STEEL],
                'min_received_power_dbm': -30.0,
                'reliability_target': 0.99
            }
        },
        {
            'name': 'New York - Heavy Rain',
            'data': {
                'lat_tx': 40.7128,
                'lon_tx': -74.0060,
                'lat_rx': 40.7589,
                'lon_rx': -73.9851,
                'avg_fog_density': 0.5,
                'avg_rain_rate': 15.0,
                'avg_surface_temp': 20.0,
                'avg_ambient_temp': 15.0,
                'min_height': 10.0,
                'max_height': 100.0,
                'available_materials': [MaterialType.ALUMINUM, MaterialType.STEEL],
                'min_received_power_dbm': -25.0,
                'reliability_target': 0.95
            }
        },
        {
            'name': 'London - Foggy Conditions',
            'data': {
                'lat_tx': 51.5074,
                'lon_tx': -0.1278,
                'lat_rx': 51.5174,
                'lon_rx': -0.1178,
                'avg_fog_density': 2.0,
                'avg_rain_rate': 5.0,
                'avg_surface_temp': 15.0,
                'avg_ambient_temp': 10.0,
                'min_height': 15.0,
                'max_height': 75.0,
                'available_materials': [MaterialType.WHITE_PAINT, MaterialType.STEEL],
                'min_received_power_dbm': -20.0,
                'reliability_target': 0.999
            }
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n=== Testing: {scenario['name']} ===")
        
        result = simulate_optimization_request(manager, scenario['data'])
        
        if result['success']:
            recs = result['recommendations']
            print(f"✓ Optimization successful in {result['optimization_time']:.2f}s")
            print(f"  TX Height: {recs.get('height_tx', 'N/A'):.1f}m")
            print(f"  RX Height: {recs.get('height_rx', 'N/A'):.1f}m")
            print(f"  TX Material: {recs.get('material_tx', 'N/A')}")
            print(f"  RX Material: {recs.get('material_rx', 'N/A')}")
            print(f"  Predicted Power: {recs.get('predicted_power_dbm', 'N/A'):.1f} dBm")
            print(f"  Confidence: {result['confidence_score']:.1%}")
            print(f"  Constraints Met: {recs.get('constraints_met', 'N/A')}")
            
            results.append({
                'scenario': scenario['name'],
                'success': True,
                'tx_height': recs.get('height_tx', 0),
                'rx_height': recs.get('height_rx', 0),
                'power': recs.get('predicted_power_dbm', 0),
                'confidence': result['confidence_score'],
                'time': result['optimization_time']
            })
        else:
            print(f"✗ Optimization failed: {result['error_message']}")
            results.append({
                'scenario': scenario['name'],
                'success': False
            })
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    
    print(f"\n=== SUMMARY ===")
    print(f"Successful optimizations: {len(successful_results)}/{len(results)}")
    
    if len(successful_results) >= 2:
        heights = [r['tx_height'] + r['rx_height'] for r in successful_results]
        powers = [r['power'] for r in successful_results]
        confidences = [r['confidence'] for r in successful_results]
        times = [r['time'] for r in successful_results]
        
        print(f"Height variation: {max(heights) - min(heights):.1f}m")
        print(f"Power variation: {max(powers) - min(powers):.1f}dB")
        print(f"Average confidence: {sum(confidences)/len(confidences):.1%}")
        print(f"Average optimization time: {sum(times)/len(times):.2f}s")
        
        # Check if results are realistic and varied
        if (max(heights) - min(heights) > 10 and 
            max(powers) - min(powers) > 3 and 
            sum(confidences)/len(confidences) > 0.5):
            print("✓ Web interface simulation successful - results are varied and realistic!")
            return True
        else:
            print("⚠ Results may not be sufficiently varied or realistic")
            return False
    else:
        print("✗ Insufficient successful results for analysis")
        return False

if __name__ == "__main__":
    print("Testing web interface optimization workflow...")
    success = test_web_interface_scenarios()
    if success:
        print("\n✓ Web interface optimization is working correctly!")
    else:
        print("\n✗ Web interface optimization may have issues.")
    sys.exit(0 if success else 1)
