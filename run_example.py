#!/usr/bin/env python3
"""
Example script demonstrating FSOC link simulation and optimization.

This script shows how to use the FSOC optimization system for:
1. Single link simulation
2. Batch parameter studies
3. Model training
4. Deployment optimization
"""

import sys
import json
import time
from pathlib import Path
import pandas as pd

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.ingest.input_schema import EnvironmentInput, MaterialType
from backend.simulation.engine import FSocSimulationEngine, create_training_dataset
from backend.optimizer.models import ModelManager
from backend.ingest.mock_weather import MockWeatherAPI


def demo_single_simulation():
    """Demonstrate single link simulation."""
    print("=" * 60)
    print("DEMO 1: Single Link Simulation")
    print("=" * 60)
    
    # Create test configuration
    config = EnvironmentInput(
        lat_tx=37.7749, lon_tx=-122.4194,  # San Francisco
        lat_rx=37.7849, lon_rx=-122.4094,  # 1km away
        height_tx=25, height_rx=20,
        material_tx=MaterialType.WHITE_PAINT,
        material_rx=MaterialType.ALUMINUM,
        fog_density=0.8, rain_rate=3.0,
        surface_temp=28, ambient_temp=22,
        wavelength_nm=1550, tx_power_dbm=23
    )
    
    print(f"Configuration:")
    print(f"  Link distance: {config.link_distance_km():.2f} km")
    print(f"  Tx height: {config.height_tx} m ({config.material_tx.value})")
    print(f"  Rx height: {config.height_rx} m ({config.material_rx.value})")
    print(f"  Weather: {config.fog_density} g/m³ fog, {config.rain_rate} mm/hr rain")
    print(f"  Temperature: {config.surface_temp}°C surface, {config.ambient_temp}°C ambient")
    print()
    
    # Run simulation
    engine = FSocSimulationEngine()
    results = engine.simulate_single_link(config, detailed_output=False)
    
    # Display results
    print("Results:")
    print(f"  Received power: {results['received_power_dbm']:.1f} dBm")
    print(f"  Total loss: {results['total_loss_db']:.1f} dB")
    print(f"  Link margin: {results['link_margin_db']:.1f} dB")
    print(f"  Link available: {'YES' if results['link_available'] else 'NO'}")
    print(f"  Availability: {results['estimated_availability']:.1%}")
    print()
    
    print("Loss breakdown:")
    for loss_type, loss_value in results['loss_breakdown'].items():
        print(f"  {loss_type.replace('_', ' ').title()}: {loss_value:.2f} dB")
    print()
    
    print("Beam characteristics:")
    beam_info = results['beam_analysis']
    print(f"  Initial diameter: {beam_info['initial_diameter_m']:.3f} m")
    print(f"  Final diameter: {beam_info['final_diameter_m']:.3f} m")
    print(f"  Beam spreading: {beam_info['beam_spreading_ratio']:.2f}x")
    print(f"  Steering angle: {beam_info['total_steering_angle_mrad']:.2f} mrad")
    print()


def demo_parameter_study():
    """Demonstrate parameter study simulation."""
    print("=" * 60)
    print("DEMO 2: Parameter Study")
    print("=" * 60)
    
    # Base configuration
    base_config = EnvironmentInput(
        lat_tx=37.7749, lon_tx=-122.4194,
        lat_rx=37.7849, lon_rx=-122.4094,
        height_tx=20, height_rx=20,
        material_tx=MaterialType.WHITE_PAINT,
        material_rx=MaterialType.WHITE_PAINT,
        fog_density=0.5, rain_rate=1.0,
        surface_temp=25, ambient_temp=20,
        wavelength_nm=1550, tx_power_dbm=20
    )
    
    # Parameter ranges to study
    parameter_ranges = {
        'height_tx': [10, 15, 20, 25, 30, 40, 50],
        'fog_density': [0.0, 0.2, 0.5, 1.0, 2.0],
        'material_tx': ['white_paint', 'aluminum', 'black_paint']
    }
    
    print("Studying parameter effects:")
    print("  Heights: 10-50 m")
    print("  Fog density: 0-2 g/m³")
    print("  Materials: white paint, aluminum, black paint")
    print()
    
    # Run batch simulation
    engine = FSocSimulationEngine()
    print("Running simulations...")
    results_df = engine.batch_simulate(parameter_ranges, base_config, max_samples=50)
    
    print(f"Generated {len(results_df)} simulation results")
    print()
    
    # Analyze results
    print("Parameter effects on received power:")
    
    # Height effect
    height_effect = results_df.groupby('input_height_tx')['received_power_dbm'].mean()
    print("\nHeight vs Power:")
    for height, power in height_effect.items():
        print(f"  {height:2.0f} m: {power:6.1f} dBm")
    
    # Fog effect
    fog_effect = results_df.groupby('input_fog_density')['received_power_dbm'].mean()
    print("\nFog Density vs Power:")
    for fog, power in fog_effect.items():
        print(f"  {fog:3.1f} g/m³: {power:6.1f} dBm")
    
    # Material effect
    material_effect = results_df.groupby('input_material_tx')['received_power_dbm'].mean()
    print("\nMaterial vs Power:")
    for material, power in material_effect.items():
        print(f"  {material:12s}: {power:6.1f} dBm")
    print()


def demo_model_training():
    """Demonstrate ML model training."""
    print("=" * 60)
    print("DEMO 3: Machine Learning Model Training")
    print("=" * 60)
    
    print("Generating training dataset...")
    
    # Create training dataset
    dataset_file = create_training_dataset(num_samples=500)  # Small for demo
    print(f"Created dataset: {dataset_file}")
    
    # Load and examine dataset
    training_data = pd.read_csv(dataset_file)
    print(f"Dataset shape: {training_data.shape}")
    print(f"Columns: {list(training_data.columns)}")
    print()
    
    # Train models
    print("Training machine learning models...")
    manager = ModelManager()
    
    training_results = manager.train_power_predictor(
        training_data,
        model_types=["xgboost", "random_forest"]
    )
    
    print("\nModel Performance:")
    for model_type, metrics in training_results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  R² Score: {metrics.r2_score:.4f}")
        print(f"  RMSE: {metrics.rmse:.3f} dB")
        print(f"  MAE: {metrics.mae:.3f} dB")
        print(f"  CV Score: {metrics.cv_score_mean:.4f} ± {metrics.cv_score_std:.4f}")
    
    # Get best model and show feature importance
    best_model = manager.get_best_power_predictor()
    feature_importance = best_model.get_feature_importance()
    
    print(f"\nBest Model: {best_model.model_type}")
    print("Top 5 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
        print(f"  {i+1}. {feature}: {importance:.3f}")
    print()
    
    return manager


def demo_optimization(manager: ModelManager):
    """Demonstrate deployment optimization."""
    print("=" * 60)
    print("DEMO 4: Deployment Optimization")
    print("=" * 60)
    
    print("Finding optimal deployment parameters...")
    print()
    
    # Create optimizer
    optimizer = manager.create_deployment_optimizer()
    
    # Define optimization scenario
    base_conditions = {
        'input_lat_tx': 37.7749,
        'input_lon_tx': -122.4194,
        'input_lat_rx': 37.7849,
        'input_lon_rx': -122.4094,
        'input_fog_density': 0.5,
        'input_rain_rate': 2.0,
        'input_surface_temp': 25,
        'input_ambient_temp': 20,
        'input_wavelength_nm': 1550,
        'input_tx_power_dbm': 20,
    }
    
    constraints = {
        'min_height': 10,
        'max_height': 50,
        'available_materials': ['white_paint', 'aluminum', 'steel'],
        'min_received_power': -25
    }
    
    print("Optimization constraints:")
    print(f"  Height range: {constraints['min_height']}-{constraints['max_height']} m")
    print(f"  Available materials: {', '.join(constraints['available_materials'])}")
    print(f"  Minimum power: {constraints['min_received_power']} dBm")
    print()
    
    # Run optimization
    recommendations = optimizer.optimize_deployment(
        base_conditions, constraints, "max_power"
    )
    
    print("OPTIMIZATION RESULTS:")
    print(f"  Available keys: {list(recommendations.keys())}")
    if 'height_tx' in recommendations:
        print(f"  Optimal Tx height: {recommendations['height_tx']:.1f} m")
        print(f"  Optimal Rx height: {recommendations['height_rx']:.1f} m")
        print(f"  Optimal Tx material: {recommendations['material_tx']}")
        print(f"  Optimal Rx material: {recommendations['material_rx']}")
    print(f"  Predicted power: {recommendations['predicted_power_dbm']:.1f} dBm")
    print(f"  Optimization score: {recommendations['optimization_score']:.2f}")
    print()


def demo_weather_api():
    """Demonstrate weather API functionality."""
    print("=" * 60)
    print("DEMO 5: Weather API")
    print("=" * 60)
    
    api = MockWeatherAPI()
    
    # Test locations
    locations = {
        "San Francisco": (37.7749, -122.4194),
        "New York": (40.7128, -74.0060),
        "London": (51.5074, -0.1278),
        "Singapore": (1.3521, 103.8198)
    }
    
    print("Current weather conditions:")
    print()
    
    for city, (lat, lon) in locations.items():
        weather = api.get_current_weather(lat, lon)
        print(f"{city:12s}: Fog={weather.fog_density:4.1f} g/m³, "
              f"Rain={weather.rain_rate:4.1f} mm/hr, "
              f"Temp={weather.surface_temp:4.1f}°C/{weather.ambient_temp:4.1f}°C")
    print()


def main():
    """Run all demonstration examples."""
    print("FSOC Link Optimization System - Demo")
    print("====================================")
    print()
    
    start_time = time.time()
    
    try:
        # Run demos
        demo_single_simulation()
        demo_parameter_study()
        manager = demo_model_training()
        demo_optimization(manager)
        demo_weather_api()
        
        # Summary
        total_time = time.time() - start_time
        print("=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total execution time: {total_time:.1f} seconds")
        print()
        print("Next steps:")
        print("1. Start the API server: python -m backend.api.main")
        print("2. Visit http://localhost:8000/docs for interactive API")
        print("3. Use the trained models for real-time optimization")
        print()
        
    except Exception as e:
        print(f"DEMO FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())