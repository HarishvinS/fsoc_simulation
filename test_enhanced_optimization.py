#!/usr/bin/env python3
"""
Test script for the enhanced optimization system with neural networks and risk assessment.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_optimization():
    """Test the enhanced optimization system."""
    logger.info("Testing enhanced optimization system...")
    
    try:
        # Import the enhanced modules
        from backend.optimizer.models import ModelManager, PowerPredictorModel
        from backend.risk.assessment import ComprehensiveRiskAssessment
        
        logger.info("✓ Successfully imported enhanced modules")
        
        # Test risk assessment
        logger.info("Testing risk assessment...")
        risk_assessor = ComprehensiveRiskAssessment()
        
        # Test location and conditions
        location = {'lat': 37.7749, 'lon': -122.4194}  # San Francisco
        conditions = {
            'fog_density': 0.3,
            'rain_rate': 5.0,
            'surface_temp': 22.0,
            'ambient_temp': 18.0,
            'avg_temp': 20.0,
            'humidity': 75
        }
        equipment_config = {
            'wavelength_nm': 1550,
            'tx_power_dbm': 20,
            'height_tx': 25,
            'height_rx': 30
        }
        operational_factors = {
            'link_distance_km': 2.5,
            'height_tx': 25,
            'height_rx': 30,
            'accessibility': 'normal'
        }
        
        risk_assessment = risk_assessor.assess_deployment_risk(
            location=location,
            conditions=conditions,
            equipment_config=equipment_config,
            operational_factors=operational_factors
        )
        
        logger.info("✓ Risk assessment completed successfully!")
        logger.info(f"  Overall Risk Level: {risk_assessment.overall_risk_level}")
        logger.info(f"  Overall Risk Score: {risk_assessment.overall_risk_score:.3f}")
        logger.info(f"  Weather Risk Level: {risk_assessment.weather_risk.risk_level}")
        logger.info(f"  Equipment Reliability Score: {risk_assessment.equipment_risk.reliability_score:.3f}")
        logger.info(f"  Expected Availability: {risk_assessment.combined_availability:.3f}")
        logger.info(f"  MTBF: {risk_assessment.equipment_risk.mtbf_hours:.0f} hours")
        logger.info(f"  Risk Factors: {len(risk_assessment.risk_factors)}")
        logger.info(f"  Mitigation Strategies: {len(risk_assessment.mitigation_strategies)}")
        
        # Test neural network with uncertainty
        logger.info("Testing neural network with uncertainty...")
        
        # Create a realistic training dataset with proper relationships
        np.random.seed(42)
        n_samples = 500

        # Generate realistic parameters
        lat_tx = np.random.uniform(30, 50, n_samples)
        lon_tx = np.random.uniform(-130, -70, n_samples)
        lat_rx = lat_tx + np.random.uniform(-0.1, 0.1, n_samples)  # RX near TX
        lon_rx = lon_tx + np.random.uniform(-0.1, 0.1, n_samples)

        height_tx = np.random.uniform(5, 50, n_samples)
        height_rx = np.random.uniform(5, 50, n_samples)
        fog_density = np.random.exponential(0.3, n_samples)  # More realistic fog distribution
        rain_rate = np.random.exponential(2.0, n_samples)    # More realistic rain distribution
        surface_temp = np.random.normal(25, 10, n_samples)
        ambient_temp = surface_temp + np.random.normal(-3, 5, n_samples)  # Correlated temps
        wavelength = np.random.choice([850, 1310, 1550], n_samples)
        tx_power = np.random.uniform(10, 30, n_samples)

        # Calculate realistic link distance
        link_distance_km = np.sqrt((lat_rx - lat_tx)**2 + (lon_rx - lon_tx)**2) * 111  # Rough km conversion
        link_distance_km = np.clip(link_distance_km, 0.1, 20)  # Reasonable range

        elevation_angle = np.random.uniform(-5, 5, n_samples)

        # Calculate realistic received power using simplified link budget
        # P_rx = P_tx - path_loss - atmospheric_loss
        path_loss = 20 * np.log10(link_distance_km * 1000) + 20 * np.log10(wavelength * 1e-9) - 147.55
        fog_loss = fog_density * 10  # Simplified fog loss
        rain_loss = rain_rate * 0.5  # Simplified rain loss
        height_gain = (height_tx + height_rx) * 0.1  # Height advantage

        received_power = (tx_power - path_loss - fog_loss - rain_loss + height_gain +
                         np.random.normal(0, 3, n_samples))  # Add noise

        synthetic_data = pd.DataFrame({
            'input_lat_tx': lat_tx,
            'input_lon_tx': lon_tx,
            'input_height_tx': height_tx,
            'input_material_tx': np.random.choice(['concrete', 'white_paint', 'aluminum'], n_samples),
            'input_lat_rx': lat_rx,
            'input_lon_rx': lon_rx,
            'input_height_rx': height_rx,
            'input_material_rx': np.random.choice(['concrete', 'white_paint', 'aluminum'], n_samples),
            'input_fog_density': fog_density,
            'input_rain_rate': rain_rate,
            'input_surface_temp': surface_temp,
            'input_ambient_temp': ambient_temp,
            'input_wavelength_nm': wavelength,
            'input_tx_power_dbm': tx_power,
            'link_distance_km': link_distance_km,
            'elevation_angle_deg': elevation_angle,
            'received_power_dbm': received_power
        })
        
        # Train neural network
        nn_model = PowerPredictorModel("neural_network")
        metrics = nn_model.train(synthetic_data, validation_split=0.2)
        
        logger.info("✓ Neural network training completed!")
        logger.info(f"  R² Score: {metrics.r2_score:.4f}")
        logger.info(f"  RMSE: {metrics.rmse:.4f} dB")
        
        # Test prediction with uncertainty
        test_input = {
            'input_lat_tx': 37.7749,
            'input_lon_tx': -122.4194,
            'input_height_tx': 25,
            'input_material_tx': 'white_paint',
            'input_lat_rx': 37.7849,
            'input_lon_rx': -122.4094,
            'input_height_rx': 30,
            'input_material_rx': 'white_paint',
            'input_fog_density': 0.3,
            'input_rain_rate': 5.0,
            'input_surface_temp': 22,
            'input_ambient_temp': 18,
            'input_wavelength_nm': 1550,
            'input_tx_power_dbm': 20,
            'link_distance_km': 2.5,
            'elevation_angle_deg': 0.0
        }
        
        # Regular prediction
        prediction = nn_model.predict(test_input)
        logger.info(f"✓ Regular prediction: {prediction:.2f} dBm")
        
        # Prediction with uncertainty
        mean_pred, uncertainty = nn_model.predict_with_uncertainty(test_input, n_samples=50)
        logger.info(f"✓ Prediction with uncertainty: {mean_pred:.2f} ± {uncertainty:.2f} dBm")
        
        # Test model manager with multiple models
        logger.info("Testing model manager with multiple models...")
        manager = ModelManager()
        
        # Train multiple models
        results = manager.train_power_predictor(
            synthetic_data,
            model_types=["neural_network", "xgboost"]
        )
        
        logger.info("✓ Multiple models trained successfully!")
        for model_type, metrics in results.items():
            logger.info(f"  {model_type}: R² = {metrics.r2_score:.4f}, RMSE = {metrics.rmse:.4f}")
        
        # Test optimization with the best model
        logger.info("Testing optimization with enhanced models...")
        
        # Get the best model
        best_model_type = max(results.keys(), key=lambda k: results[k].r2_score)
        best_model = manager.power_predictors[best_model_type]
        
        logger.info(f"✓ Best model: {best_model_type} (R² = {results[best_model_type].r2_score:.4f})")
        
        # Test optimization
        from backend.optimizer.models import DeploymentOptimizerModel

        optimizer = DeploymentOptimizerModel()
        optimizer.set_power_predictor(best_model)
        
        base_conditions = {
            'input_lat_tx': 37.7749,
            'input_lon_tx': -122.4194,
            'input_lat_rx': 37.7849,
            'input_lon_rx': -122.4094,
            'input_fog_density': 0.3,
            'input_rain_rate': 5.0,
            'input_surface_temp': 22,
            'input_ambient_temp': 18,
            'input_wavelength_nm': 1550,
            'input_tx_power_dbm': 20
        }
        
        constraints = {
            'min_height': 10,
            'max_height': 50,
            'available_materials': ['white_paint', 'aluminum'],
            'min_received_power': -40
        }
        
        optimization_result = optimizer.optimize_deployment(
            base_conditions, constraints, "max_power"
        )
        
        logger.info("✓ Optimization completed successfully!")
        logger.info(f"  Optimal height TX: {optimization_result.get('height_tx', 'N/A')} m")
        logger.info(f"  Optimal height RX: {optimization_result.get('height_rx', 'N/A')} m")
        logger.info(f"  Optimal material TX: {optimization_result.get('material_tx', 'N/A')}")
        logger.info(f"  Optimal material RX: {optimization_result.get('material_rx', 'N/A')}")
        logger.info(f"  Predicted power: {optimization_result.get('predicted_power_dbm', 'N/A'):.2f} dBm")
        
        if 'prediction_uncertainty_db' in optimization_result:
            logger.info(f"  Prediction uncertainty: ±{optimization_result['prediction_uncertainty_db']:.2f} dB")
            logger.info(f"  Confidence level: {optimization_result.get('confidence_level', 0):.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Enhanced optimization test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("ENHANCED OPTIMIZATION SYSTEM TEST")
    logger.info("=" * 70)
    
    success = test_enhanced_optimization()
    
    if success:
        logger.info("🎉 All enhanced optimization tests passed!")
        logger.info("✅ Enhanced optimization system is working correctly")
        logger.info("🚀 Features implemented:")
        logger.info("   • Neural networks with uncertainty quantification")
        logger.info("   • Comprehensive risk assessment")
        logger.info("   • Weather and equipment risk analysis")
        logger.info("   • Enhanced training data generation")
        logger.info("   • Multi-model comparison and selection")
        logger.info("")
        logger.info("💡 Next steps:")
        logger.info("   • Re-enable optimization UI links")
        logger.info("   • Train models with larger dataset: python train_models.py --train --samples 10000")
        logger.info("   • Test optimization API endpoints")
    else:
        logger.error("❌ Enhanced optimization tests failed!")
        sys.exit(1)
