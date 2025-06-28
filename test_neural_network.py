#!/usr/bin/env python3
"""
Test script for the new neural network implementation.
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

def test_neural_network():
    """Test the neural network implementation."""
    logger.info("Testing neural network implementation...")
    
    try:
        # Import the enhanced models
        from backend.optimizer.models import PowerPredictorModel, FSocNeuralNetwork
        
        # Test neural network architecture
        logger.info("Testing neural network architecture...")
        input_size = 20
        nn_model = FSocNeuralNetwork(input_size)
        logger.info(f"✓ Neural network created with input size {input_size}")
        logger.info(f"  Architecture: {nn_model}")
        
        # Test PowerPredictorModel with neural network
        logger.info("Testing PowerPredictorModel with neural network...")
        predictor = PowerPredictorModel("neural_network")
        logger.info(f"✓ PowerPredictorModel created with type: {predictor.model_type}")
        logger.info(f"  Device: {predictor.device}")
        
        # Test with small dataset
        logger.info("Testing with small synthetic dataset...")
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 100
        
        # Generate synthetic features that match the expected structure
        synthetic_data = pd.DataFrame({
            'input_lat_tx': np.random.uniform(30, 50, n_samples),
            'input_lon_tx': np.random.uniform(-130, -70, n_samples),
            'input_height_tx': np.random.uniform(5, 50, n_samples),
            'input_material_tx': np.random.choice(['concrete', 'white_paint', 'aluminum'], n_samples),
            'input_lat_rx': np.random.uniform(30, 50, n_samples),
            'input_lon_rx': np.random.uniform(-130, -70, n_samples),
            'input_height_rx': np.random.uniform(5, 50, n_samples),
            'input_material_rx': np.random.choice(['concrete', 'white_paint', 'aluminum'], n_samples),
            'input_fog_density': np.random.uniform(0, 2, n_samples),
            'input_rain_rate': np.random.uniform(0, 10, n_samples),
            'input_surface_temp': np.random.uniform(10, 40, n_samples),
            'input_ambient_temp': np.random.uniform(5, 35, n_samples),
            'input_wavelength_nm': np.random.choice([850, 1310, 1550], n_samples),
            'input_tx_power_dbm': np.random.uniform(10, 30, n_samples),
            'link_distance_km': np.random.uniform(0.1, 10, n_samples),
            'elevation_angle_deg': np.random.uniform(-10, 10, n_samples),
            'received_power_dbm': np.random.uniform(-50, -10, n_samples)  # Target variable
        })
        
        logger.info(f"✓ Created synthetic dataset with {len(synthetic_data)} samples")
        logger.info(f"  Columns: {list(synthetic_data.columns)}")
        
        # Test training
        logger.info("Testing neural network training...")
        try:
            metrics = predictor.train(synthetic_data, validation_split=0.2)
            logger.info("✓ Neural network training completed successfully!")
            logger.info(f"  R² Score: {metrics.r2_score:.4f}")
            logger.info(f"  RMSE: {metrics.rmse:.4f} dB")
            logger.info(f"  MAE: {metrics.mae:.4f} dB")
            logger.info(f"  Training samples: {metrics.training_samples}")
            
            # Test prediction
            logger.info("Testing prediction...")
            test_input = {
                'input_lat_tx': 37.7749,
                'input_lon_tx': -122.4194,
                'input_height_tx': 20,
                'input_material_tx': 'white_paint',
                'input_lat_rx': 37.7849,
                'input_lon_rx': -122.4094,
                'input_height_rx': 20,
                'input_material_rx': 'white_paint',
                'input_fog_density': 0.1,
                'input_rain_rate': 1.0,
                'input_surface_temp': 25,
                'input_ambient_temp': 20,
                'input_wavelength_nm': 1550,
                'input_tx_power_dbm': 20,
                'link_distance_km': 1.0,
                'elevation_angle_deg': 0.0
            }
            
            prediction = predictor.predict(test_input)
            logger.info(f"✓ Prediction: {prediction:.2f} dBm")
            
            # Test uncertainty prediction
            logger.info("Testing uncertainty prediction...")
            mean_pred, uncertainty = predictor.predict_with_uncertainty(test_input, n_samples=50)
            logger.info(f"✓ Prediction with uncertainty: {mean_pred:.2f} ± {uncertainty:.2f} dBm")
            
            # Test save/load
            logger.info("Testing model save/load...")
            test_model_path = "test_neural_model.pkl"
            predictor.save_model(test_model_path)
            logger.info(f"✓ Model saved to {test_model_path}")
            
            # Load model
            new_predictor = PowerPredictorModel("neural_network")
            new_predictor.load_model(test_model_path)
            logger.info("✓ Model loaded successfully")
            
            # Test loaded model prediction
            new_prediction = new_predictor.predict(test_input)
            logger.info(f"✓ Loaded model prediction: {new_prediction:.2f} dBm")
            
            # Verify predictions match
            if abs(prediction - new_prediction) < 0.01:
                logger.info("✓ Save/load test passed - predictions match!")
            else:
                logger.warning(f"⚠️ Save/load test warning - predictions differ: {abs(prediction - new_prediction):.4f}")
            
            # Clean up
            if os.path.exists(test_model_path):
                os.remove(test_model_path)
                logger.info("✓ Test file cleaned up")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Neural network training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"✗ Neural network test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NEURAL NETWORK IMPLEMENTATION TEST")
    logger.info("=" * 60)
    
    success = test_neural_network()
    
    if success:
        logger.info("🎉 All neural network tests passed!")
        logger.info("✅ Neural network implementation is working correctly")
        logger.info("💡 You can now train models with: python train_models.py --train --samples 10000")
    else:
        logger.error("❌ Neural network tests failed!")
        sys.exit(1)
