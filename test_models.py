#!/usr/bin/env python3
"""
Script to test if prediction models are loaded correctly.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Import our modules
from backend.optimizer.models import ModelManager

def test_models():
    """Test if models can be loaded and used for prediction."""
    print("Testing prediction models...")
    
    # Create model manager
    models_dir = Path(__file__).parent / "models"
    manager = ModelManager(models_dir=str(models_dir))
    
    # Try to load models
    try:
        # Load random forest model
        rf_model_path = models_dir / "power_predictor_random_forest.pkl"
        if rf_model_path.exists():
            print(f"Loading Random Forest model from {rf_model_path}")
            rf_model = manager.power_predictors.get("random_forest")
            if rf_model is None:
                from backend.optimizer.models import PowerPredictorModel
                rf_model = PowerPredictorModel("random_forest")
                rf_model.load_model(str(rf_model_path))
                manager.power_predictors["random_forest"] = rf_model
                print("  Random Forest model loaded successfully")
            else:
                print("  Random Forest model already loaded")
        else:
            print(f"  Error: Random Forest model file not found at {rf_model_path}")
            return False
        
        # Load XGBoost model
        xgb_model_path = models_dir / "power_predictor_xgboost.pkl"
        if xgb_model_path.exists():
            print(f"Loading XGBoost model from {xgb_model_path}")
            xgb_model = manager.power_predictors.get("xgboost")
            if xgb_model is None:
                from backend.optimizer.models import PowerPredictorModel
                xgb_model = PowerPredictorModel("xgboost")
                xgb_model.load_model(str(xgb_model_path))
                manager.power_predictors["xgboost"] = xgb_model
                print("  XGBoost model loaded successfully")
            else:
                print("  XGBoost model already loaded")
        else:
            print(f"  Error: XGBoost model file not found at {xgb_model_path}")
            return False
        
        # Test prediction with a sample input
        print("\nTesting prediction with sample input...")
        sample_input = {
            'input_height_tx': 30.0,
            'input_height_rx': 20.0,
            'input_fog_density': 0.2,
            'input_rain_rate': 1.0,
            'input_surface_temp': 25.0,
            'input_ambient_temp': 20.0,
            'input_wavelength_nm': 1550.0,
            'input_tx_power_dbm': 20.0,
            'link_distance_km': 1.5,
            'elevation_angle_deg': 0.0,
            'input_material_tx': 'aluminum',
            'input_material_rx': 'steel'
        }
        
        # Get best model
        best_model = manager.get_best_power_predictor()
        print(f"Using best model: {best_model.model_type}")
        
        # Make prediction
        prediction = best_model.predict(sample_input)
        print(f"Predicted received power: {prediction:.2f} dBm")
        
        # Create optimizer
        print("\nTesting optimizer creation...")
        optimizer = manager.create_deployment_optimizer()
        print("Optimizer created successfully")
        
        return True
        
    except Exception as e:
        print(f"Error testing models: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_models()
    if success:
        print("\nModel test completed successfully. Models are ready for use.")
    else:
        print("\nModel test failed. Please train models first using 'python train_models.py --train'")
        sys.exit(1)