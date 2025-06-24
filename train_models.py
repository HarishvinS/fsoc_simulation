#!/usr/bin/env python3
"""
Script to train and load prediction models for FSOC link optimization.
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

# Import our modules
from backend.optimizer.models import ModelManager, PowerPredictorModel
from backend.api.main import model_manager

def train_models():
    """Train and save prediction models."""
    logger.info("Starting model training process...")
    
    # Path to training data
    data_path = Path(__file__).parent / "backend" / "data" / "fsoc_training_dataset_500samples.csv"
    
    # Generate training data if it doesn't exist
    if not data_path.exists():
        logger.info("Training data not found, generating...")
        from backend.simulation.engine import create_training_dataset
        try:
            create_training_dataset(num_samples=500)
            logger.info("Training data generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate training data: {e}")
            return False
    
    if not data_path.exists():
        logger.error(f"Training data still not found at {data_path}")
        return False
    
    # Load training data
    logger.info(f"Loading training data from {data_path}")
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} training samples")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return False
    
    # Create model manager
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    manager = ModelManager(models_dir=str(models_dir))
    
    # Train models
    logger.info("Training prediction models...")
    try:
        # Train both random forest and XGBoost models
        results = manager.train_power_predictor(
            data, 
            model_types=["random_forest", "xgboost"]
        )
        
        # Log results
        for model_type, metrics in results.items():
            logger.info(f"{model_type.upper()} model metrics:")
            logger.info(f"  R² Score: {metrics.r2_score:.4f}")
            logger.info(f"  RMSE: {metrics.rmse:.4f} dB")
            logger.info(f"  MAE: {metrics.mae:.4f} dB")
            logger.info(f"  CV Score: {metrics.cv_score_mean:.4f} ± {metrics.cv_score_std:.4f}")
        
        logger.info("Model training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def load_models_into_manager():
    """Load trained models into the global model manager."""
    logger.info("Loading trained models into global model manager...")
    
    models_dir = Path(__file__).parent / "models"
    
    # Check if model files exist
    rf_model_path = models_dir / "power_predictor_random_forest.pkl"
    xgb_model_path = models_dir / "power_predictor_xgboost.pkl"
    
    if not rf_model_path.exists() or not xgb_model_path.exists():
        logger.error("Model files not found. Please train models first.")
        return False
    
    try:
        # Load random forest model
        rf_model = PowerPredictorModel("random_forest")
        rf_model.load_model(str(rf_model_path))
        model_manager.power_predictors["random_forest"] = rf_model
        
        # Load XGBoost model
        xgb_model = PowerPredictorModel("xgboost")
        xgb_model.load_model(str(xgb_model_path))
        model_manager.power_predictors["xgboost"] = xgb_model
        
        logger.info(f"Successfully loaded {len(model_manager.power_predictors)} models")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Check if we need to train models
    if "--train" in sys.argv:
        success = train_models()
        if not success:
            sys.exit(1)
    
    # Load models into the global model manager
    success = load_models_into_manager()
    if not success:
        sys.exit(1)
    
    # Print status
    print("\nModel Status:")
    print(f"  Models loaded: {len(model_manager.power_predictors)}")
    for model_type, model in model_manager.power_predictors.items():
        print(f"  - {model_type}: {'Trained' if model.is_trained else 'Not trained'}")
    
    print("\nOptimization feature is now ready to use!")