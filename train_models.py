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

# Import our modules will be done dynamically to avoid path issues

def train_models(num_samples: int = 10000):
    """Train and save prediction models."""
    logger.info("Starting enhanced model training process...")

    # Generate enhanced training data if needed
    data_path = Path(__file__).parent / "backend" / "data" / f"fsoc_training_dataset_{num_samples}samples.csv"

    if not data_path.exists() or num_samples > 5000:
        logger.info(f"Generating enhanced training dataset with {num_samples} samples...")
        try:
            # Add current directory to path for imports
            import sys
            sys.path.append(str(Path(__file__).parent))

            from backend.simulation.engine import create_training_dataset
            dataset_file = create_training_dataset(
                output_dir="backend/data",
                num_samples=num_samples,
                enhanced=True
            )
            data_path = Path(dataset_file)
            logger.info(f"Generated training dataset: {data_path}")
        except Exception as e:
            logger.error(f"Failed to generate training data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    # Load training data
    logger.info(f"Loading training data from {data_path}")
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} training samples")
        logger.info(f"Dataset shape: {data.shape}")
        logger.info(f"Features: {list(data.columns)}")
    except Exception as e:
        logger.error(f"Failed to load training data: {e}")
        return False
    
    # Create model manager
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    try:
        # Add current directory to path for imports
        import sys
        sys.path.append(str(Path(__file__).parent))

        from backend.optimizer.models import ModelManager
        manager = ModelManager(models_dir=str(models_dir))
    except Exception as e:
        logger.error(f"Failed to import ModelManager: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # Train models
    logger.info("Training prediction models...")
    try:
        # Train neural network, XGBoost, and random forest models
        results = manager.train_power_predictor(
            data,
            model_types=["neural_network", "xgboost", "random_forest"]
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

    try:
        # Add current directory to path for imports
        import sys
        sys.path.append(str(Path(__file__).parent))

        from backend.optimizer.models import ModelManager, PowerPredictorModel
        from backend.api.main import model_manager

        models_dir = Path(__file__).parent / "models"

        # Check if model files exist
        rf_model_path = models_dir / "power_predictor_random_forest.pkl"
        xgb_model_path = models_dir / "power_predictor_xgboost.pkl"

        if not rf_model_path.exists() or not xgb_model_path.exists():
            logger.error("Model files not found. Please train models first.")
            return False

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
        # Check for number of samples argument
        num_samples = 10000  # Default enhanced dataset size
        for i, arg in enumerate(sys.argv):
            if arg == "--samples" and i + 1 < len(sys.argv):
                try:
                    num_samples = int(sys.argv[i + 1])
                    logger.info(f"Using {num_samples} training samples")
                except ValueError:
                    logger.warning("Invalid samples argument, using default 10000")

        success = train_models(num_samples)
        if not success:
            sys.exit(1)
    
    # Load models into the global model manager
    success = load_models_into_manager()
    if not success:
        sys.exit(1)
    
    # Print status
    try:
        # Add current directory to path for imports
        import sys
        sys.path.append(str(Path(__file__).parent))

        from backend.api.main import model_manager

        print("\nModel Status:")
        print(f"  Models loaded: {len(model_manager.power_predictors)}")
        for model_type, model in model_manager.power_predictors.items():
            print(f"  - {model_type}: {'Trained' if model.is_trained else 'Not trained'}")

        print("\nOptimization feature is now ready to use!")
    except Exception as e:
        print(f"Could not check model status: {e}")
        print("\nTraining completed. Please restart the application to use the optimization feature.")