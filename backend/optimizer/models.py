"""
Machine learning models for FSOC link optimization.

Implements two main model types:
1. PowerPredictorModel - Predicts received power given environmental conditions
2. DeploymentOptimizerModel - Suggests optimal deployment parameters

Models are trained on simulation data and provide fast inference
for real-time deployment optimization.
"""

import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

# ML libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    r2_score: float
    rmse: float
    mae: float
    cv_score_mean: float
    cv_score_std: float
    training_samples: int


class PowerPredictorModel:
    """
    Predicts received optical power based on environmental and system parameters.
    
    This model serves as the core performance predictor, learning from
    physics-based simulation data to provide fast inference.
    """
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize power prediction model.
        
        Args:
            model_type: Type of ML model ("random_forest", "xgboost", "lightgbm")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = "received_power_dbm"
        self.metrics = None
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for training or prediction.
        
        Args:
            data: Raw simulation data
            
        Returns:
            Processed feature matrix
        """
        # Select relevant input features
        feature_columns = [
            'input_height_tx', 'input_height_rx',
            'input_fog_density', 'input_rain_rate',
            'input_surface_temp', 'input_ambient_temp',
            'input_wavelength_nm', 'input_tx_power_dbm',
            'link_distance_km', 'elevation_angle_deg'
        ]
        
        # Material features (categorical)
        categorical_features = ['input_material_tx', 'input_material_rx']
        
        # Start with numerical features
        features = data[feature_columns].copy()
        
        # Handle categorical features
        for cat_feature in categorical_features:
            if cat_feature in data.columns:
                if not self.is_trained:
                    # Training: fit label encoder
                    if cat_feature not in self.label_encoders:
                        self.label_encoders[cat_feature] = LabelEncoder()
                    features[cat_feature] = self.label_encoders[cat_feature].fit_transform(
                        data[cat_feature].astype(str)
                    )
                else:
                    # Prediction: use existing encoder
                    if cat_feature in self.label_encoders:
                        # Handle unseen categories
                        known_categories = set(self.label_encoders[cat_feature].classes_)
                        data_categories = data[cat_feature].astype(str)
                        
                        # Replace unseen categories with most common one
                        most_common = self.label_encoders[cat_feature].classes_[0]
                        data_categories = data_categories.apply(
                            lambda x: x if x in known_categories else most_common
                        )
                        features[cat_feature] = self.label_encoders[cat_feature].transform(data_categories)
                    else:
                        features[cat_feature] = 0  # Default value
        
        # Add engineered features
        features = self._add_engineered_features(features)
        
        return features
    
    def _add_engineered_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features based on domain knowledge."""
        # Thermal gradient magnitude
        features['thermal_gradient'] = abs(features['input_surface_temp'] - features['input_ambient_temp'])
        
        # Total atmospheric loading
        features['atmospheric_loading'] = features['input_fog_density'] + features['input_rain_rate'] * 0.1
        
        # Height differential
        features['height_differential'] = abs(features['input_height_tx'] - features['input_height_rx'])
        
        # Path loss indicator (distance and elevation combined)
        features['path_complexity'] = features['link_distance_km'] * (1 + abs(features['elevation_angle_deg']) / 90)
        
        # Wavelength-dependent scattering indicator
        features['scattering_potential'] = features['input_fog_density'] * (1550 / features['input_wavelength_nm']) ** 1.3
        
        return features
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2) -> ModelMetrics:
        """
        Train the power prediction model.
        
        Args:
            data: Training data from simulations
            validation_split: Fraction of data for validation
            
        Returns:
            Model performance metrics
        """
        print(f"Training {self.model_type} power prediction model...")
        
        # Prepare features and target
        X = self.prepare_features(data)
        y = data[self.target_name]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        if self.model_type in ["random_forest"]:
            # Tree-based models don't need scaling
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
        else:
            # Gradient boosting models
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='r2')
        
        self.metrics = ModelMetrics(
            r2_score=r2,
            rmse=rmse,
            mae=mae,
            cv_score_mean=cv_scores.mean(),
            cv_score_std=cv_scores.std(),
            training_samples=len(X_train)
        )
        
        self.is_trained = True
        
        print(f"Model training completed:")
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f} dB")
        print(f"  MAE: {mae:.4f} dB")
        print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.metrics
    
    def predict(self, input_data: Union[Dict, pd.DataFrame]) -> Union[float, np.ndarray]:
        """
        Predict received power for given conditions.
        
        Args:
            input_data: Environmental and system parameters
            
        Returns:
            Predicted received power in dBm
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Convert single prediction to DataFrame
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Prepare features
        features = self.prepare_features(input_data)
        
        # Ensure feature order matches training
        if set(features.columns) != set(self.feature_names):
            missing_features = set(self.feature_names) - set(features.columns)
            extra_features = set(features.columns) - set(self.feature_names)
            
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            if extra_features:
                features = features[self.feature_names]
        
        # Scale features if needed
        if self.model_type in ["xgboost", "lightgbm"]:
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)
        else:
            prediction = self.model.predict(features)
        
        return prediction[0] if len(prediction) == 1 else prediction
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'target_name': self.target_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.metrics = model_data['metrics']
        self.target_name = model_data['target_name']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


class DeploymentOptimizerModel:
    """
    Suggests optimal deployment parameters for FSOC links.
    
    Uses reinforcement learning concepts to find parameter combinations
    that maximize link performance under given constraints.
    """
    
    def __init__(self):
        self.power_predictor = None
        self.optimization_history = []
        self.parameter_bounds = {}
        
    def set_power_predictor(self, predictor: PowerPredictorModel):
        """Set the power prediction model for optimization."""
        self.power_predictor = predictor
    
    def optimize_deployment(self,
                          base_conditions: Dict,
                          constraints: Dict,
                          optimization_target: str = "max_power") -> Dict:
        """
        Optimize deployment parameters for given conditions and constraints.
        
        Args:
            base_conditions: Fixed environmental conditions
            constraints: Parameter bounds and requirements
            optimization_target: Optimization objective
            
        Returns:
            Optimal parameter recommendations
        """
        if self.power_predictor is None:
            raise ValueError("Power predictor model must be set first")
        
        # Define parameter search space
        param_space = self._create_parameter_space(constraints)
        
        # Grid search or optimization algorithm
        best_params = self._grid_search_optimization(
            base_conditions, param_space, optimization_target
        )
        
        return best_params
    
    def _create_parameter_space(self, constraints: Dict) -> Dict:
        """Create parameter search space from constraints."""
        param_space = {}
        
        # Height parameters
        height_min = constraints.get('min_height', 5)
        height_max = constraints.get('max_height', 100)
        param_space['height_tx'] = np.linspace(height_min, height_max, 10)
        param_space['height_rx'] = np.linspace(height_min, height_max, 10)
        
        # Material options
        available_materials = constraints.get('available_materials', 
                                            ['white_paint', 'aluminum', 'steel'])
        param_space['material_tx'] = available_materials
        param_space['material_rx'] = available_materials
        
        return param_space
    
    def _grid_search_optimization(self,
                                base_conditions: Dict,
                                param_space: Dict,
                                target: str) -> Dict:
        """Perform grid search optimization."""
        from itertools import product
        import math
        
        best_score = -float('inf')
        best_params = {}
        
        # Get parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        
        # Calculate link distance and elevation angle from coordinates
        lat1, lon1 = base_conditions['input_lat_tx'], base_conditions['input_lon_tx']
        lat2, lon2 = base_conditions['input_lat_rx'], base_conditions['input_lon_rx']
        
        # Calculate great circle distance
        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        link_distance_km = 6371.0 * c
        
        for combination in product(*param_values):
            # Create test configuration
            test_config = base_conditions.copy()
            for name, value in zip(param_names, combination):
                test_config[f'input_{name}'] = value
            
            # Add required derived features
            test_config['link_distance_km'] = link_distance_km
            
            # Calculate elevation angle
            height_diff = test_config.get('input_height_tx', 20) - test_config.get('input_height_rx', 20)
            elevation_angle_deg = math.degrees(math.atan2(height_diff, link_distance_km * 1000))
            test_config['elevation_angle_deg'] = elevation_angle_deg
            
            # Predict performance
            try:
                predicted_power = self.power_predictor.predict(test_config)
                
                # Calculate score based on target
                if target == "max_power":
                    score = predicted_power
                elif target == "min_cost":
                    # Simple cost model (height = cost)
                    height_cost = (test_config.get('input_height_tx', 20) + 
                                 test_config.get('input_height_rx', 20)) / 2
                    score = predicted_power - height_cost * 0.1
                else:
                    score = predicted_power
                
                if score > best_score:
                    best_score = score
                    best_params = {name: value for name, value in zip(param_names, combination)}
                    best_params['predicted_power_dbm'] = predicted_power
                    best_params['optimization_score'] = score
                    
            except Exception as e:
                print(f"Optimization failed for combination {dict(zip(param_names, combination))}: {e}")
                continue
        
        return best_params


# Model factory and management
class ModelManager:
    """Manages multiple ML models for FSOC optimization."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.power_predictors = {}
        self.optimizers = {}
    
    def train_power_predictor(self, 
                            training_data: pd.DataFrame,
                            model_types: List[str] = ["xgboost", "random_forest"]) -> Dict[str, ModelMetrics]:
        """Train multiple power prediction models and compare performance."""
        results = {}
        
        for model_type in model_types:
            print(f"\nTraining {model_type} model...")
            
            model = PowerPredictorModel(model_type)
            metrics = model.train(training_data)
            
            # Save model
            model_path = self.models_dir / f"power_predictor_{model_type}.pkl"
            model.save_model(str(model_path))
            
            self.power_predictors[model_type] = model
            results[model_type] = metrics
        
        return results
    
    def get_best_power_predictor(self) -> PowerPredictorModel:
        """Get the best performing power prediction model."""
        if not self.power_predictors:
            raise ValueError("No trained models available")
        
        best_model = None
        best_r2 = -1
        
        for model_type, model in self.power_predictors.items():
            if model.metrics and model.metrics.r2_score > best_r2:
                best_r2 = model.metrics.r2_score
                best_model = model
        
        return best_model
    
    def create_deployment_optimizer(self) -> DeploymentOptimizerModel:
        """Create deployment optimizer with best power predictor."""
        optimizer = DeploymentOptimizerModel()
        optimizer.set_power_predictor(self.get_best_power_predictor())
        return optimizer


# Example usage and testing
if __name__ == "__main__":
    # This would normally load real simulation data
    print("ML Models module loaded successfully")
    print("Use ModelManager to train and manage models")
    print("Example:")
    print("  manager = ModelManager()")
    print("  results = manager.train_power_predictor(training_data)")
    print("  optimizer = manager.create_deployment_optimizer()")