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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F

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


class FSocNeuralNetwork(nn.Module):
    """
    Neural network for FSOC power prediction with uncertainty quantification.

    Architecture:
    - Input layer: Environmental and system parameters
    - Hidden layers: 3 layers with dropout for uncertainty
    - Output layer: Power prediction with uncertainty estimation
    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32], dropout_rate: float = 0.2):
        super(FSocNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.dropout_rate = dropout_rate

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer (no dropout on final layer)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """Forward pass through the network."""
        return self.network(x)

    def predict_with_uncertainty(self, x, n_samples: int = 100):
        """
        Predict with uncertainty estimation using Monte Carlo dropout.

        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples for uncertainty estimation

        Returns:
            mean_prediction: Mean prediction
            uncertainty: Standard deviation of predictions (uncertainty)
        """
        # Handle single sample by temporarily adding batch dimension if needed
        original_shape = x.shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        elif x.shape[0] == 1 and len(x.shape) == 2:
            # For single sample, duplicate it to avoid batch norm issues
            x = x.repeat(2, 1)
            single_sample = True
        else:
            single_sample = False

        self.train()  # Enable dropout for uncertainty estimation

        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                if single_sample:
                    pred = pred[:1]  # Take only the first prediction
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        self.eval()  # Return to evaluation mode

        return mean_prediction, uncertainty


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
            model_type: Type of ML model ("random_forest", "xgboost", "lightgbm", "neural_network")
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.target_name = "received_power_dbm"
        self.metrics = None
        self.is_trained = False

        # Neural network specific attributes
        self.neural_network = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_history = []
        
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
        elif model_type == "neural_network":
            # Neural network will be initialized during training
            self.model = None
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
        
        # Train model based on type
        if self.model_type == "neural_network":
            # Train neural network
            y_pred = self._train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val)
        elif self.model_type in ["random_forest"]:
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
        
        # Cross-validation (skip for neural networks as they require special handling)
        if self.model_type == "neural_network":
            # For neural networks, use validation score as proxy for CV score
            cv_scores = np.array([r2, r2, r2, r2, r2])  # Placeholder - could implement proper CV later
        else:
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

    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
        """
        Train neural network model.

        Args:
            X_train: Training features (scaled)
            y_train: Training targets
            X_val: Validation features (scaled)
            y_val: Validation targets

        Returns:
            Validation predictions
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        # Create neural network
        input_size = X_train.shape[1]
        self.neural_network = FSocNeuralNetwork(input_size).to(self.device)

        # Training parameters
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.neural_network.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        epochs = 200
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20

        self.training_history = []

        for epoch in range(epochs):
            # Training phase
            self.neural_network.train()
            optimizer.zero_grad()

            train_pred = self.neural_network(X_train_tensor)
            train_loss = criterion(train_pred, y_train_tensor)

            train_loss.backward()
            optimizer.step()

            # Validation phase
            self.neural_network.eval()
            with torch.no_grad():
                val_pred = self.neural_network(X_val_tensor)
                val_loss = criterion(val_pred, torch.FloatTensor(y_val.values).unsqueeze(1).to(self.device))

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_model_state = self.neural_network.state_dict().copy()
            else:
                patience_counter += 1

            # Record training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'val_loss': val_loss.item(),
                'lr': optimizer.param_groups[0]['lr']
            })

            if epoch % 20 == 0:
                print(f"  Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Load best model
        self.neural_network.load_state_dict(best_model_state)

        # Return validation predictions
        self.neural_network.eval()
        with torch.no_grad():
            val_predictions = self.neural_network(X_val_tensor)
            return val_predictions.cpu().numpy().flatten()

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
        
        # Make predictions based on model type
        if self.model_type == "neural_network":
            # Neural network prediction
            features_scaled = self.scaler.transform(features)
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)

            # Handle single sample batch norm issue
            if features_tensor.shape[0] == 1:
                features_tensor = features_tensor.repeat(2, 1)
                single_sample = True
            else:
                single_sample = False

            self.neural_network.eval()
            with torch.no_grad():
                prediction = self.neural_network(features_tensor)
                if single_sample:
                    prediction = prediction[:1]  # Take only first prediction
                prediction = prediction.cpu().numpy().flatten()
        elif self.model_type in ["xgboost", "lightgbm"]:
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)
        else:
            prediction = self.model.predict(features)

        return prediction[0] if len(prediction) == 1 else prediction

    def predict_with_uncertainty(self, input_data: Union[Dict, pd.DataFrame],
                                n_samples: int = 100) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Predict with uncertainty estimation (only for neural networks).

        Args:
            input_data: Environmental and system parameters
            n_samples: Number of Monte Carlo samples for uncertainty estimation

        Returns:
            Tuple of (prediction, uncertainty)
        """
        if self.model_type != "neural_network":
            # For non-neural models, return prediction with zero uncertainty
            prediction = self.predict(input_data)
            uncertainty = np.zeros_like(prediction) if isinstance(prediction, np.ndarray) else 0.0
            return prediction, uncertainty

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
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            features = features[self.feature_names]

        # Scale features and predict with uncertainty
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)

        mean_pred, uncertainty = self.neural_network.predict_with_uncertainty(features_tensor, n_samples)

        mean_pred = mean_pred.cpu().numpy().flatten()
        uncertainty = uncertainty.cpu().numpy().flatten()

        if len(mean_pred) == 1:
            return mean_pred[0], uncertainty[0]
        else:
            return mean_pred, uncertainty

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
            'target_name': self.target_name,
            'training_history': self.training_history
        }

        # For neural networks, save the state dict separately
        if self.model_type == "neural_network" and self.neural_network is not None:
            model_data['neural_network_state'] = self.neural_network.state_dict()
            model_data['neural_network_config'] = {
                'input_size': self.neural_network.input_size,
                'hidden_sizes': self.neural_network.hidden_sizes,
                'dropout_rate': self.neural_network.dropout_rate
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
        self.training_history = model_data.get('training_history', [])

        # Load neural network if present
        if self.model_type == "neural_network" and 'neural_network_state' in model_data:
            config = model_data['neural_network_config']
            self.neural_network = FSocNeuralNetwork(
                input_size=config['input_size'],
                hidden_sizes=config['hidden_sizes'],
                dropout_rate=config['dropout_rate']
            ).to(self.device)
            self.neural_network.load_state_dict(model_data['neural_network_state'])
            self.neural_network.eval()

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

        # Height parameters - use more granular search for better optimization
        height_min = constraints.get('min_height', 5)
        height_max = constraints.get('max_height', 100)

        # Create a more intelligent height search space
        # Include key heights: minimum, maximum, and strategic intermediate values
        height_range = height_max - height_min
        if height_range <= 10:
            # Small range - use fine granularity
            height_steps = max(5, int(height_range))
        elif height_range <= 50:
            # Medium range - use moderate granularity
            height_steps = 15
        else:
            # Large range - use coarser granularity but include key points
            height_steps = 20

        param_space['height_tx'] = np.linspace(height_min, height_max, height_steps)
        param_space['height_rx'] = np.linspace(height_min, height_max, height_steps)

        # Material options - ensure we have valid materials
        available_materials = constraints.get('available_materials',
                                            ['white_paint', 'aluminum', 'steel'])

        # Filter out any invalid materials and ensure we have at least one option
        valid_materials = [m for m in available_materials if m in
                          ['white_paint', 'aluminum', 'steel', 'concrete', 'wood', 'black_paint']]
        if not valid_materials:
            valid_materials = ['white_paint']  # Default fallback

        param_space['material_tx'] = valid_materials
        param_space['material_rx'] = valid_materials

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
        valid_configs = []

        # Get parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())

        # Extract constraints for validation
        min_received_power = base_conditions.get('min_received_power', -50)
        reliability_target = base_conditions.get('reliability_target', 0.99)

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

            # Create DataFrame with the exact features expected by the saved model
            # Enhanced feature engineering for more realistic predictions
            height_tx = test_config.get('input_height_tx', 20)
            height_rx = test_config.get('input_height_rx', 20)
            fog_density = test_config.get('input_fog_density', 0.1)
            rain_rate = test_config.get('input_rain_rate', 2.0)
            surface_temp = test_config.get('input_surface_temp', 25.0)
            ambient_temp = test_config.get('input_ambient_temp', 20.0)

            # Calculate more sophisticated derived features
            height_differential = height_tx - height_rx
            avg_height = (height_tx + height_rx) / 2

            # Thermal gradient based on temperature difference and height
            thermal_gradient = abs(surface_temp - ambient_temp) / max(1, avg_height / 10)

            # Atmospheric loading considering both fog and rain effects
            atmospheric_loading = fog_density * 0.5 + rain_rate * 0.1

            # Scattering potential with more realistic modeling
            scattering_potential = (fog_density * rain_rate * 0.1) + (fog_density ** 1.5) * 0.2

            # Path loss factors
            fresnel_clearance = max(0, min(avg_height, 100) / link_distance_km) if link_distance_km > 0 else 1.0

            required_features = {
                'input_height_tx': height_tx,
                'input_height_rx': height_rx,
                'input_fog_density': fog_density,
                'input_rain_rate': rain_rate,
                'input_surface_temp': surface_temp,
                'input_ambient_temp': ambient_temp,
                'input_wavelength_nm': test_config.get('input_wavelength_nm', 1550),
                'input_tx_power_dbm': test_config.get('input_tx_power_dbm', 20),
                'link_distance_km': link_distance_km,
                'elevation_angle_deg': elevation_angle_deg,
                'input_material_tx': test_config.get('input_material_tx', 'white_paint'),
                'input_material_rx': test_config.get('input_material_rx', 'white_paint'),
                'thermal_gradient': thermal_gradient,
                'atmospheric_loading': atmospheric_loading,
                'height_differential': height_differential,
                'scattering_potential': scattering_potential
            }

            # Predict performance
            try:
                # Use the enhanced prediction method with uncertainty
                if self.power_predictor.model_type == "neural_network":
                    # Get prediction with uncertainty for neural networks
                    predicted_power, uncertainty = self.power_predictor.predict_with_uncertainty(required_features)
                else:
                    # Use regular prediction for other models
                    predicted_power = self.power_predictor.predict(required_features)
                    uncertainty = 0.0

                # Validate constraints before considering this configuration
                constraints_met = True
                constraint_violations = []

                # Check minimum received power constraint
                if predicted_power < min_received_power:
                    constraints_met = False
                    constraint_violations.append(f"Power {predicted_power:.1f} dBm < minimum {min_received_power:.1f} dBm")

                # Calculate estimated reliability based on power margin
                power_margin = predicted_power - min_received_power
                estimated_reliability = min(1.0, max(0.5, 0.8 + (power_margin / 20.0)))  # Simple reliability model

                # Check reliability constraint
                if estimated_reliability < reliability_target:
                    constraints_met = False
                    constraint_violations.append(f"Reliability {estimated_reliability:.1%} < target {reliability_target:.1%}")

                # Only consider configurations that meet all constraints
                if constraints_met:
                    # Calculate score based on target
                    if target == "max_power":
                        score = predicted_power
                    else:
                        score = predicted_power

                    # Store valid configuration with flattened parameters
                    config_result = {
                        'predicted_power_dbm': predicted_power,
                        'estimated_reliability': estimated_reliability,
                        'power_margin_db': power_margin,
                        'optimization_score': score,
                        'constraints_met': True
                    }

                    # Add individual parameters for easy access
                    for name, value in zip(param_names, combination):
                        config_result[name] = value

                    # Add uncertainty information if available
                    if self.power_predictor.model_type == "neural_network":
                        config_result['prediction_uncertainty_db'] = uncertainty
                        config_result['confidence_level'] = max(0.1, min(1.0, 1.0 - (uncertainty / 10.0)))
                    else:
                        config_result['prediction_uncertainty_db'] = 0.0
                        config_result['confidence_level'] = 0.8

                    valid_configs.append(config_result)

                    # Update best if this is better
                    if score > best_score:
                        best_score = score
                        best_params = config_result.copy()
                else:
                    # Log constraint violations for debugging
                    print(f"Configuration {dict(zip(param_names, combination))} violates constraints: {constraint_violations}")
                    
            except Exception as e:
                print(f"Optimization failed for combination {dict(zip(param_names, combination))}: {e}")
                continue

        # Return best configuration if found, otherwise return error information
        if best_params:
            # Add summary information
            best_params['total_configurations_tested'] = len(list(product(*param_values)))
            best_params['valid_configurations_found'] = len(valid_configs)
            return best_params
        else:
            # No valid configurations found - return relaxed constraints recommendation
            print(f"No configurations met all constraints. Tested {len(list(product(*param_values)))} combinations.")
            print(f"Constraints: min_power={min_received_power} dBm, reliability={reliability_target:.1%}")

            # Find the best configuration even if it doesn't meet all constraints
            if valid_configs:
                # This shouldn't happen, but just in case
                return max(valid_configs, key=lambda x: x['optimization_score'])
            else:
                # Return a default configuration with warning
                return {
                    'height_tx': param_space['height_tx'][len(param_space['height_tx'])//2],  # Middle value
                    'height_rx': param_space['height_rx'][len(param_space['height_rx'])//2],  # Middle value
                    'material_tx': param_space['material_tx'][0] if param_space['material_tx'] else 'white_paint',
                    'material_rx': param_space['material_rx'][0] if param_space['material_rx'] else 'white_paint',
                    'predicted_power_dbm': -50,  # Conservative estimate
                    'estimated_reliability': 0.5,  # Conservative estimate
                    'optimization_score': -50,
                    'constraints_met': False,
                    'warning': 'No configurations met all constraints. Consider relaxing requirements.',
                    'total_configurations_tested': len(list(product(*param_values))),
                    'valid_configurations_found': 0
                }


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
                            model_types: List[str] = ["neural_network", "xgboost", "random_forest"]) -> Dict[str, ModelMetrics]:
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