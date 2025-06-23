"""
Basic tests for FSOC optimization system components.

Tests core functionality of input validation, physics simulation,
and model training components.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from backend.ingest.input_schema import EnvironmentInput, MaterialType
from backend.ingest.mock_weather import MockWeatherAPI
from backend.physics.layer import AtmosphereProfile, AtmosphereLayer
from backend.physics.propagation import BeamSimulator, BeamParameters
from backend.simulation.engine import FSocSimulationEngine
from backend.optimizer.models import PowerPredictorModel, ModelManager


class TestInputValidation:
    """Test input validation and schema."""
    
    def test_valid_environment_input(self):
        """Test valid environment input creation."""
        config = EnvironmentInput(
            lat_tx=37.7749, lon_tx=-122.4194,
            lat_rx=37.7849, lon_rx=-122.4094,
            height_tx=20, height_rx=15,
            material_tx=MaterialType.WHITE_PAINT,
            material_rx=MaterialType.ALUMINUM,
            fog_density=0.5, rain_rate=2.0,
            surface_temp=25, ambient_temp=20,
            wavelength_nm=1550, tx_power_dbm=20
        )
        
        assert config.lat_tx == 37.7749
        assert config.material_tx == MaterialType.WHITE_PAINT
        assert config.fog_density == 0.5
        
    def test_link_distance_calculation(self):
        """Test link distance calculation."""
        config = EnvironmentInput(
            lat_tx=37.7749, lon_tx=-122.4194,
            lat_rx=37.7849, lon_rx=-122.4094,
            height_tx=20, height_rx=15,
            material_tx=MaterialType.WHITE_PAINT,
            material_rx=MaterialType.ALUMINUM,
            fog_density=0.5, rain_rate=2.0,
            surface_temp=25, ambient_temp=20
        )
        
        distance = config.link_distance_km()
        assert 0.5 < distance < 2.0  # Approximate 1km
        
    def test_invalid_coordinates(self):
        """Test invalid coordinate validation."""
        with pytest.raises(ValueError):
            EnvironmentInput(
                lat_tx=91,  # Invalid latitude
                lon_tx=-122.4194,
                lat_rx=37.7849, lon_rx=-122.4094,
                height_tx=20, height_rx=15,
                material_tx=MaterialType.WHITE_PAINT,
                material_rx=MaterialType.ALUMINUM,
                fog_density=0.5, rain_rate=2.0,
                surface_temp=25, ambient_temp=20
            )
    
    def test_thermal_gradient_calculation(self):
        """Test thermal gradient calculation."""
        config = EnvironmentInput(
            lat_tx=37.7749, lon_tx=-122.4194,
            lat_rx=37.7849, lon_rx=-122.4094,
            height_tx=20, height_rx=15,
            material_tx=MaterialType.WHITE_PAINT,
            material_rx=MaterialType.ALUMINUM,
            fog_density=0.5, rain_rate=2.0,
            surface_temp=30, ambient_temp=20
        )
        
        gradient = config.thermal_gradient_k_per_m()
        assert gradient > 0  # Should be positive for surface heating


class TestWeatherAPI:
    """Test mock weather API functionality."""
    
    def test_weather_api_creation(self):
        """Test weather API instantiation."""
        api = MockWeatherAPI()
        assert api is not None
        
    def test_current_weather(self):
        """Test current weather retrieval."""
        api = MockWeatherAPI()
        weather = api.get_current_weather(37.7749, -122.4194)
        
        assert weather.fog_density >= 0
        assert weather.rain_rate >= 0
        assert -50 < weather.ambient_temp < 50
        assert weather.surface_temp >= weather.ambient_temp
        
    def test_weather_consistency(self):
        """Test weather API consistency (same input -> same output)."""
        api = MockWeatherAPI()
        weather1 = api.get_current_weather(37.7749, -122.4194)
        weather2 = api.get_current_weather(37.7749, -122.4194)
        
        # Should be identical for same location and time
        assert weather1.fog_density == weather2.fog_density
        assert weather1.rain_rate == weather2.rain_rate


class TestAtmosphericPhysics:
    """Test atmospheric layer and propagation physics."""
    
    def test_atmosphere_layer_creation(self):
        """Test atmospheric layer creation."""
        layer = AtmosphereLayer(
            z_bottom=0, z_top=10, thickness=10,
            fog_density=0.5, rain_rate=2.0,
            temperature=293.15, pressure=101325,
            humidity=70.0
        )
        
        assert layer.thickness == 10
        assert layer.fog_density == 0.5
        assert layer.temperature == 293.15
        
    def test_mie_attenuation_calculation(self):
        """Test Mie scattering calculation."""
        layer = AtmosphereLayer(
            z_bottom=0, z_top=10, thickness=10,
            fog_density=1.0, rain_rate=0.0,
            temperature=293.15, pressure=101325,
            humidity=70.0
        )
        
        alpha_mie = layer.calculate_mie_attenuation(1550)
        assert alpha_mie > 0
        
        # Higher fog density should give higher attenuation
        layer_high_fog = AtmosphereLayer(
            z_bottom=0, z_top=10, thickness=10,
            fog_density=2.0, rain_rate=0.0,
            temperature=293.15, pressure=101325,
            humidity=70.0
        )
        alpha_mie_high = layer_high_fog.calculate_mie_attenuation(1550)
        assert alpha_mie_high > alpha_mie
        
    def test_rain_attenuation_calculation(self):
        """Test rain attenuation calculation."""
        layer = AtmosphereLayer(
            z_bottom=0, z_top=10, thickness=10,
            fog_density=0.0, rain_rate=5.0,
            temperature=293.15, pressure=101325,
            humidity=70.0
        )
        
        alpha_rain = layer.calculate_rain_attenuation(1550)
        assert alpha_rain > 0
        
    def test_atmosphere_profile_creation(self):
        """Test atmospheric profile creation."""
        profile = AtmosphereProfile.create_uniform_profile(
            height=50,
            fog_density=0.5,
            rain_rate=2.0,
            temperature=20,
            surface_temp=25
        )
        
        assert len(profile.layers) > 0
        assert profile.total_height == 50
        
        # Test summary statistics
        stats = profile.summary_statistics()
        assert 'num_layers' in stats
        assert 'total_optical_depth' in stats


class TestBeamPropagation:
    """Test beam propagation simulation."""
    
    def test_beam_parameters(self):
        """Test beam parameter creation."""
        beam = BeamParameters(
            wavelength_nm=1550,
            power_dbm=20,
            beam_divergence=0.001
        )
        
        assert beam.wavelength_nm == 1550
        assert beam.power_dbm == 20
        assert beam.power_watts == 0.1  # 20 dBm = 100 mW = 0.1 W
        
    def test_beam_simulation(self):
        """Test beam propagation simulation."""
        # Create simple atmosphere
        atmosphere = AtmosphereProfile.create_uniform_profile(
            height=50,
            fog_density=0.5,
            rain_rate=1.0,
            temperature=20,
            surface_temp=25
        )
        
        # Create simulator
        simulator = BeamSimulator(
            atmosphere=atmosphere,
            link_distance=1000,  # 1 km
            tx_height=20,
            rx_height=20
        )
        
        # Create beam
        beam = BeamParameters(wavelength_nm=1550, power_dbm=20)
        
        # Run simulation
        result = simulator.simulate(beam)
        
        assert result.initial_power_dbm == 20
        assert result.final_power_dbm < result.initial_power_dbm  # Power loss expected
        assert result.power_loss_db > 0
        assert result.fog_loss_db >= 0
        assert result.rain_loss_db >= 0


class TestSimulationEngine:
    """Test simulation engine functionality."""
    
    def test_simulation_engine_creation(self):
        """Test simulation engine instantiation."""
        engine = FSocSimulationEngine()
        assert engine is not None
        assert engine.weather_api is not None
        
    def test_single_link_simulation(self):
        """Test single link simulation."""
        engine = FSocSimulationEngine()
        
        config = EnvironmentInput(
            lat_tx=37.7749, lon_tx=-122.4194,
            lat_rx=37.7849, lon_rx=-122.4094,
            height_tx=20, height_rx=15,
            material_tx=MaterialType.WHITE_PAINT,
            material_rx=MaterialType.ALUMINUM,
            fog_density=0.5, rain_rate=2.0,
            surface_temp=25, ambient_temp=20,
            wavelength_nm=1550, tx_power_dbm=20
        )
        
        results = engine.simulate_single_link(config)
        
        # Check required fields
        assert 'received_power_dbm' in results
        assert 'total_loss_db' in results
        assert 'link_margin_db' in results
        assert 'link_available' in results
        assert 'loss_breakdown' in results
        
        # Sanity checks
        assert results['total_loss_db'] > 0
        assert results['received_power_dbm'] < 20  # Input power
        
    def test_batch_simulation(self):
        """Test batch simulation functionality."""
        engine = FSocSimulationEngine()
        
        base_config = EnvironmentInput(
            lat_tx=37.7749, lon_tx=-122.4194,
            lat_rx=37.7849, lon_rx=-122.4094,
            height_tx=20, height_rx=20,
            material_tx=MaterialType.WHITE_PAINT,
            material_rx=MaterialType.WHITE_PAINT,
            fog_density=0.5, rain_rate=1.0,
            surface_temp=25, ambient_temp=20
        )
        
        parameter_ranges = {
            'height_tx': [15, 20, 25],
            'fog_density': [0.1, 0.5, 1.0]
        }
        
        results_df = engine.batch_simulate(parameter_ranges, base_config, max_samples=10)
        
        assert len(results_df) <= 10
        assert 'received_power_dbm' in results_df.columns
        assert 'input_height_tx' in results_df.columns


class TestMLModels:
    """Test machine learning models."""
    
    def test_power_predictor_creation(self):
        """Test power predictor model creation."""
        model = PowerPredictorModel("xgboost")
        assert model.model_type == "xgboost"
        assert not model.is_trained
        
    def test_model_manager_creation(self):
        """Test model manager creation."""
        manager = ModelManager()
        assert manager is not None
        assert len(manager.power_predictors) == 0
        
    def test_feature_preparation(self):
        """Test feature preparation for ML models."""
        # Create sample data
        data = pd.DataFrame({
            'input_height_tx': [20, 25, 30],
            'input_height_rx': [15, 20, 25],
            'input_fog_density': [0.1, 0.5, 1.0],
            'input_rain_rate': [0.0, 2.0, 5.0],
            'input_surface_temp': [25, 30, 35],
            'input_ambient_temp': [20, 25, 30],
            'input_wavelength_nm': [1550, 1550, 1550],
            'input_tx_power_dbm': [20, 20, 20],
            'input_material_tx': ['white_paint', 'aluminum', 'steel'],
            'input_material_rx': ['white_paint', 'aluminum', 'steel'],
            'link_distance_km': [1.0, 1.5, 2.0],
            'elevation_angle_deg': [0.0, 0.5, 1.0]
        })
        
        model = PowerPredictorModel("xgboost")
        features = model.prepare_features(data)
        
        assert len(features) == 3
        assert 'input_height_tx' in features.columns
        assert 'thermal_gradient' in features.columns  # Engineered feature


# Test fixtures and utilities
@pytest.fixture
def sample_environment():
    """Sample environment configuration for testing."""
    return EnvironmentInput(
        lat_tx=37.7749, lon_tx=-122.4194,
        lat_rx=37.7849, lon_rx=-122.4094,
        height_tx=20, height_rx=15,
        material_tx=MaterialType.WHITE_PAINT,
        material_rx=MaterialType.ALUMINUM,
        fog_density=0.5, rain_rate=2.0,
        surface_temp=25, ambient_temp=20,
        wavelength_nm=1550, tx_power_dbm=20
    )


@pytest.fixture
def sample_training_data():
    """Sample training data for ML models."""
    np.random.seed(42)  # For reproducibility
    
    n_samples = 100
    data = pd.DataFrame({
        'input_height_tx': np.random.uniform(10, 50, n_samples),
        'input_height_rx': np.random.uniform(10, 50, n_samples),
        'input_fog_density': np.random.exponential(0.5, n_samples),
        'input_rain_rate': np.random.exponential(2.0, n_samples),
        'input_surface_temp': np.random.normal(25, 10, n_samples),
        'input_ambient_temp': np.random.normal(20, 8, n_samples),
        'input_wavelength_nm': np.random.choice([850, 1310, 1550], n_samples),
        'input_tx_power_dbm': np.random.uniform(15, 25, n_samples),
        'input_material_tx': np.random.choice(['white_paint', 'aluminum', 'steel'], n_samples),
        'input_material_rx': np.random.choice(['white_paint', 'aluminum', 'steel'], n_samples),
        'link_distance_km': np.random.uniform(0.5, 5.0, n_samples),
        'elevation_angle_deg': np.random.uniform(-5, 5, n_samples),
        'received_power_dbm': np.random.normal(-15, 5, n_samples)  # Target variable
    })
    
    return data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])