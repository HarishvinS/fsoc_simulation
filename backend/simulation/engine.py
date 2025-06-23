"""
Main simulation engine that coordinates all components.

Combines input processing, atmospheric modeling, and beam propagation
to generate comprehensive FSOC link performance predictions.
"""

import numpy as np
import pandas as pd
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
import json
import time
from pathlib import Path

from ..ingest.input_schema import EnvironmentInput, MaterialType
from ..ingest.mock_weather import MockWeatherAPI
from ..physics.layer import AtmosphereProfile
from ..physics.propagation import BeamSimulator, BeamParameters
from ..physics.constants import MATERIAL_PROPERTIES


class FSocSimulationEngine:
    """
    Central simulation engine for FSOC link analysis.
    
    Orchestrates all components to provide comprehensive link performance
    predictions and generates datasets for machine learning training.
    """
    
    def __init__(self):
        self.weather_api = MockWeatherAPI()
        self.simulation_cache = {}
        self.last_simulation_time = 0
        
    def simulate_single_link(self, 
                           environment: EnvironmentInput,
                           detailed_output: bool = False) -> Dict[str, Any]:
        """
        Simulate a single FSOC link with given environment parameters.
        
        Args:
            environment: Complete environment specification
            detailed_output: Include layer-by-layer results
            
        Returns:
            Comprehensive simulation results dictionary
        """
        start_time = time.time()
        
        # Calculate link geometry
        link_distance = environment.link_distance_km() * 1000  # Convert to meters
        
        # Create atmospheric profiles for both ends
        tx_atmosphere = AtmosphereProfile.create_realistic_profile(
            height=environment.height_tx + 50,  # Add buffer above mount
            surface_conditions={
                "fog_density": environment.fog_density,
                "rain_rate": environment.rain_rate,
                "surface_temp": environment.surface_temp,
                "ambient_temp": environment.ambient_temp
            },
            material_type=environment.material_tx.value
        )
        
        rx_atmosphere = AtmosphereProfile.create_realistic_profile(
            height=environment.height_rx + 50,
            surface_conditions={
                "fog_density": environment.fog_density,
                "rain_rate": environment.rain_rate,
                "surface_temp": environment.surface_temp,
                "ambient_temp": environment.ambient_temp
            },
            material_type=environment.material_rx.value
        )
        
        # Use average atmosphere for propagation (simplification)
        # In practice, you might want more sophisticated path modeling
        avg_atmosphere = tx_atmosphere  # Simplified assumption
        
        # Create beam simulator
        simulator = BeamSimulator(
            atmosphere=avg_atmosphere,
            link_distance=link_distance,
            tx_height=environment.height_tx,
            rx_height=environment.height_rx
        )
        
        # Define beam parameters
        beam = BeamParameters(
            wavelength_nm=environment.wavelength_nm,
            power_dbm=environment.tx_power_dbm,
            beam_divergence=0.001  # 1 mrad typical
        )
        
        # Run propagation simulation
        prop_result = simulator.simulate(beam)
        link_budget = simulator.analyze_link_budget(beam)
        
        # Compile comprehensive results
        results = {
            # Input parameters
            "input_parameters": environment.dict(),
            
            # Link geometry
            "link_distance_km": link_distance / 1000,
            "elevation_angle_deg": math.degrees(simulator.elevation_angle),
            "path_length_m": simulator.path_length,
            
            # Atmospheric conditions
            "atmosphere_summary": avg_atmosphere.summary_statistics(),
            
            # Primary performance metrics
            "received_power_dbm": prop_result.final_power_dbm,
            "total_loss_db": prop_result.power_loss_db,
            "link_margin_db": link_budget["link_margin_db"],
            "link_available": link_budget["link_available"],
            "estimated_availability": link_budget["estimated_availability"],
            
            # Loss breakdown
            "loss_breakdown": {
                "fog_loss_db": prop_result.fog_loss_db,
                "rain_loss_db": prop_result.rain_loss_db,
                "molecular_loss_db": prop_result.molecular_loss_db,
                "geometric_loss_db": prop_result.geometric_loss_db,
                "total_atmospheric_loss_db": prop_result.total_attenuation_db
            },
            
            # Beam characteristics
            "beam_analysis": {
                "initial_diameter_m": prop_result.initial_beam_diameter,
                "final_diameter_m": prop_result.final_beam_diameter,
                "beam_spreading_ratio": prop_result.final_beam_diameter / prop_result.initial_beam_diameter,
                "total_steering_angle_mrad": prop_result.total_steering_angle * 1000,
                "scintillation_index": prop_result.scintillation_index
            },
            
            # Material effects
            "material_effects": {
                "tx_material": environment.material_tx.value,
                "rx_material": environment.material_rx.value,
                "tx_surface_properties": MATERIAL_PROPERTIES[environment.material_tx.value],
                "rx_surface_properties": MATERIAL_PROPERTIES[environment.material_rx.value]
            },
            
            # Simulation metadata
            "simulation_time_s": time.time() - start_time,
            "timestamp": time.time()
        }
        
        # Add detailed layer results if requested
        if detailed_output and prop_result.layer_results:
            results["layer_results"] = [
                {
                    "layer_index": r["layer_index"],
                    "segment_length_m": r["segment_length"],
                    "power_watts": r["power_watts"],
                    "beam_diameter_m": r["beam_diameter"],
                    "steering_angle_mrad": r["steering_angle"] * 1000,
                    "layer_properties": {
                        "fog_density": r["layer"].fog_density if r["layer"] else 0,
                        "rain_rate": r["layer"].rain_rate if r["layer"] else 0,
                        "temperature_c": r["layer"].temperature - 273.15 if r["layer"] else 0,
                        "alpha_total": r["layer"].total_attenuation_coefficient() if r["layer"] else 0
                    }
                }
                for r in prop_result.layer_results
            ]
        
        self.last_simulation_time = time.time() - start_time
        return results
    
    def batch_simulate(self, 
                      parameter_ranges: Dict[str, List],
                      base_config: EnvironmentInput,
                      max_samples: int = 1000) -> pd.DataFrame:
        """
        Generate batch simulation results for ML training data.
        
        Args:
            parameter_ranges: Dictionary of parameters and their ranges
            base_config: Base configuration to vary
            max_samples: Maximum number of samples to generate
            
        Returns:
            DataFrame with simulation results
        """
        results = []
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(
            parameter_ranges, max_samples
        )
        
        print(f"Running {len(param_combinations)} simulations...")
        
        for i, params in enumerate(param_combinations):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(param_combinations)} ({100*i/len(param_combinations):.1f}%)")
            
            # Create modified configuration
            config_dict = base_config.dict()
            config_dict.update(params)
            
            try:
                modified_config = EnvironmentInput(**config_dict)
                result = self.simulate_single_link(modified_config)
                
                # Flatten result for DataFrame
                flat_result = self._flatten_simulation_result(result)
                results.append(flat_result)
                
            except Exception as e:
                print(f"Simulation failed for parameters {params}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _generate_parameter_combinations(self, 
                                       parameter_ranges: Dict[str, List],
                                       max_samples: int) -> List[Dict]:
        """Generate parameter combinations for batch simulation."""
        from itertools import product
        
        # Get all parameter names and values
        param_names = list(parameter_ranges.keys())
        param_values = list(parameter_ranges.values())
        
        # Generate all combinations
        all_combinations = list(product(*param_values))
        
        # Limit to max_samples
        if len(all_combinations) > max_samples:
            # Random sampling
            import random
            selected_indices = random.sample(range(len(all_combinations)), max_samples)
            selected_combinations = [all_combinations[i] for i in selected_indices]
        else:
            selected_combinations = all_combinations
        
        # Convert to list of dictionaries
        return [
            dict(zip(param_names, combo))
            for combo in selected_combinations
        ]
    
    def _flatten_simulation_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested simulation result for DataFrame storage."""
        flat = {}
        
        # Input parameters
        for key, value in result["input_parameters"].items():
            flat[f"input_{key}"] = value
        
        # Link geometry
        flat["link_distance_km"] = result["link_distance_km"]
        flat["elevation_angle_deg"] = result["elevation_angle_deg"]
        flat["path_length_m"] = result["path_length_m"]
        
        # Performance metrics
        flat["received_power_dbm"] = result["received_power_dbm"]
        flat["total_loss_db"] = result["total_loss_db"]
        flat["link_margin_db"] = result["link_margin_db"]
        flat["link_available"] = result["link_available"]
        flat["estimated_availability"] = result["estimated_availability"]
        
        # Loss breakdown
        for key, value in result["loss_breakdown"].items():
            flat[f"loss_{key}"] = value
        
        # Beam analysis
        for key, value in result["beam_analysis"].items():
            flat[f"beam_{key}"] = value
        
        # Atmosphere summary
        for key, value in result["atmosphere_summary"].items():
            flat[f"atm_{key}"] = value
        
        return flat
    
    def save_simulation_results(self, 
                              results: pd.DataFrame, 
                              filename: str,
                              format: str = "csv"):
        """Save simulation results to file."""
        if format.lower() == "csv":
            results.to_csv(filename, index=False)
        elif format.lower() == "json":
            results.to_json(filename, orient="records", indent=2)
        elif format.lower() == "parquet":
            results.to_parquet(filename)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Saved {len(results)} simulation results to {filename}")
    
    def load_simulation_results(self, filename: str) -> pd.DataFrame:
        """Load simulation results from file."""
        if filename.endswith('.csv'):
            return pd.read_csv(filename)
        elif filename.endswith('.json'):
            return pd.read_json(filename, orient="records")
        elif filename.endswith('.parquet'):
            return pd.read_parquet(filename)
        else:
            raise ValueError(f"Unsupported file format: {filename}")


# Utility functions for common simulation tasks
def create_training_dataset(output_dir: str = "backend/data",
                          num_samples: int = 1000) -> str:
    """
    Create a comprehensive training dataset for ML models.
    
    Args:
        output_dir: Directory to save dataset
        num_samples: Number of samples to generate
        
    Returns:
        Path to saved dataset file
    """
    engine = FSocSimulationEngine()
    
    # Define parameter ranges for diverse dataset
    parameter_ranges = {
        "height_tx": [5, 10, 15, 20, 30, 40, 50, 75, 100],
        "height_rx": [5, 10, 15, 20, 30, 40, 50, 75, 100],
        "material_tx": [m.value for m in MaterialType],
        "material_rx": [m.value for m in MaterialType],
        "fog_density": [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0],
        "rain_rate": [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
        "surface_temp": [0, 10, 20, 25, 30, 35, 40, 50],
        "ambient_temp": [0, 10, 15, 20, 25, 30, 35, 40],
        "wavelength_nm": [850, 1310, 1550],
        "tx_power_dbm": [10, 15, 20, 25, 30]
    }
    
    # Base configuration (San Francisco area)
    base_config = EnvironmentInput(
        lat_tx=37.7749, lon_tx=-122.4194,
        lat_rx=37.7849, lon_rx=-122.4094,
        height_tx=20, height_rx=20,
        material_tx=MaterialType.WHITE_PAINT,
        material_rx=MaterialType.WHITE_PAINT,
        fog_density=0.1, rain_rate=1.0,
        surface_temp=25, ambient_temp=20,
        wavelength_nm=1550, tx_power_dbm=20
    )
    
    print("Generating training dataset...")
    dataset = engine.batch_simulate(parameter_ranges, base_config, num_samples)
    
    # Save dataset
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = f"{output_dir}/fsoc_training_dataset_{num_samples}samples.csv"
    engine.save_simulation_results(dataset, output_file)
    
    return output_file


if __name__ == "__main__":
    # Example usage
    import math
    
    # Create simple test case
    test_config = EnvironmentInput(
        lat_tx=37.7749, lon_tx=-122.4194,  # San Francisco
        lat_rx=37.7849, lon_rx=-122.4094,  # 1km away
        height_tx=20, height_rx=15,
        material_tx=MaterialType.WHITE_PAINT,
        material_rx=MaterialType.ALUMINUM,
        fog_density=0.5, rain_rate=2.0,
        surface_temp=25, ambient_temp=20,
        wavelength_nm=1550, tx_power_dbm=20
    )
    
    # Run simulation
    engine = FSocSimulationEngine()
    results = engine.simulate_single_link(test_config, detailed_output=True)
    
    print("Simulation Results:")
    print(f"Link distance: {results['link_distance_km']:.2f} km")
    print(f"Received power: {results['received_power_dbm']:.1f} dBm")
    print(f"Total loss: {results['total_loss_db']:.1f} dB")
    print(f"Link margin: {results['link_margin_db']:.1f} dB")
    print(f"Link available: {results['link_available']}")
    print(f"Estimated availability: {results['estimated_availability']:.3f}")
    
    print("\nLoss breakdown:")
    for loss_type, loss_value in results['loss_breakdown'].items():
        print(f"  {loss_type}: {loss_value:.2f} dB")