"""
Beam propagation simulation through atmospheric layers.

Implements step-by-step propagation of optical beams through
stratified atmosphere, accounting for attenuation, scattering,
and refractive effects.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from .layer import AtmosphereLayer, AtmosphereProfile
from .constants import SPEED_OF_LIGHT, TYPICAL_BEAM_DIVERGENCE


@dataclass
class BeamParameters:
    """Optical beam characteristics."""
    wavelength_nm: float
    power_dbm: float
    beam_divergence: float = TYPICAL_BEAM_DIVERGENCE  # radians
    beam_diameter: float = 0.1  # meters at transmitter
    polarization: str = "linear"  # linear, circular
    
    @property
    def power_watts(self) -> float:
        """Convert power from dBm to watts."""
        return 10 ** ((self.power_dbm - 30) / 10)
    
    @property
    def wavelength_m(self) -> float:
        """Wavelength in meters."""
        return self.wavelength_nm * 1e-9


@dataclass
class PropagationResult:
    """Results of beam propagation simulation."""
    
    # Power evolution
    initial_power_dbm: float
    final_power_dbm: float
    power_loss_db: float
    
    # Beam characteristics
    initial_beam_diameter: float
    final_beam_diameter: float
    beam_divergence_angle: float
    
    # Atmospheric effects
    fog_loss_db: float
    rain_loss_db: float
    molecular_loss_db: float
    geometric_loss_db: float
    
    # Beam steering and distortion
    total_steering_angle: float  # radians
    scintillation_index: float
    
    # Layer-by-layer results
    layer_results: List[Dict] = None
    
    @property
    def total_attenuation_db(self) -> float:
        """Total atmospheric attenuation in dB."""
        return self.fog_loss_db + self.rain_loss_db + self.molecular_loss_db
    
    @property
    def link_margin_db(self) -> float:
        """Available link margin (positive = good)."""
        return self.final_power_dbm - (-30)  # Assume -30 dBm sensitivity


class BeamSimulator:
    """
    Comprehensive beam propagation simulator.
    
    Simulates optical beam propagation through stratified atmosphere
    with layer-by-layer calculation of attenuation and distortion effects.
    """
    
    def __init__(self, 
                 atmosphere: AtmosphereProfile,
                 link_distance: float,
                 tx_height: float,
                 rx_height: float):
        """
        Initialize beam simulator.
        
        Args:
            atmosphere: Atmospheric profile with layer structure
            link_distance: Horizontal distance between Tx and Rx (m)
            tx_height: Transmitter height above ground (m)
            rx_height: Receiver height above ground (m)
        """
        self.atmosphere = atmosphere
        self.link_distance = link_distance
        self.tx_height = tx_height
        self.rx_height = rx_height
        
        # Calculate path geometry
        self.elevation_angle = math.atan2(rx_height - tx_height, link_distance)
        self.path_length = math.sqrt(link_distance**2 + (rx_height - tx_height)**2)
    
    def simulate(self, beam: BeamParameters) -> PropagationResult:
        """
        Run complete beam propagation simulation.
        
        Args:
            beam: Beam parameters (wavelength, power, etc.)
            
        Returns:
            PropagationResult with detailed simulation outcomes
        """
        # Update atmospheric optical properties for this wavelength
        self.atmosphere.update_all_optical_properties(beam.wavelength_nm)
        
        # Initialize simulation state
        current_power = beam.power_watts
        current_diameter = beam.beam_diameter
        total_steering = 0.0
        
        # Tracking variables for loss breakdown
        fog_loss = 0.0
        rain_loss = 0.0
        molecular_loss = 0.0
        layer_results = []
        
        # Step through atmospheric layers
        path_segments = self._calculate_path_segments()
        
        for segment in path_segments:
            layer = segment["layer"]
            segment_length = segment["length"]
            
            if layer is None:  # Free space segment
                # Only geometric spreading loss
                geometric_factor = 1.0  # Minimal for terrestrial links
            else:
                # Calculate losses through this layer
                layer_transmission = math.exp(-layer.total_attenuation_coefficient() * segment_length)
                
                # Break down losses by mechanism
                fog_transmission = math.exp(-layer.alpha_fog * segment_length)
                rain_transmission = math.exp(-layer.alpha_rain * segment_length)
                molecular_transmission = math.exp(-layer.alpha_molecular * segment_length)
                
                # Accumulate power losses
                current_power *= layer_transmission
                
                # Track loss contributions (convert to dB)
                fog_loss += -10 * math.log10(fog_transmission)
                rain_loss += -10 * math.log10(rain_transmission)
                molecular_loss += -10 * math.log10(molecular_transmission)
                
                # Beam steering accumulation
                steering_increment = layer.beam_steering_angle(segment_length)
                total_steering += steering_increment
                
                # Beam spreading due to turbulence (simplified model)
                if layer.alpha_fog > 0.001:  # Significant scattering
                    turbulence_spreading = layer.alpha_fog * segment_length * 0.1
                    current_diameter += turbulence_spreading
            
            # Geometric beam spreading
            current_diameter += beam.beam_divergence * segment_length
            
            # Record layer results
            layer_results.append({
                "layer_index": len(layer_results),
                "layer": layer,
                "segment_length": segment_length,
                "power_watts": current_power,
                "beam_diameter": current_diameter,
                "steering_angle": total_steering
            })
        
        # Calculate geometric spreading loss
        geometric_loss_db = 20 * math.log10(
            (beam.beam_diameter + current_diameter) / (2 * beam.beam_diameter)
        )
        
        # Calculate scintillation index (simplified Rytov approximation)
        scintillation_index = self._calculate_scintillation_index(beam)
        
        # Compile final results
        final_power_dbm = 10 * math.log10(current_power * 1000)  # Convert to dBm
        power_loss_db = beam.power_dbm - final_power_dbm
        
        return PropagationResult(
            initial_power_dbm=beam.power_dbm,
            final_power_dbm=final_power_dbm,
            power_loss_db=power_loss_db,
            initial_beam_diameter=beam.beam_diameter,
            final_beam_diameter=current_diameter,
            beam_divergence_angle=beam.beam_divergence,
            fog_loss_db=fog_loss,
            rain_loss_db=rain_loss,
            molecular_loss_db=molecular_loss,
            geometric_loss_db=geometric_loss_db,
            total_steering_angle=total_steering,
            scintillation_index=scintillation_index,
            layer_results=layer_results
        )
    
    def _calculate_path_segments(self) -> List[Dict]:
        """
        Calculate path segments through atmospheric layers.
        
        Divides the beam path into segments based on layer boundaries
        and path geometry.
        """
        segments = []
        
        # Simple approach: assume path goes through layers proportionally
        # More sophisticated ray tracing could be implemented later
        
        remaining_distance = self.path_length
        height_step = (self.rx_height - self.tx_height) / len(self.atmosphere.layers)
        
        for i, layer in enumerate(self.atmosphere.layers):
            # Calculate fraction of path through this layer
            current_height = self.tx_height + i * height_step
            
            if (current_height >= layer.z_bottom and current_height < layer.z_top):
                # Path intersects this layer
                segment_fraction = layer.thickness / self.atmosphere.total_height
                segment_length = self.path_length * segment_fraction
                
                segments.append({
                    "layer": layer,
                    "length": segment_length,
                    "height": current_height
                })
                
                remaining_distance -= segment_length
        
        return segments
    
    def _calculate_scintillation_index(self, beam: BeamParameters) -> float:
        """
        Calculate scintillation index using Rytov approximation.
        
        Simplified model for atmospheric turbulence effects.
        """
        # Rytov variance for weak turbulence
        k = 2 * math.pi / beam.wavelength_m  # Wave number
        
        # Simplified structure parameter (depends on atmospheric conditions)
        avg_temp_gradient = np.mean([abs(layer.dn_dz) for layer in self.atmosphere.layers])
        cn_squared = 1e-14 * (1 + avg_temp_gradient * 1e6)  # Simplified model
        
        # Rytov variance
        sigma_r_squared = 1.23 * cn_squared * k**(7/6) * self.path_length**(11/6)
        
        return min(sigma_r_squared, 1.0)  # Cap at unity for validity of weak turbulence theory
    
    def analyze_link_budget(self, beam: BeamParameters, rx_sensitivity_dbm: float = -30) -> Dict:
        """
        Perform complete link budget analysis.
        
        Args:
            beam: Beam parameters
            rx_sensitivity_dbm: Receiver sensitivity threshold
            
        Returns:
            Detailed link budget breakdown
        """
        result = self.simulate(beam)
        
        link_margin = result.final_power_dbm - rx_sensitivity_dbm
        availability = 1.0 / (1.0 + result.scintillation_index)  # Simplified availability model
        
        return {
            "transmit_power_dbm": result.initial_power_dbm,
            "total_loss_db": result.power_loss_db,
            "received_power_dbm": result.final_power_dbm,
            "receiver_sensitivity_dbm": rx_sensitivity_dbm,
            "link_margin_db": link_margin,
            "link_available": link_margin > 0,
            "estimated_availability": availability,
            "loss_breakdown": {
                "fog_attenuation_db": result.fog_loss_db,
                "rain_attenuation_db": result.rain_loss_db,
                "molecular_scattering_db": result.molecular_loss_db,
                "geometric_spreading_db": result.geometric_loss_db,
                "total_atmospheric_db": result.total_attenuation_db
            },
            "beam_effects": {
                "initial_diameter_m": result.initial_beam_diameter,
                "final_diameter_m": result.final_beam_diameter,
                "beam_spreading_factor": result.final_beam_diameter / result.initial_beam_diameter,
                "steering_angle_mrad": result.total_steering_angle * 1000,
                "scintillation_index": result.scintillation_index
            }
        }


# Utility functions for common calculations
def quick_link_analysis(tx_power_dbm: float,
                       wavelength_nm: float,
                       link_distance_km: float,
                       fog_density: float,
                       rain_rate: float,
                       tx_height: float = 20,
                       rx_height: float = 20) -> Dict:
    """
    Quick link analysis for initial assessment.
    
    Simplified calculation without detailed atmospheric modeling.
    """
    # Create simple uniform atmosphere
    atmosphere = AtmosphereProfile.create_uniform_profile(
        height=max(tx_height, rx_height) + 50,
        fog_density=fog_density,
        rain_rate=rain_rate,
        temperature=20,  # °C
        surface_temp=25
    )
    
    # Create beam simulator
    simulator = BeamSimulator(
        atmosphere=atmosphere,
        link_distance=link_distance_km * 1000,
        tx_height=tx_height,
        rx_height=rx_height
    )
    
    # Define beam
    beam = BeamParameters(
        wavelength_nm=wavelength_nm,
        power_dbm=tx_power_dbm
    )
    
    return simulator.analyze_link_budget(beam)