"""
Atmospheric layer modeling for beam propagation analysis.

Implements physics-based models for various atmospheric effects:
- Mie scattering from fog and aerosols
- Rain attenuation using ITU-R models
- Thermal gradient effects on beam steering
- Layer-by-layer propagation simulation
"""

import math
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .constants import (
    MIE_COEFFICIENT_K, MIE_WAVELENGTH_EXPONENT,
    get_rain_coefficients, wavelength_scaling_factor,
    DN_DT_AIR, refractive_index_air,
    DEFAULT_LAYER_THICKNESS, MATERIAL_PROPERTIES
)


@dataclass
class AtmosphereLayer:
    """
    Single atmospheric layer with uniform properties.
    
    Represents a slice of atmosphere with constant environmental
    conditions for propagation calculations.
    """
    
    # Layer geometry
    z_bottom: float     # Bottom altitude (m)
    z_top: float        # Top altitude (m) 
    thickness: float    # Layer thickness (m)
    
    # Environmental conditions
    fog_density: float      # Water content (g/m³)
    rain_rate: float        # Precipitation rate (mm/hr)
    temperature: float      # Air temperature (K)
    pressure: float         # Air pressure (Pa)
    humidity: float         # Relative humidity (%)
    
    # Derived optical properties
    alpha_fog: float = 0.0      # Fog attenuation coefficient (m⁻¹)
    alpha_rain: float = 0.0     # Rain attenuation coefficient (m⁻¹)
    alpha_molecular: float = 0.0 # Molecular scattering coefficient (m⁻¹)
    
    # Refractive effects
    refractive_index: float = 1.0
    dn_dz: float = 0.0          # Refractive index gradient (m⁻¹)
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.thickness = self.z_top - self.z_bottom
        self.refractive_index = refractive_index_air(self.temperature, self.pressure)
    
    def calculate_mie_attenuation(self, wavelength_nm: float) -> float:
        """
        Calculate Mie scattering attenuation from fog particles.
        
        Uses empirical relationship: α = k * ρ * (λ₀/λ)^β
        where ρ is fog density, λ is wavelength.
        """
        if self.fog_density <= 0:
            return 0.0
            
        # Wavelength scaling relative to reference
        wavelength_factor = wavelength_scaling_factor(wavelength_nm)
        
        # Convert density to appropriate units and apply scaling
        alpha_fog = (MIE_COEFFICIENT_K * self.fog_density * wavelength_factor) / 1000.0  # m⁻¹
        
        return alpha_fog
    
    def calculate_rain_attenuation(self, wavelength_nm: float) -> float:
        """
        Calculate rain attenuation using modified ITU-R model.
        
        Adapts ITU-R P.838 coefficients for optical wavelengths
        based on geometric scattering from raindrops.
        """
        if self.rain_rate <= 0:
            return 0.0
        
        # Get wavelength-specific coefficients
        coeff_a, coeff_b = get_rain_coefficients(wavelength_nm)
        
        # ITU-R formula adapted for optical wavelengths
        alpha_rain = coeff_a * (self.rain_rate ** coeff_b) / 1000.0  # Convert to m⁻¹
        
        return alpha_rain
    
    def calculate_molecular_attenuation(self, wavelength_nm: float) -> float:
        """
        Calculate Rayleigh scattering from air molecules.
        
        Typically small compared to fog/rain but included for completeness.
        """
        # Rayleigh scattering scales as λ⁻⁴
        wavelength_m = wavelength_nm * 1e-9
        rayleigh_coeff = 1.17e-5 * (550e-9 / wavelength_m) ** 4
        
        # Scale by air density
        density_factor = self.pressure / 101325 * 288.15 / self.temperature
        
        return rayleigh_coeff * density_factor
    
    def update_optical_properties(self, wavelength_nm: float):
        """Update all wavelength-dependent optical properties."""
        self.alpha_fog = self.calculate_mie_attenuation(wavelength_nm)
        self.alpha_rain = self.calculate_rain_attenuation(wavelength_nm)
        self.alpha_molecular = self.calculate_molecular_attenuation(wavelength_nm)
    
    def total_attenuation_coefficient(self) -> float:
        """Total attenuation coefficient for this layer."""
        return self.alpha_fog + self.alpha_rain + self.alpha_molecular
    
    def transmission_factor(self) -> float:
        """Beer-Lambert transmission through this layer."""
        total_alpha = self.total_attenuation_coefficient()
        return math.exp(-total_alpha * self.thickness)
    
    def calculate_refractive_gradient(self, surface_temp_k: Optional[float] = None):
        """
        Calculate refractive index gradient from temperature profile.
        
        Uses thermal gradient to estimate dn/dz for beam steering calculations.
        """
        if surface_temp_k is None:
            # Assume standard atmospheric gradient
            self.dn_dz = DN_DT_AIR * (-6.5e-3)  # Standard lapse rate
        else:
            # Calculate gradient from surface heating
            temp_gradient = (surface_temp_k - self.temperature) / self.z_bottom if self.z_bottom > 0 else 0
            self.dn_dz = DN_DT_AIR * temp_gradient
    
    def beam_steering_angle(self, path_length: float) -> float:
        """
        Calculate beam steering angle due to refractive index gradient.
        
        Uses small-angle approximation for beam bending.
        """
        if abs(self.dn_dz) < 1e-12:
            return 0.0
        
        # Simple ray bending formula
        steering_angle = self.dn_dz * path_length
        return steering_angle


class AtmosphereProfile:
    """
    Complete atmospheric profile composed of multiple layers.
    
    Manages the vertical structure of atmospheric conditions
    and provides methods for beam propagation calculations.
    """
    
    def __init__(self, layers: List[AtmosphereLayer]):
        self.layers = sorted(layers, key=lambda l: l.z_bottom)
        self.total_height = max(layer.z_top for layer in self.layers)
    
    @classmethod
    def create_uniform_profile(cls, 
                             height: float,
                             fog_density: float,
                             rain_rate: float,
                             temperature: float,
                             surface_temp: float,
                             layer_thickness: float = DEFAULT_LAYER_THICKNESS):
        """
        Create atmospheric profile with uniform conditions.
        
        Useful for simple simulations where vertical variation is minimal.
        """
        layers = []
        num_layers = int(math.ceil(height / layer_thickness))
        
        for i in range(num_layers):
            z_bottom = i * layer_thickness
            z_top = min((i + 1) * layer_thickness, height)
            
            # Simple temperature profile with surface heating
            if i == 0:  # Surface layer
                layer_temp = (surface_temp + temperature) / 2
            else:
                # Linear decrease with height
                layer_temp = temperature - 0.0065 * z_bottom  # Standard lapse rate
            
            layer = AtmosphereLayer(
                z_bottom=z_bottom,
                z_top=z_top,
                thickness=z_top - z_bottom,
                fog_density=fog_density,
                rain_rate=rain_rate,
                temperature=layer_temp + 273.15,  # Convert to Kelvin
                pressure=101325 * (1 - 0.0065 * z_bottom / 288.15) ** 5.26,  # Barometric formula
                humidity=70.0  # Default humidity
            )
            
            # Calculate refractive gradient for surface heating
            layer.calculate_refractive_gradient(surface_temp + 273.15)
            
            layers.append(layer)
        
        return cls(layers)
    
    @classmethod
    def create_realistic_profile(cls,
                               height: float,
                               surface_conditions: dict,
                               material_type: str = "white_paint"):
        """
        Create realistic atmospheric profile with material-specific surface heating.
        
        Incorporates surface material properties to model thermal gradients.
        """
        material_props = MATERIAL_PROPERTIES.get(material_type, MATERIAL_PROPERTIES["white_paint"])
        
        # Estimate surface heating based on material absorptivity
        solar_irradiance = 800  # W/m² typical daytime value
        surface_heating = material_props["absorptivity"] * solar_irradiance / 100
        enhanced_surface_temp = surface_conditions["surface_temp"] + surface_heating
        
        return cls.create_uniform_profile(
            height=height,
            fog_density=surface_conditions["fog_density"],
            rain_rate=surface_conditions["rain_rate"],
            temperature=surface_conditions["ambient_temp"],
            surface_temp=enhanced_surface_temp
        )
    
    def update_all_optical_properties(self, wavelength_nm: float):
        """Update optical properties for all layers at given wavelength."""
        for layer in self.layers:
            layer.update_optical_properties(wavelength_nm)
    
    def total_optical_depth(self) -> float:
        """Calculate total optical depth through all layers."""
        return sum(layer.total_attenuation_coefficient() * layer.thickness 
                  for layer in self.layers)
    
    def total_transmission(self) -> float:
        """Calculate total transmission through entire atmosphere."""
        return math.exp(-self.total_optical_depth())
    
    def get_layer_at_height(self, height: float) -> Optional[AtmosphereLayer]:
        """Get the atmospheric layer at specified height."""
        for layer in self.layers:
            if layer.z_bottom <= height < layer.z_top:
                return layer
        return None
    
    def summary_statistics(self) -> dict:
        """Return summary statistics of the atmospheric profile."""
        return {
            "num_layers": len(self.layers),
            "total_height": self.total_height,
            "avg_fog_density": np.mean([l.fog_density for l in self.layers]),
            "avg_rain_rate": np.mean([l.rain_rate for l in self.layers]),
            "avg_temperature": np.mean([l.temperature for l in self.layers]) - 273.15,
            "total_optical_depth": self.total_optical_depth(),
            "total_transmission": self.total_transmission()
        }