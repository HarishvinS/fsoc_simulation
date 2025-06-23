"""
Physical constants and empirical coefficients for atmospheric modeling.

Contains well-established constants from atmospheric optics literature
and ITU-R recommendations for FSOC link analysis.
"""

import math

# Fundamental constants
SPEED_OF_LIGHT = 2.998e8  # m/s
BOLTZMANN_CONSTANT = 1.38e-23  # J/K
AVOGADRO_NUMBER = 6.022e23  # mol⁻¹

# Atmospheric parameters
STANDARD_PRESSURE = 101325  # Pa
STANDARD_TEMPERATURE = 288.15  # K (15°C)
STANDARD_DENSITY = 1.225  # kg/m³
MOLECULAR_WEIGHT_AIR = 28.97e-3  # kg/mol

# Mie scattering coefficients for fog
# Based on empirical studies for water droplets at optical wavelengths
MIE_COEFFICIENT_K = 3.91  # km⁻¹/(g/m³) for λ = 1550nm
MIE_WAVELENGTH_EXPONENT = 1.3  # Approximate wavelength dependence

# Rain attenuation coefficients (ITU-R P.838-3)
# Frequency-dependent coefficients for millimeter wave
# Adapted for optical wavelengths based on droplet scattering theory
RAIN_COEFF_A = {
    1310: 0.0045,  # nm -> coefficient
    1550: 0.0042,
    850: 0.0055
}
RAIN_COEFF_B = {
    1310: 0.95,
    1550: 0.94,
    850: 0.98
}

# Thermal gradient effects
# Refractive index temperature coefficient for air
DN_DT_AIR = -1.0e-6  # K⁻¹ at standard conditions
THERMAL_MIXING_LENGTH = 50.0  # meters, typical boundary layer mixing

# Scattering cross-sections
# Rayleigh scattering coefficient for air molecules
RAYLEIGH_COEFF = 1.17e-5  # m⁻¹ at 550nm, standard conditions

# Material thermal properties for surface heating effects
MATERIAL_PROPERTIES = {
    "white_paint": {
        "absorptivity": 0.2,
        "emissivity": 0.9,
        "thermal_conductivity": 0.8  # W/m·K
    },
    "black_paint": {
        "absorptivity": 0.95,
        "emissivity": 0.9,
        "thermal_conductivity": 0.8
    },
    "aluminum": {
        "absorptivity": 0.15,
        "emissivity": 0.1,
        "thermal_conductivity": 237
    },
    "steel": {
        "absorptivity": 0.65,
        "emissivity": 0.8,
        "thermal_conductivity": 50
    },
    "concrete": {
        "absorptivity": 0.7,
        "emissivity": 0.9,
        "thermal_conductivity": 1.7
    },
    "wood": {
        "absorptivity": 0.6,
        "emissivity": 0.9,
        "thermal_conductivity": 0.15
    }
}

# Typical atmospheric layer thicknesses for modeling
DEFAULT_LAYER_THICKNESS = 10.0  # meters
BOUNDARY_LAYER_HEIGHT = 100.0  # meters
MIXING_LAYER_HEIGHT = 1000.0  # meters

# Beam parameters
TYPICAL_BEAM_DIVERGENCE = 0.001  # radians (1 mrad)
COHERENCE_LENGTH = 1.0  # meters at 1550nm

# ITU-R climate zones for regional adjustments
CLIMATE_ZONES = {
    "tropical": {"rain_factor": 1.5, "fog_factor": 0.8},
    "temperate": {"rain_factor": 1.0, "fog_factor": 1.0},
    "continental": {"rain_factor": 0.8, "fog_factor": 1.2},
    "maritime": {"rain_factor": 1.2, "fog_factor": 1.8},
    "desert": {"rain_factor": 0.3, "fog_factor": 0.2}
}


def get_rain_coefficients(wavelength_nm: float) -> tuple[float, float]:
    """Get rain attenuation coefficients for given wavelength."""
    # Interpolate or use nearest neighbor for unlisted wavelengths
    available_wavelengths = sorted(RAIN_COEFF_A.keys())
    
    if wavelength_nm in RAIN_COEFF_A:
        return RAIN_COEFF_A[wavelength_nm], RAIN_COEFF_B[wavelength_nm]
    
    # Simple linear interpolation
    if wavelength_nm < min(available_wavelengths):
        wl = min(available_wavelengths)
    elif wavelength_nm > max(available_wavelengths):
        wl = max(available_wavelengths)
    else:
        # Find nearest wavelength
        wl = min(available_wavelengths, key=lambda x: abs(x - wavelength_nm))
    
    return RAIN_COEFF_A[wl], RAIN_COEFF_B[wl]


def wavelength_scaling_factor(wavelength_nm: float, reference_nm: float = 1550) -> float:
    """Calculate wavelength scaling factor for scattering processes."""
    return (reference_nm / wavelength_nm) ** MIE_WAVELENGTH_EXPONENT


def refractive_index_air(temperature_k: float, pressure_pa: float = STANDARD_PRESSURE) -> float:
    """Calculate refractive index of air at given temperature and pressure."""
    # Edlen formula for air refractive index
    n_std = 1.0002926  # at standard conditions
    temp_factor = STANDARD_TEMPERATURE / temperature_k
    pressure_factor = pressure_pa / STANDARD_PRESSURE
    
    return 1 + (n_std - 1) * temp_factor * pressure_factor