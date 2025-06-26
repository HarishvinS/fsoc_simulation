"""
Input validation schemas for FSOC system parameters.

Defines Pydantic models for validating user inputs including
geographic coordinates, environmental conditions, and system parameters.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from enum import Enum
import math


class MaterialType(str, Enum):
    """Surface materials for mounting structures."""
    WHITE_PAINT = "white_paint"
    BLACK_PAINT = "black_paint"
    ALUMINUM = "aluminum"
    STEEL = "steel"
    CONCRETE = "concrete"
    WOOD = "wood"


class EnvironmentInput(BaseModel):
    """Complete environment and system specification for FSOC link analysis."""
    
    # Transmitter location and setup
    lat_tx: float = Field(..., ge=-90, le=90, description="Transmitter latitude in degrees")
    lon_tx: float = Field(..., ge=-180, le=180, description="Transmitter longitude in degrees")
    height_tx: float = Field(..., ge=1, le=1000, description="Transmitter mounting height in meters")
    material_tx: MaterialType = Field(..., description="Transmitter mount surface material")
    
    # Receiver location and setup  
    lat_rx: float = Field(..., ge=-90, le=90, description="Receiver latitude in degrees")
    lon_rx: float = Field(..., ge=-180, le=180, description="Receiver longitude in degrees")
    height_rx: float = Field(..., ge=1, le=1000, description="Receiver mounting height in meters")
    material_rx: MaterialType = Field(..., description="Receiver mount surface material")
    
    # Weather data options
    use_real_weather: bool = Field(
        default=False,
        description="Whether to fetch real weather data from OpenMeteo API"
    )

    # Atmospheric conditions (optional when using real weather data)
    fog_density: Optional[float] = Field(
        default=None,
        ge=0,
        le=10,
        description="Fog water content in g/m³ (required if use_real_weather=False)"
    )
    rain_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=200,
        description="Rain rate in mm/hr (required if use_real_weather=False)"
    )
    surface_temp: Optional[float] = Field(
        default=None,
        ge=-40,
        le=80,
        description="Surface temperature in °C (required if use_real_weather=False)"
    )
    ambient_temp: Optional[float] = Field(
        default=None,
        ge=-40,
        le=60,
        description="Ambient air temperature in °C (required if use_real_weather=False)"
    )
    
    # System parameters
    wavelength_nm: float = Field(
        default=1550, 
        ge=800, 
        le=2000, 
        description="Optical wavelength in nanometers"
    )
    tx_power_dbm: float = Field(
        default=20, 
        ge=-10, 
        le=50, 
        description="Transmitter power in dBm"
    )
    
    @field_validator('fog_density', 'rain_rate', 'surface_temp', 'ambient_temp')
    @classmethod
    def validate_weather_data_requirements(cls, v, info):
        """Ensure weather data is provided when not using real weather API."""
        if info.data:
            use_real_weather = info.data.get('use_real_weather', False)

            if not use_real_weather and v is None:
                raise ValueError(f"{info.field_name} is required when use_real_weather=False")

        return v

    @field_validator('surface_temp')
    @classmethod
    def validate_temperature_relationship(cls, v, info):
        """Ensure surface temperature is physically reasonable relative to ambient."""
        if v is not None and info.data and 'ambient_temp' in info.data and info.data['ambient_temp'] is not None:
            ambient = info.data['ambient_temp']
            # Surface can be warmer than ambient but not excessively colder
            if v < ambient - 10:
                raise ValueError(f"Surface temperature {v}°C is unrealistically low compared to ambient {ambient}°C")
        return v

    @field_validator('lat_rx', 'lon_rx')
    @classmethod
    def validate_rx_location(cls, v, info):
        """Ensure transmitter and receiver are not at identical locations."""
        if info.data and 'lat_tx' in info.data and 'lon_tx' in info.data:
            lat_tx, lon_tx = info.data['lat_tx'], info.data['lon_tx']
            if lat_tx is not None and lon_tx is not None:
                # Calculate approximate distance
                if info.field_name == 'lat_rx':
                    if abs(v - lat_tx) < 0.0001 and abs(info.data.get('lon_rx', 0) - lon_tx) < 0.0001:
                        raise ValueError("Transmitter and receiver cannot be at the same location")
                elif info.field_name == 'lon_rx':
                    if abs(info.data.get('lat_rx', 0) - lat_tx) < 0.0001 and abs(v - lon_tx) < 0.0001:
                        raise ValueError("Transmitter and receiver cannot be at the same location")
        return v
    
    def link_distance_km(self) -> float:
        """Calculate great circle distance between transmitter and receiver."""
        lat1, lon1 = math.radians(self.lat_tx), math.radians(self.lon_tx)
        lat2, lon2 = math.radians(self.lat_rx), math.radians(self.lon_rx)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in km
        R = 6371.0
        return R * c
    
    def thermal_gradient_k_per_m(self) -> float:
        """Estimate thermal gradient from surface and ambient temperatures."""
        # Simple model: assume linear gradient over typical boundary layer height
        boundary_layer_height = 100  # meters
        return (self.surface_temp - self.ambient_temp) / boundary_layer_height


class BatchSimulationInput(BaseModel):
    """Input for batch simulation requests."""
    base_config: EnvironmentInput
    parameter_ranges: dict = Field(
        ..., 
        description="Parameter ranges for batch simulation",
        example={
            "height_tx": [10, 20, 30, 50],
            "fog_density": [0.1, 0.5, 1.0, 2.0],
            "material_tx": ["white_paint", "aluminum"]
        }
    )
    num_samples: int = Field(default=100, ge=1, le=10000, description="Number of simulation samples")


class OptimizationRequest(BaseModel):
    """Request for deployment optimization recommendations."""
    lat_tx: float = Field(..., ge=-90, le=90)
    lon_tx: float = Field(..., ge=-180, le=180)
    lat_rx: float = Field(..., ge=-90, le=90)
    lon_rx: float = Field(..., ge=-180, le=180)
    
    # Expected conditions (can be historical averages)
    avg_fog_density: float = Field(default=0.1, ge=0, le=5)
    avg_rain_rate: float = Field(default=2.0, ge=0, le=50)
    avg_surface_temp: float = Field(default=25.0, ge=-20, le=60)
    avg_ambient_temp: float = Field(default=20.0, ge=-30, le=50)
    
    # Constraints
    min_height: float = Field(default=5, ge=1, le=100)
    max_height: float = Field(default=100, ge=10, le=1000)
    available_materials: list[MaterialType] = Field(
        default=[MaterialType.WHITE_PAINT, MaterialType.ALUMINUM, MaterialType.STEEL]
    )
    
    # Performance requirements
    min_received_power_dbm: float = Field(default=-30, description="Minimum acceptable received power")
    reliability_target: float = Field(default=0.99, ge=0.9, le=1.0, description="Required link availability")


# Example configurations for testing
EXAMPLE_URBAN_LINK = EnvironmentInput(
    lat_tx=37.7749, lon_tx=-122.4194,  # San Francisco
    lat_rx=37.7849, lon_rx=-122.4094,  # 1km away
    height_tx=20, height_rx=15,
    material_tx=MaterialType.WHITE_PAINT,
    material_rx=MaterialType.ALUMINUM,
    use_real_weather=False,
    fog_density=0.5, rain_rate=2.0,
    surface_temp=25, ambient_temp=20,
    wavelength_nm=1550, tx_power_dbm=20
)

EXAMPLE_RURAL_LINK = EnvironmentInput(
    lat_tx=40.7128, lon_tx=-74.0060,  # NYC area
    lat_rx=40.7628, lon_rx=-73.9560,  # 5km away
    height_tx=50, height_rx=45,
    material_tx=MaterialType.STEEL,
    material_rx=MaterialType.STEEL,
    use_real_weather=False,
    fog_density=0.1, rain_rate=5.0,
    surface_temp=30, ambient_temp=25,
    wavelength_nm=1550, tx_power_dbm=25
)

EXAMPLE_REAL_WEATHER_LINK = EnvironmentInput(
    lat_tx=37.7749, lon_tx=-122.4194,  # San Francisco
    lat_rx=37.7849, lon_rx=-122.4094,  # 1km away
    height_tx=20, height_rx=15,
    material_tx=MaterialType.WHITE_PAINT,
    material_rx=MaterialType.ALUMINUM,
    use_real_weather=True,
    wavelength_nm=1550, tx_power_dbm=20
)