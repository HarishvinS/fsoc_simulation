"""
Mock weather API for development and testing.

Provides realistic synthetic weather data that simulates responses
from real weather APIs like Open-Meteo. Will be replaced with
actual API calls in Phase 5.
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class WeatherCondition:
    """Single weather observation."""
    timestamp: str
    fog_density: float      # g/m³
    rain_rate: float        # mm/hr
    surface_temp: float     # °C
    ambient_temp: float     # °C
    wind_speed: float       # m/s
    humidity: float         # %
    pressure: float         # hPa


class MockWeatherAPI:
    """Mock weather service that generates realistic synthetic data."""
    
    def __init__(self):
        self.base_seed = 42
        
    def _get_seasonal_bias(self, lat: float, lon: float, timestamp: datetime) -> Dict[str, float]:
        """Apply seasonal and geographic biases to weather generation."""
        # Simplified seasonal model
        day_of_year = timestamp.timetuple().tm_yday
        season_factor = math.sin(2 * math.pi * (day_of_year - 80) / 365)
        
        # Latitude effect on temperature
        temp_bias = -abs(lat) * 0.6  # Colder at higher latitudes
        
        # Coastal vs inland effects (simplified by longitude)
        coastal_factor = math.sin(lon * math.pi / 180) * 0.3
        
        return {
            "temp_bias": temp_bias + season_factor * 10,
            "fog_bias": coastal_factor * 0.5,  # More fog near coasts
            "rain_bias": abs(season_factor) * 2,  # More rain in wet seasons
        }
    
    def get_current_weather(self, lat: float, lon: float, timestamp: Optional[datetime] = None) -> WeatherCondition:
        """Get current weather conditions for a location."""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Seed random generator for consistent results
        random.seed(self.base_seed + int(lat * 1000) + int(lon * 1000) + timestamp.hour)
        
        biases = self._get_seasonal_bias(lat, lon, timestamp)
        
        # Generate base weather
        base_temp = 20 + biases["temp_bias"]
        ambient_temp = base_temp + random.gauss(0, 8)
        
        # Surface temperature influenced by ambient + heating
        surface_temp = ambient_temp + random.uniform(0, 15)
        
        # Fog density (more likely in certain conditions)
        fog_base = 0.1 + biases["fog_bias"]
        if random.random() < 0.3:  # 30% chance of significant fog
            fog_density = fog_base + random.expovariate(2.0)  # lambda = 1/mean
        else:
            fog_density = fog_base * random.uniform(0.1, 1.0)
        fog_density = max(0, min(fog_density, 5.0))
        
        # Rain rate (exponential distribution)
        rain_prob = 0.2 + biases["rain_bias"] * 0.1
        if random.random() < rain_prob:
            rain_rate = random.expovariate(1.0/3.0)  # lambda = 1/mean
        else:
            rain_rate = 0
        rain_rate = min(rain_rate, 50.0)
        
        # Additional weather variables
        wind_speed = random.expovariate(1.0/3.0)  # lambda = 1/mean
        humidity = max(30, min(100, 70 + random.gauss(0, 20)))
        pressure = 1013.25 + random.gauss(0, 10)
        
        return WeatherCondition(
            timestamp=timestamp.isoformat(),
            fog_density=round(fog_density, 3),
            rain_rate=round(rain_rate, 2),
            surface_temp=round(surface_temp, 1),
            ambient_temp=round(ambient_temp, 1),
            wind_speed=round(wind_speed, 1),
            humidity=round(humidity, 1),
            pressure=round(pressure, 2)
        )
    
    def get_historical_weather(self, lat: float, lon: float, days_back: int = 7) -> List[WeatherCondition]:
        """Get historical weather data for the past N days."""
        conditions = []
        for i in range(days_back):
            timestamp = datetime.now() - timedelta(days=i)
            condition = self.get_current_weather(lat, lon, timestamp)
            conditions.append(condition)
        return conditions
    
    def get_forecast(self, lat: float, lon: float, days_ahead: int = 7) -> List[WeatherCondition]:
        """Get weather forecast for the next N days."""
        conditions = []
        for i in range(1, days_ahead + 1):
            timestamp = datetime.now() + timedelta(days=i)
            condition = self.get_current_weather(lat, lon, timestamp)
            conditions.append(condition)
        return conditions
    
    def export_sample_data(self, filename: str, locations: Dict[str, tuple], days: int = 30):
        """Export sample weather data to JSON file for testing."""
        data = {}
        for name, (lat, lon) in locations.items():
            data[name] = []
            for i in range(days):
                timestamp = datetime.now() - timedelta(days=i)
                condition = self.get_current_weather(lat, lon, timestamp)
                data[name].append({
                    "timestamp": condition.timestamp,
                    "fog_density": condition.fog_density,
                    "rain_rate": condition.rain_rate,
                    "surface_temp": condition.surface_temp,
                    "ambient_temp": condition.ambient_temp,
                    "wind_speed": condition.wind_speed,
                    "humidity": condition.humidity,
                    "pressure": condition.pressure
                })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


# Predefined test locations
TEST_LOCATIONS = {
    "san_francisco": (37.7749, -122.4194),
    "new_york": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "singapore": (1.3521, 103.8198),
    "seattle": (47.6062, -122.3321),
    "miami": (25.7617, -80.1918),
    "denver": (39.7392, -104.9903),
    "phoenix": (33.4484, -112.0740)
}


# Usage examples and testing
if __name__ == "__main__":
    api = MockWeatherAPI()
    
    # Test current weather
    print("Current weather in San Francisco:")
    sf_weather = api.get_current_weather(37.7749, -122.4194)
    print(f"  Fog: {sf_weather.fog_density} g/m³")
    print(f"  Rain: {sf_weather.rain_rate} mm/hr")
    print(f"  Surface: {sf_weather.surface_temp}°C")
    print(f"  Ambient: {sf_weather.ambient_temp}°C")
    
    # Test historical data
    print("\nHistorical weather (last 3 days):")
    history = api.get_historical_weather(37.7749, -122.4194, days_back=3)
    for condition in history:
        print(f"  {condition.timestamp[:10]}: Fog={condition.fog_density}, Rain={condition.rain_rate}")
    
    # Export sample data for all test locations
    print("\nExporting sample data...")
    # api.export_sample_data("sample_weather_data.json", TEST_LOCATIONS)
    print("Sample weather data ready for development use.")