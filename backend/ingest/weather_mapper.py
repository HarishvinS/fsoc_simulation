"""
Weather data mapping utilities for FSOC simulation.

Provides unified interface for weather data from different sources
(mock data, OpenMeteo API) and maps them to the internal format
used by the simulation engine.
"""

from typing import Dict, Optional, Union
from datetime import datetime
import logging

from .mock_weather import WeatherCondition, MockWeatherAPI
from .openmeteo_weather import OpenMeteoWeatherAPI
from .input_schema import EnvironmentInput

logger = logging.getLogger(__name__)


class WeatherDataMapper:
    """
    Unified weather data provider that handles both mock and real weather data.
    
    Automatically selects the appropriate weather source based on user preferences
    and provides a consistent interface for the simulation engine.
    """
    
    def __init__(self):
        """Initialize weather data mapper with both mock and real weather APIs."""
        self.mock_api = MockWeatherAPI()
        self.real_api = OpenMeteoWeatherAPI(fallback_to_mock=True)
        
    def get_weather_for_environment(self, environment: EnvironmentInput) -> WeatherCondition:
        """
        Get weather data for the given environment configuration.
        
        Args:
            environment: Environment input with location and weather preferences
            
        Returns:
            WeatherCondition object with weather data
        """
        if environment.use_real_weather:
            # Use real weather data from OpenMeteo API
            logger.info(f"Fetching real weather data for {environment.lat_tx}, {environment.lon_tx}")
            try:
                # Use transmitter location for weather data
                weather = self.real_api.get_current_weather(
                    environment.lat_tx, 
                    environment.lon_tx
                )
                logger.info(f"Successfully fetched real weather data")
                return weather
            except Exception as e:
                logger.warning(f"Failed to fetch real weather data: {e}")
                # Fall back to mock data
                return self.mock_api.get_current_weather(
                    environment.lat_tx, 
                    environment.lon_tx
                )
        else:
            # Use provided manual weather data
            logger.debug(f"Using manual weather data")
            return WeatherCondition(
                timestamp=datetime.now().isoformat(),
                fog_density=environment.fog_density,
                rain_rate=environment.rain_rate,
                surface_temp=environment.surface_temp,
                ambient_temp=environment.ambient_temp,
                wind_speed=5.0,  # Default wind speed
                humidity=60.0,   # Default humidity
                pressure=1013.25  # Default pressure
            )
    
    def get_weather_for_location(self, lat: float, lon: float, 
                               use_real_weather: bool = True) -> WeatherCondition:
        """
        Get weather data for a specific location.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            use_real_weather: Whether to use real weather data
            
        Returns:
            WeatherCondition object with weather data
        """
        if use_real_weather:
            try:
                return self.real_api.get_current_weather(lat, lon)
            except Exception as e:
                logger.warning(f"Failed to fetch real weather data: {e}")
                return self.mock_api.get_current_weather(lat, lon)
        else:
            return self.mock_api.get_current_weather(lat, lon)
    
    def populate_environment_with_weather(self, environment: EnvironmentInput) -> EnvironmentInput:
        """
        Populate environment with weather data if using real weather.
        
        Args:
            environment: Environment input that may need weather data
            
        Returns:
            Environment input with weather data populated
        """
        if environment.use_real_weather:
            # Fetch real weather data and populate the environment
            weather = self.get_weather_for_environment(environment)
            
            # Create a new environment with the fetched weather data
            env_dict = environment.dict()
            env_dict.update({
                'fog_density': weather.fog_density,
                'rain_rate': weather.rain_rate,
                'surface_temp': weather.surface_temp,
                'ambient_temp': weather.ambient_temp,
                'use_real_weather': True  # Keep the flag
            })
            
            return EnvironmentInput(**env_dict)
        else:
            # Return as-is if using manual weather data
            return environment
    
    def validate_weather_data(self, weather: WeatherCondition) -> bool:
        """
        Validate weather data for simulation use.
        
        Args:
            weather: Weather condition to validate
            
        Returns:
            True if weather data is valid for simulation
        """
        try:
            # Check for reasonable ranges
            if not (-50 <= weather.ambient_temp <= 60):
                logger.warning(f"Ambient temperature {weather.ambient_temp}°C is outside reasonable range")
                return False
            
            if not (-50 <= weather.surface_temp <= 80):
                logger.warning(f"Surface temperature {weather.surface_temp}°C is outside reasonable range")
                return False
            
            if not (0 <= weather.fog_density <= 10):
                logger.warning(f"Fog density {weather.fog_density} g/m³ is outside reasonable range")
                return False
            
            if not (0 <= weather.rain_rate <= 500):
                logger.warning(f"Rain rate {weather.rain_rate} mm/hr is outside reasonable range")
                return False
            
            if not (0 <= weather.humidity <= 100):
                logger.warning(f"Humidity {weather.humidity}% is outside reasonable range")
                return False
            
            if not (800 <= weather.pressure <= 1200):
                logger.warning(f"Pressure {weather.pressure} hPa is outside reasonable range")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating weather data: {e}")
            return False
    
    def get_weather_summary(self, weather: WeatherCondition) -> Dict[str, Union[str, float]]:
        """
        Get a human-readable summary of weather conditions.
        
        Args:
            weather: Weather condition to summarize
            
        Returns:
            Dictionary with weather summary
        """
        # Categorize conditions
        fog_level = "Clear"
        if weather.fog_density > 2.0:
            fog_level = "Dense fog"
        elif weather.fog_density > 0.5:
            fog_level = "Moderate fog"
        elif weather.fog_density > 0.1:
            fog_level = "Light fog"
        
        rain_level = "No rain"
        if weather.rain_rate > 10:
            rain_level = "Heavy rain"
        elif weather.rain_rate > 2:
            rain_level = "Moderate rain"
        elif weather.rain_rate > 0.1:
            rain_level = "Light rain"
        
        temp_desc = "Moderate"
        if weather.ambient_temp > 30:
            temp_desc = "Hot"
        elif weather.ambient_temp > 20:
            temp_desc = "Warm"
        elif weather.ambient_temp > 10:
            temp_desc = "Cool"
        elif weather.ambient_temp <= 10:
            temp_desc = "Cold"
        
        return {
            "timestamp": weather.timestamp,
            "temperature_description": temp_desc,
            "ambient_temp_c": weather.ambient_temp,
            "surface_temp_c": weather.surface_temp,
            "fog_description": fog_level,
            "fog_density_g_m3": weather.fog_density,
            "rain_description": rain_level,
            "rain_rate_mm_hr": weather.rain_rate,
            "humidity_percent": weather.humidity,
            "pressure_hpa": weather.pressure,
            "wind_speed_ms": weather.wind_speed
        }
    
    def test_weather_sources(self) -> Dict[str, Dict]:
        """
        Test both weather data sources.
        
        Returns:
            Dictionary with test results for each source
        """
        results = {}
        
        # Test mock API
        try:
            mock_weather = self.mock_api.get_current_weather(37.7749, -122.4194)
            results['mock_api'] = {
                'success': True,
                'weather': self.get_weather_summary(mock_weather)
            }
        except Exception as e:
            results['mock_api'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test real API
        try:
            real_weather = self.real_api.get_current_weather(37.7749, -122.4194)
            results['real_api'] = {
                'success': True,
                'weather': self.get_weather_summary(real_weather)
            }
        except Exception as e:
            results['real_api'] = {
                'success': False,
                'error': str(e)
            }
        
        # Test API connection
        results['api_connection'] = self.real_api.test_api_connection()
        
        return results


# Global instance for easy access
weather_mapper = WeatherDataMapper()


# Usage examples and testing
if __name__ == "__main__":
    mapper = WeatherDataMapper()
    
    print("Testing weather data sources...")
    test_results = mapper.test_weather_sources()
    
    for source, result in test_results.items():
        print(f"\n{source.upper()}:")
        if result.get('success'):
            if 'weather' in result:
                weather = result['weather']
                print(f"  Temperature: {weather['ambient_temp_c']:.1f}°C ({weather['temperature_description']})")
                print(f"  Fog: {weather['fog_density_g_m3']:.3f} g/m³ ({weather['fog_description']})")
                print(f"  Rain: {weather['rain_rate_mm_hr']:.1f} mm/hr ({weather['rain_description']})")
            else:
                print(f"  {result.get('message', 'Success')}")
        else:
            print(f"  Error: {result.get('error', result.get('message', 'Unknown error'))}")
