"""
OpenMeteo Weather API integration for real weather data.

Provides real weather data from the OpenMeteo API service, which offers
free access to weather forecasts and current conditions without requiring
an API key. Falls back to mock data if the API is unavailable.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging
import time

from .mock_weather import WeatherCondition, MockWeatherAPI

logger = logging.getLogger(__name__)


class OpenMeteoWeatherAPI:
    """
    Real weather data provider using the OpenMeteo API.
    
    Fetches current weather conditions and forecasts from OpenMeteo's
    free weather API service. Includes automatic fallback to mock data
    if the service is unavailable.
    """
    
    def __init__(self, fallback_to_mock: bool = True):
        """
        Initialize OpenMeteo weather API client.
        
        Args:
            fallback_to_mock: Whether to fall back to mock data if API fails
        """
        self.base_url = "https://api.open-meteo.com/v1"
        self.fallback_to_mock = fallback_to_mock
        self.mock_api = MockWeatherAPI() if fallback_to_mock else None
        self.request_timeout = 10  # seconds
        self.last_request_time = 0
        self.min_request_interval = 1  # minimum seconds between requests
        
    def _rate_limit(self):
        """Simple rate limiting to be respectful to the API."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """
        Make a request to the OpenMeteo API with error handling.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters
            
        Returns:
            API response data or None if failed
        """
        try:
            self._rate_limit()
            url = f"{self.base_url}/{endpoint}"
            
            logger.debug(f"Making OpenMeteo API request to {url} with params: {params}")
            
            response = requests.get(
                url, 
                params=params, 
                timeout=self.request_timeout,
                headers={'User-Agent': 'FSOC-Simulation/1.0'}
            )
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"OpenMeteo API response received successfully")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"OpenMeteo API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse OpenMeteo API response: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in OpenMeteo API request: {e}")
            return None
    
    def _map_openmeteo_to_weather_condition(self, 
                                          data: Dict, 
                                          lat: float, 
                                          lon: float,
                                          timestamp: Optional[datetime] = None) -> WeatherCondition:
        """
        Convert OpenMeteo API response to our internal WeatherCondition format.
        
        Args:
            data: OpenMeteo API response data
            lat: Latitude for the weather data
            lon: Longitude for the weather data
            timestamp: Timestamp for the weather data
            
        Returns:
            WeatherCondition object with mapped data
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract current weather data
        current = data.get('current', {})
        
        # Map OpenMeteo fields to our internal format
        # Temperature (already in Celsius)
        ambient_temp = current.get('temperature_2m', 20.0)
        surface_temp = ambient_temp + 5.0  # Estimate surface heating
        
        # Humidity (already in percentage)
        humidity = current.get('relative_humidity_2m', 50.0)
        
        # Pressure (convert from hPa to hPa - already correct)
        pressure = current.get('surface_pressure', 1013.25)
        
        # Wind speed (convert from km/h to m/s)
        wind_speed_kmh = current.get('wind_speed_10m', 0.0)
        wind_speed = wind_speed_kmh / 3.6  # km/h to m/s
        
        # Precipitation (mm/h - already correct)
        rain_rate = current.get('precipitation', 0.0)
        
        # Fog density estimation from visibility and humidity
        visibility_m = current.get('visibility', 10000.0)  # meters
        fog_density = self._estimate_fog_density(visibility_m, humidity)
        
        return WeatherCondition(
            timestamp=timestamp.isoformat(),
            fog_density=fog_density,
            rain_rate=rain_rate,
            surface_temp=surface_temp,
            ambient_temp=ambient_temp,
            wind_speed=wind_speed,
            humidity=humidity,
            pressure=pressure
        )
    
    def _estimate_fog_density(self, visibility_m: float, humidity: float) -> float:
        """
        Estimate fog density from visibility and humidity.
        
        Args:
            visibility_m: Visibility in meters
            humidity: Relative humidity in percentage
            
        Returns:
            Estimated fog density in g/m³
        """
        # Basic fog density estimation
        # Lower visibility and higher humidity indicate more fog
        if visibility_m >= 10000:  # Clear conditions
            base_fog = 0.01
        elif visibility_m >= 5000:  # Light haze
            base_fog = 0.05
        elif visibility_m >= 1000:  # Moderate fog
            base_fog = 0.2
        elif visibility_m >= 500:   # Dense fog
            base_fog = 0.5
        else:  # Very dense fog
            base_fog = 1.0
        
        # Adjust based on humidity
        humidity_factor = max(0.1, (humidity - 60) / 40)  # Scale from 60-100% humidity
        fog_density = base_fog * max(0.1, humidity_factor)
        
        return min(fog_density, 5.0)  # Cap at 5 g/m³

    def get_current_weather(self, lat: float, lon: float,
                          timestamp: Optional[datetime] = None) -> WeatherCondition:
        """
        Get current weather conditions for a location.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            timestamp: Optional timestamp (for compatibility, uses current time)

        Returns:
            WeatherCondition object with current weather data
        """
        # Prepare API parameters for current weather
        params = {
            'latitude': lat,
            'longitude': lon,
            'current': [
                'temperature_2m',
                'relative_humidity_2m',
                'surface_pressure',
                'wind_speed_10m',
                'precipitation',
                'visibility'
            ],
            'timezone': 'auto',
            'temperature_unit': 'celsius',
            'wind_speed_unit': 'kmh',
            'precipitation_unit': 'mm'
        }

        # Make API request
        data = self._make_request('forecast', params)

        if data is None:
            # Fall back to mock data if API fails
            if self.fallback_to_mock and self.mock_api:
                logger.info(f"Falling back to mock weather data for {lat}, {lon}")
                return self.mock_api.get_current_weather(lat, lon, timestamp)
            else:
                raise Exception("OpenMeteo API failed and no fallback available")

        # Convert to our internal format
        return self._map_openmeteo_to_weather_condition(data, lat, lon, timestamp)

    def get_historical_weather(self, lat: float, lon: float,
                             days_back: int = 7) -> List[WeatherCondition]:
        """
        Get historical weather data for a location.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            days_back: Number of days of historical data to retrieve

        Returns:
            List of WeatherCondition objects for historical data
        """
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)

        # Prepare API parameters for historical data
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'daily': [
                'temperature_2m_max',
                'temperature_2m_min',
                'relative_humidity_2m_mean',
                'surface_pressure_mean',
                'wind_speed_10m_max',
                'precipitation_sum'
            ],
            'timezone': 'auto',
            'temperature_unit': 'celsius',
            'wind_speed_unit': 'kmh',
            'precipitation_unit': 'mm'
        }

        # Make API request
        data = self._make_request('forecast', params)

        if data is None:
            # Fall back to mock data if API fails
            if self.fallback_to_mock and self.mock_api:
                logger.info(f"Falling back to mock historical weather data for {lat}, {lon}")
                return self.mock_api.get_historical_weather(lat, lon, days_back)
            else:
                raise Exception("OpenMeteo API failed and no fallback available")

        # Convert daily data to our internal format
        conditions = []
        daily_data = data.get('daily', {})
        times = daily_data.get('time', [])

        for i, date_str in enumerate(times):
            # Create a datetime object for each day
            date_obj = datetime.fromisoformat(date_str)

            # Extract daily values
            temp_max = daily_data.get('temperature_2m_max', [20.0])[i] if i < len(daily_data.get('temperature_2m_max', [])) else 20.0
            temp_min = daily_data.get('temperature_2m_min', [15.0])[i] if i < len(daily_data.get('temperature_2m_min', [])) else 15.0
            humidity = daily_data.get('relative_humidity_2m_mean', [50.0])[i] if i < len(daily_data.get('relative_humidity_2m_mean', [])) else 50.0
            pressure = daily_data.get('surface_pressure_mean', [1013.25])[i] if i < len(daily_data.get('surface_pressure_mean', [])) else 1013.25
            wind_speed_kmh = daily_data.get('wind_speed_10m_max', [0.0])[i] if i < len(daily_data.get('wind_speed_10m_max', [])) else 0.0
            precipitation = daily_data.get('precipitation_sum', [0.0])[i] if i < len(daily_data.get('precipitation_sum', [])) else 0.0

            # Calculate average temperature and other derived values
            ambient_temp = (temp_max + temp_min) / 2
            surface_temp = ambient_temp + 5.0
            wind_speed = wind_speed_kmh / 3.6  # km/h to m/s
            rain_rate = precipitation / 24.0  # Convert daily sum to hourly rate

            # Estimate fog density (simplified for historical data)
            fog_density = self._estimate_fog_density(8000.0, humidity)  # Assume moderate visibility

            condition = WeatherCondition(
                timestamp=date_obj.isoformat(),
                fog_density=fog_density,
                rain_rate=rain_rate,
                surface_temp=surface_temp,
                ambient_temp=ambient_temp,
                wind_speed=wind_speed,
                humidity=humidity,
                pressure=pressure
            )
            conditions.append(condition)

        return conditions

    def get_forecast_weather(self, lat: float, lon: float,
                           days_ahead: int = 7) -> List[WeatherCondition]:
        """
        Get weather forecast for a location.

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            days_ahead: Number of days of forecast data to retrieve

        Returns:
            List of WeatherCondition objects for forecast data
        """
        # Prepare API parameters for forecast data
        params = {
            'latitude': lat,
            'longitude': lon,
            'forecast_days': min(days_ahead, 16),  # OpenMeteo supports up to 16 days
            'hourly': [
                'temperature_2m',
                'relative_humidity_2m',
                'surface_pressure',
                'wind_speed_10m',
                'precipitation',
                'visibility'
            ],
            'timezone': 'auto',
            'temperature_unit': 'celsius',
            'wind_speed_unit': 'kmh',
            'precipitation_unit': 'mm'
        }

        # Make API request
        data = self._make_request('forecast', params)

        if data is None:
            # Fall back to mock data if API fails
            if self.fallback_to_mock and self.mock_api:
                logger.info(f"Falling back to mock forecast weather data for {lat}, {lon}")
                # Mock API doesn't have forecast, so generate some based on current
                conditions = []
                base_condition = self.mock_api.get_current_weather(lat, lon)
                for i in range(days_ahead * 24):  # Hourly data
                    future_time = datetime.now() + timedelta(hours=i)
                    # Add some variation to the base condition
                    varied_condition = WeatherCondition(
                        timestamp=future_time.isoformat(),
                        fog_density=max(0, base_condition.fog_density + (i % 24 - 12) * 0.01),
                        rain_rate=max(0, base_condition.rain_rate + (i % 48 - 24) * 0.1),
                        surface_temp=base_condition.surface_temp + (i % 24 - 12) * 0.5,
                        ambient_temp=base_condition.ambient_temp + (i % 24 - 12) * 0.4,
                        wind_speed=max(0, base_condition.wind_speed + (i % 12 - 6) * 0.2),
                        humidity=max(10, min(100, base_condition.humidity + (i % 36 - 18) * 0.5)),
                        pressure=base_condition.pressure + (i % 72 - 36) * 0.1
                    )
                    conditions.append(varied_condition)
                return conditions
            else:
                raise Exception("OpenMeteo API failed and no fallback available")

        # Convert hourly data to our internal format
        conditions = []
        hourly_data = data.get('hourly', {})
        times = hourly_data.get('time', [])

        for i, time_str in enumerate(times):
            # Create a datetime object for each hour
            time_obj = datetime.fromisoformat(time_str.replace('T', ' '))

            # Extract hourly values (with bounds checking)
            ambient_temp = hourly_data.get('temperature_2m', [20.0])[i] if i < len(hourly_data.get('temperature_2m', [])) else 20.0
            humidity = hourly_data.get('relative_humidity_2m', [50.0])[i] if i < len(hourly_data.get('relative_humidity_2m', [])) else 50.0
            pressure = hourly_data.get('surface_pressure', [1013.25])[i] if i < len(hourly_data.get('surface_pressure', [])) else 1013.25
            wind_speed_kmh = hourly_data.get('wind_speed_10m', [0.0])[i] if i < len(hourly_data.get('wind_speed_10m', [])) else 0.0
            precipitation = hourly_data.get('precipitation', [0.0])[i] if i < len(hourly_data.get('precipitation', [])) else 0.0
            visibility_m = hourly_data.get('visibility', [10000.0])[i] if i < len(hourly_data.get('visibility', [])) else 10000.0

            # Calculate derived values
            surface_temp = ambient_temp + 5.0
            wind_speed = wind_speed_kmh / 3.6  # km/h to m/s
            fog_density = self._estimate_fog_density(visibility_m, humidity)

            condition = WeatherCondition(
                timestamp=time_obj.isoformat(),
                fog_density=fog_density,
                rain_rate=precipitation,
                surface_temp=surface_temp,
                ambient_temp=ambient_temp,
                wind_speed=wind_speed,
                humidity=humidity,
                pressure=pressure
            )
            conditions.append(condition)

        return conditions

    def test_api_connection(self) -> Dict[str, Union[bool, str]]:
        """
        Test the connection to the OpenMeteo API.

        Returns:
            Dictionary with connection status and details
        """
        try:
            # Test with a simple request to a known location
            test_data = self._make_request('forecast', {
                'latitude': 52.52,
                'longitude': 13.41,
                'current': 'temperature_2m'
            })

            if test_data is not None:
                return {
                    'success': True,
                    'message': 'OpenMeteo API connection successful',
                    'api_url': self.base_url
                }
            else:
                return {
                    'success': False,
                    'message': 'OpenMeteo API request failed',
                    'api_url': self.base_url
                }
        except Exception as e:
            return {
                'success': False,
                'message': f'OpenMeteo API connection error: {str(e)}',
                'api_url': self.base_url
            }


# Usage examples and testing
if __name__ == "__main__":
    # Test the OpenMeteo API
    api = OpenMeteoWeatherAPI()

    print("Testing OpenMeteo API connection...")
    connection_test = api.test_api_connection()
    print(f"Connection test: {connection_test}")

    if connection_test['success']:
        print("\nTesting current weather in San Francisco:")
        try:
            sf_weather = api.get_current_weather(37.7749, -122.4194)
            print(f"  Timestamp: {sf_weather.timestamp}")
            print(f"  Fog: {sf_weather.fog_density:.3f} g/m³")
            print(f"  Rain: {sf_weather.rain_rate:.1f} mm/hr")
            print(f"  Surface: {sf_weather.surface_temp:.1f}°C")
            print(f"  Ambient: {sf_weather.ambient_temp:.1f}°C")
            print(f"  Wind: {sf_weather.wind_speed:.1f} m/s")
            print(f"  Humidity: {sf_weather.humidity:.1f}%")
            print(f"  Pressure: {sf_weather.pressure:.1f} hPa")
        except Exception as e:
            print(f"  Error: {e}")

        print("\nTesting historical weather (last 3 days):")
        try:
            history = api.get_historical_weather(37.7749, -122.4194, days_back=3)
            for condition in history:
                print(f"  {condition.timestamp[:10]}: Temp={condition.ambient_temp:.1f}°C, Rain={condition.rain_rate:.1f}mm/hr")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("API connection failed, testing fallback to mock data...")
        try:
            sf_weather = api.get_current_weather(37.7749, -122.4194)
            print(f"  Fallback weather - Temp: {sf_weather.ambient_temp:.1f}°C, Fog: {sf_weather.fog_density:.3f} g/m³")
        except Exception as e:
            print(f"  Fallback error: {e}")
