"""
Weather risk analysis for FSOC deployments.

Analyzes weather patterns and provides risk assessments for different
environmental conditions and their impact on link performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import math


@dataclass
class WeatherRisk:
    """Weather risk assessment results."""
    risk_level: str  # "low", "medium", "high", "extreme"
    risk_score: float  # 0.0 to 1.0
    primary_factors: List[str]
    seasonal_variation: Dict[str, float]
    mitigation_recommendations: List[str]
    expected_availability: float  # 0.0 to 1.0


class WeatherRiskAnalyzer:
    """
    Analyzes weather-related risks for FSOC deployments.
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'fog_density': {'low': 0.2, 'medium': 0.5, 'high': 1.0, 'extreme': 2.0},
            'rain_rate': {'low': 2.0, 'medium': 5.0, 'high': 15.0, 'extreme': 30.0},
            'temperature_variation': {'low': 10, 'medium': 20, 'high': 30, 'extreme': 40},
            'wind_speed': {'low': 10, 'medium': 20, 'high': 35, 'extreme': 50}
        }
        
        # Seasonal risk factors by latitude
        self.seasonal_factors = {
            'fog': {
                'spring': 1.2, 'summer': 0.8, 'autumn': 1.5, 'winter': 1.3
            },
            'rain': {
                'spring': 1.3, 'summer': 1.1, 'autumn': 1.2, 'winter': 0.9
            },
            'temperature': {
                'spring': 1.0, 'summer': 1.4, 'autumn': 1.1, 'winter': 1.3
            }
        }
    
    def analyze_weather_risk(self, 
                           lat: float, 
                           lon: float,
                           avg_conditions: Dict[str, float],
                           historical_data: Optional[Dict] = None) -> WeatherRisk:
        """
        Analyze weather risk for a specific location and conditions.
        
        Args:
            lat: Latitude of deployment
            lon: Longitude of deployment
            avg_conditions: Average weather conditions
            historical_data: Optional historical weather data
            
        Returns:
            WeatherRisk assessment
        """
        risk_factors = []
        risk_scores = []
        
        # Analyze fog risk
        fog_risk = self._analyze_fog_risk(avg_conditions.get('fog_density', 0.1), lat, lon)
        risk_factors.append(f"Fog density: {fog_risk['level']}")
        risk_scores.append(fog_risk['score'])
        
        # Analyze rain risk
        rain_risk = self._analyze_rain_risk(avg_conditions.get('rain_rate', 2.0), lat, lon)
        risk_factors.append(f"Rain rate: {rain_risk['level']}")
        risk_scores.append(rain_risk['score'])
        
        # Analyze temperature variation risk
        temp_risk = self._analyze_temperature_risk(
            avg_conditions.get('surface_temp', 25.0),
            avg_conditions.get('ambient_temp', 20.0),
            lat
        )
        risk_factors.append(f"Temperature variation: {temp_risk['level']}")
        risk_scores.append(temp_risk['score'])
        
        # Calculate overall risk
        overall_risk_score = np.mean(risk_scores)
        risk_level = self._score_to_level(overall_risk_score)
        
        # Generate seasonal variation
        seasonal_variation = self._calculate_seasonal_variation(lat, avg_conditions)
        
        # Generate mitigation recommendations
        mitigation_recommendations = self._generate_weather_mitigations(
            avg_conditions, risk_level, risk_factors
        )
        
        # Estimate expected availability
        expected_availability = max(0.5, 1.0 - (overall_risk_score * 0.4))
        
        return WeatherRisk(
            risk_level=risk_level,
            risk_score=overall_risk_score,
            primary_factors=risk_factors,
            seasonal_variation=seasonal_variation,
            mitigation_recommendations=mitigation_recommendations,
            expected_availability=expected_availability
        )
    
    def _analyze_fog_risk(self, fog_density: float, lat: float, lon: float) -> Dict:
        """Analyze fog-related risk."""
        # Coastal areas have higher fog risk
        coastal_factor = 1.0
        if abs(lat) < 60:  # Not polar regions
            # Simplified coastal detection (would use real geographic data in production)
            if abs(lon) > 100:  # Pacific coast regions
                coastal_factor = 1.3
            elif abs(lon) < 30:  # Atlantic/European coast regions
                coastal_factor = 1.2
        
        adjusted_fog = fog_density * coastal_factor
        
        if adjusted_fog <= self.risk_thresholds['fog_density']['low']:
            return {'level': 'low', 'score': 0.1}
        elif adjusted_fog <= self.risk_thresholds['fog_density']['medium']:
            return {'level': 'medium', 'score': 0.3}
        elif adjusted_fog <= self.risk_thresholds['fog_density']['high']:
            return {'level': 'high', 'score': 0.6}
        else:
            return {'level': 'extreme', 'score': 0.9}
    
    def _analyze_rain_risk(self, rain_rate: float, lat: float, lon: float) -> Dict:
        """Analyze rain-related risk."""
        # Tropical regions have higher rain risk
        tropical_factor = 1.0
        if abs(lat) < 23.5:  # Tropical zone
            tropical_factor = 1.4
        elif abs(lat) < 40:  # Subtropical zone
            tropical_factor = 1.2
        
        adjusted_rain = rain_rate * tropical_factor
        
        if adjusted_rain <= self.risk_thresholds['rain_rate']['low']:
            return {'level': 'low', 'score': 0.1}
        elif adjusted_rain <= self.risk_thresholds['rain_rate']['medium']:
            return {'level': 'medium', 'score': 0.3}
        elif adjusted_rain <= self.risk_thresholds['rain_rate']['high']:
            return {'level': 'high', 'score': 0.6}
        else:
            return {'level': 'extreme', 'score': 0.9}
    
    def _analyze_temperature_risk(self, surface_temp: float, ambient_temp: float, lat: float) -> Dict:
        """Analyze temperature variation risk."""
        temp_variation = abs(surface_temp - ambient_temp)
        
        # Higher latitudes have more temperature variation
        latitude_factor = 1.0 + (abs(lat) / 90.0) * 0.5
        adjusted_variation = temp_variation * latitude_factor
        
        if adjusted_variation <= self.risk_thresholds['temperature_variation']['low']:
            return {'level': 'low', 'score': 0.1}
        elif adjusted_variation <= self.risk_thresholds['temperature_variation']['medium']:
            return {'level': 'medium', 'score': 0.3}
        elif adjusted_variation <= self.risk_thresholds['temperature_variation']['high']:
            return {'level': 'high', 'score': 0.6}
        else:
            return {'level': 'extreme', 'score': 0.9}
    
    def _score_to_level(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score <= 0.2:
            return 'low'
        elif score <= 0.4:
            return 'medium'
        elif score <= 0.7:
            return 'high'
        else:
            return 'extreme'
    
    def _calculate_seasonal_variation(self, lat: float, avg_conditions: Dict) -> Dict[str, float]:
        """Calculate seasonal risk variation."""
        base_risk = 0.3
        
        seasonal_risks = {}
        for season in ['spring', 'summer', 'autumn', 'winter']:
            fog_factor = self.seasonal_factors['fog'][season]
            rain_factor = self.seasonal_factors['rain'][season]
            temp_factor = self.seasonal_factors['temperature'][season]
            
            # Weight factors by latitude
            latitude_weight = abs(lat) / 90.0
            
            seasonal_risk = base_risk * (
                fog_factor * 0.4 + 
                rain_factor * 0.4 + 
                temp_factor * 0.2 * latitude_weight
            )
            
            seasonal_risks[season] = min(1.0, seasonal_risk)
        
        return seasonal_risks
    
    def _generate_weather_mitigations(self, 
                                    conditions: Dict, 
                                    risk_level: str, 
                                    risk_factors: List[str]) -> List[str]:
        """Generate weather-specific mitigation recommendations."""
        mitigations = []
        
        if risk_level in ['high', 'extreme']:
            mitigations.append("Consider backup communication links during adverse weather")
            mitigations.append("Implement adaptive power control to compensate for atmospheric losses")
        
        if conditions.get('fog_density', 0) > 0.5:
            mitigations.append("Install fog detection sensors for proactive link management")
            mitigations.append("Consider shorter wavelengths (850nm) for better fog penetration")
        
        if conditions.get('rain_rate', 0) > 10:
            mitigations.append("Implement rain fade mitigation with automatic gain control")
            mitigations.append("Consider diversity reception with multiple receivers")
        
        if abs(conditions.get('surface_temp', 25) - conditions.get('ambient_temp', 20)) > 15:
            mitigations.append("Install thermal management systems to reduce scintillation")
            mitigations.append("Use adaptive optics for beam stabilization")
        
        if not mitigations:
            mitigations.append("Weather conditions are favorable - standard deployment recommended")
        
        return mitigations
