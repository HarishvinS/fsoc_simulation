"""
Equipment reliability modeling for FSOC deployments.

Models equipment failure rates, maintenance requirements, and
provides reliability-based recommendations for deployment planning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class EquipmentReliability:
    """Equipment reliability assessment results."""
    mtbf_hours: float  # Mean Time Between Failures
    annual_failure_rate: float  # Failures per year
    maintenance_interval_months: int
    reliability_score: float  # 0.0 to 1.0
    critical_components: List[str]
    maintenance_recommendations: List[str]
    expected_lifetime_years: float


class EquipmentReliabilityAnalyzer:
    """
    Analyzes equipment reliability for FSOC deployments.
    """
    
    def __init__(self):
        # Component reliability data (MTBF in hours)
        self.component_mtbf = {
            'laser_diode': 50000,      # ~5.7 years
            'photodetector': 80000,    # ~9.1 years
            'optical_amplifier': 40000, # ~4.6 years
            'beam_steering': 30000,    # ~3.4 years
            'power_supply': 60000,     # ~6.8 years
            'cooling_system': 25000,   # ~2.9 years
            'control_electronics': 45000, # ~5.1 years
            'optical_components': 70000,  # ~8.0 years
            'mechanical_mount': 100000,   # ~11.4 years
            'weatherproofing': 35000      # ~4.0 years
        }
        
        # Environmental stress factors
        self.stress_factors = {
            'temperature': {
                'low': (-10, 10, 1.0),    # (min, max, factor)
                'normal': (10, 35, 1.0),
                'high': (35, 50, 1.3),
                'extreme': (50, 70, 1.8)
            },
            'humidity': {
                'low': (0, 40, 1.0),
                'normal': (40, 70, 1.1),
                'high': (70, 85, 1.4),
                'extreme': (85, 100, 2.0)
            },
            'vibration': {
                'low': 1.0,
                'normal': 1.1,
                'high': 1.3,
                'extreme': 1.6
            }
        }
    
    def analyze_equipment_reliability(self,
                                    deployment_conditions: Dict,
                                    equipment_config: Dict,
                                    environmental_factors: Dict) -> EquipmentReliability:
        """
        Analyze equipment reliability for specific deployment conditions.
        
        Args:
            deployment_conditions: Deployment environment (temp, humidity, etc.)
            equipment_config: Equipment configuration and specifications
            environmental_factors: Environmental stress factors
            
        Returns:
            EquipmentReliability assessment
        """
        # Calculate stress factors
        temp_stress = self._calculate_temperature_stress(
            deployment_conditions.get('avg_temp', 25)
        )
        humidity_stress = self._calculate_humidity_stress(
            deployment_conditions.get('humidity', 60)
        )
        vibration_stress = self._calculate_vibration_stress(
            deployment_conditions.get('wind_exposure', 'normal')
        )
        
        # Calculate component reliabilities
        component_reliabilities = {}
        critical_components = []
        
        for component, base_mtbf in self.component_mtbf.items():
            # Apply stress factors
            stressed_mtbf = base_mtbf / (temp_stress * humidity_stress * vibration_stress)
            component_reliabilities[component] = stressed_mtbf
            
            # Identify critical components (low MTBF)
            if stressed_mtbf < 30000:  # Less than ~3.4 years
                critical_components.append(component)
        
        # Calculate system reliability (series system - weakest link)
        system_mtbf = self._calculate_system_mtbf(component_reliabilities)
        annual_failure_rate = 8760 / system_mtbf  # Hours per year / MTBF
        
        # Calculate reliability score
        reliability_score = self._calculate_reliability_score(system_mtbf, critical_components)
        
        # Determine maintenance interval
        maintenance_interval = self._calculate_maintenance_interval(system_mtbf)
        
        # Generate maintenance recommendations
        maintenance_recommendations = self._generate_maintenance_recommendations(
            critical_components, deployment_conditions, system_mtbf
        )
        
        # Calculate expected lifetime
        expected_lifetime = self._calculate_expected_lifetime(
            system_mtbf, deployment_conditions
        )
        
        return EquipmentReliability(
            mtbf_hours=system_mtbf,
            annual_failure_rate=annual_failure_rate,
            maintenance_interval_months=maintenance_interval,
            reliability_score=reliability_score,
            critical_components=critical_components,
            maintenance_recommendations=maintenance_recommendations,
            expected_lifetime_years=expected_lifetime
        )
    
    def _calculate_temperature_stress(self, avg_temp: float) -> float:
        """Calculate temperature stress factor."""
        for level, (min_temp, max_temp, factor) in self.stress_factors['temperature'].items():
            if min_temp <= avg_temp < max_temp:
                return factor
        return 2.0  # Extreme conditions
    
    def _calculate_humidity_stress(self, humidity: float) -> float:
        """Calculate humidity stress factor."""
        for level, (min_hum, max_hum, factor) in self.stress_factors['humidity'].items():
            if min_hum <= humidity < max_hum:
                return factor
        return 2.0  # Extreme conditions
    
    def _calculate_vibration_stress(self, wind_exposure: str) -> float:
        """Calculate vibration stress factor."""
        return self.stress_factors['vibration'].get(wind_exposure, 1.1)
    
    def _calculate_system_mtbf(self, component_reliabilities: Dict[str, float]) -> float:
        """Calculate system MTBF using series reliability model."""
        # For series system: 1/MTBF_system = sum(1/MTBF_component)
        failure_rates = [1/mtbf for mtbf in component_reliabilities.values()]
        system_failure_rate = sum(failure_rates)
        return 1 / system_failure_rate
    
    def _calculate_reliability_score(self, system_mtbf: float, critical_components: List[str]) -> float:
        """Calculate overall reliability score."""
        # Base score from MTBF - use more realistic baseline for FSOC equipment
        # 10,000 hours (~1.14 years) = excellent reliability for complex optical systems
        # 5,000 hours (~0.57 years) = good reliability
        # 2,000 hours (~0.23 years) = poor reliability

        if system_mtbf >= 10000:
            base_score = 0.9 + min(0.1, (system_mtbf - 10000) / 40000)  # 90-100% for MTBF >= 10k hours
        elif system_mtbf >= 5000:
            base_score = 0.7 + (system_mtbf - 5000) / 5000 * 0.2  # 70-90% for 5k-10k hours
        elif system_mtbf >= 2000:
            base_score = 0.4 + (system_mtbf - 2000) / 3000 * 0.3  # 40-70% for 2k-5k hours
        else:
            base_score = max(0.1, system_mtbf / 2000 * 0.3)  # 10-40% for < 2k hours

        # Penalty for critical components (reduced penalty)
        critical_penalty = len(critical_components) * 0.05  # Reduced from 0.1 to 0.05

        return max(0.1, base_score - critical_penalty)
    
    def _calculate_maintenance_interval(self, system_mtbf: float) -> int:
        """Calculate recommended maintenance interval in months."""
        # Maintenance at 10% of MTBF, but at least every 6 months, max 24 months
        interval_hours = system_mtbf * 0.1
        interval_months = int(interval_hours / (30 * 24))  # Convert to months
        return max(6, min(24, interval_months))
    
    def _calculate_expected_lifetime(self, system_mtbf: float, conditions: Dict) -> float:
        """Calculate expected equipment lifetime in years."""
        # Base lifetime from MTBF
        base_lifetime = system_mtbf / 8760  # Convert hours to years
        
        # Environmental factors
        temp = conditions.get('avg_temp', 25)
        if temp > 40:
            base_lifetime *= 0.8  # Reduce lifetime in high temp
        elif temp < 0:
            base_lifetime *= 0.9  # Reduce lifetime in extreme cold
        
        # Maintenance factor
        maintenance_factor = 1.2  # Good maintenance extends life
        
        return base_lifetime * maintenance_factor
    
    def _generate_maintenance_recommendations(self,
                                            critical_components: List[str],
                                            conditions: Dict,
                                            system_mtbf: float) -> List[str]:
        """Generate maintenance recommendations."""
        recommendations = []
        
        if system_mtbf < 20000:  # Less than ~2.3 years
            recommendations.append("High-frequency maintenance required - consider redundant systems")
        
        if 'laser_diode' in critical_components:
            recommendations.append("Monitor laser diode performance closely - implement power monitoring")
        
        if 'cooling_system' in critical_components:
            recommendations.append("Regular cooling system maintenance critical - check filters monthly")
        
        if 'beam_steering' in critical_components:
            recommendations.append("Calibrate beam steering system quarterly")
        
        if conditions.get('avg_temp', 25) > 35:
            recommendations.append("Implement enhanced cooling due to high operating temperature")
        
        if conditions.get('humidity', 60) > 80:
            recommendations.append("Use desiccants and enhanced sealing for high humidity environment")
        
        if not recommendations:
            recommendations.append("Standard maintenance schedule sufficient for current conditions")
        
        # Add general recommendations
        recommendations.extend([
            f"Schedule maintenance every {self._calculate_maintenance_interval(system_mtbf)} months",
            "Implement condition monitoring for early failure detection",
            "Maintain spare parts inventory for critical components"
        ])
        
        return recommendations
