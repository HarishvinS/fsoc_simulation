"""
Comprehensive risk assessment for FSOC deployments.

Combines weather risk, equipment reliability, and other factors
to provide overall deployment risk assessment and mitigation strategies.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .weather_analysis import WeatherRiskAnalyzer, WeatherRisk
from .equipment_reliability import EquipmentReliabilityAnalyzer, EquipmentReliability


@dataclass
class DeploymentRisk:
    """Comprehensive deployment risk assessment."""
    overall_risk_level: str  # "low", "medium", "high", "extreme"
    overall_risk_score: float  # 0.0 to 1.0
    weather_risk: WeatherRisk
    equipment_risk: EquipmentReliability
    combined_availability: float  # Expected system availability
    risk_factors: List[str]
    mitigation_strategies: List[str]
    deployment_recommendations: List[str]
    cost_impact: Dict[str, float]


class ComprehensiveRiskAssessment:
    """
    Provides comprehensive risk assessment for FSOC deployments.
    """
    
    def __init__(self):
        self.weather_analyzer = WeatherRiskAnalyzer()
        self.equipment_analyzer = EquipmentReliabilityAnalyzer()
        
        # Risk weighting factors
        self.risk_weights = {
            'weather': 0.4,
            'equipment': 0.3,
            'operational': 0.2,
            'environmental': 0.1
        }
    
    def assess_deployment_risk(self,
                             location: Dict[str, float],
                             conditions: Dict[str, float],
                             equipment_config: Dict,
                             operational_factors: Optional[Dict] = None) -> DeploymentRisk:
        """
        Perform comprehensive risk assessment for FSOC deployment.
        
        Args:
            location: Deployment location (lat, lon)
            conditions: Environmental conditions
            equipment_config: Equipment configuration
            operational_factors: Operational considerations
            
        Returns:
            DeploymentRisk assessment
        """
        # Analyze weather risk
        weather_risk = self.weather_analyzer.analyze_weather_risk(
            lat=location['lat'],
            lon=location['lon'],
            avg_conditions=conditions
        )
        
        # Analyze equipment reliability
        equipment_risk = self.equipment_analyzer.analyze_equipment_reliability(
            deployment_conditions=conditions,
            equipment_config=equipment_config,
            environmental_factors={}
        )
        
        # Calculate operational risk
        operational_risk = self._assess_operational_risk(
            location, conditions, operational_factors or {}
        )
        
        # Calculate environmental risk
        environmental_risk = self._assess_environmental_risk(location, conditions)
        
        # Combine risks
        overall_risk_score = (
            weather_risk.risk_score * self.risk_weights['weather'] +
            (1.0 - equipment_risk.reliability_score) * self.risk_weights['equipment'] +
            operational_risk * self.risk_weights['operational'] +
            environmental_risk * self.risk_weights['environmental']
        )
        
        overall_risk_level = self._score_to_level(overall_risk_score)
        
        # Calculate combined availability
        combined_availability = self._calculate_combined_availability(
            weather_risk, equipment_risk
        )
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            weather_risk, equipment_risk, operational_risk, environmental_risk
        )
        
        # Generate mitigation strategies
        mitigation_strategies = self._generate_mitigation_strategies(
            weather_risk, equipment_risk, overall_risk_level
        )
        
        # Generate deployment recommendations
        deployment_recommendations = self._generate_deployment_recommendations(
            overall_risk_level, weather_risk, equipment_risk
        )
        
        # Calculate cost impact
        cost_impact = self._calculate_cost_impact(
            overall_risk_level, weather_risk, equipment_risk
        )
        
        return DeploymentRisk(
            overall_risk_level=overall_risk_level,
            overall_risk_score=overall_risk_score,
            weather_risk=weather_risk,
            equipment_risk=equipment_risk,
            combined_availability=combined_availability,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies,
            deployment_recommendations=deployment_recommendations,
            cost_impact=cost_impact
        )
    
    def _assess_operational_risk(self,
                               location: Dict,
                               conditions: Dict,
                               operational_factors: Dict) -> float:
        """Assess operational risk factors."""
        risk_score = 0.0
        
        # Distance-based risk
        # Longer links have higher operational complexity
        distance = operational_factors.get('link_distance_km', 1.0)
        if distance > 10:
            risk_score += 0.3
        elif distance > 5:
            risk_score += 0.2
        elif distance > 2:
            risk_score += 0.1
        
        # Height-based risk
        # Higher installations have more operational challenges
        avg_height = (
            operational_factors.get('height_tx', 20) + 
            operational_factors.get('height_rx', 20)
        ) / 2
        if avg_height > 50:
            risk_score += 0.2
        elif avg_height > 30:
            risk_score += 0.1
        
        # Accessibility risk
        accessibility = operational_factors.get('accessibility', 'normal')
        if accessibility == 'difficult':
            risk_score += 0.3
        elif accessibility == 'challenging':
            risk_score += 0.2
        
        return min(1.0, risk_score)
    
    def _assess_environmental_risk(self, location: Dict, conditions: Dict) -> float:
        """Assess environmental risk factors."""
        risk_score = 0.0
        
        # Latitude-based risk (extreme latitudes have more challenges)
        lat = abs(location.get('lat', 0))
        if lat > 60:  # Arctic/Antarctic regions
            risk_score += 0.4
        elif lat < 10:  # Equatorial regions
            risk_score += 0.2
        
        # Altitude-based risk (if available)
        altitude = location.get('altitude_m', 0)
        if altitude > 3000:  # High altitude
            risk_score += 0.3
        elif altitude > 1500:  # Moderate altitude
            risk_score += 0.1
        
        # Coastal proximity (if available)
        coastal_distance = location.get('coastal_distance_km', 100)
        if coastal_distance < 5:  # Very close to coast
            risk_score += 0.2
        elif coastal_distance < 20:  # Near coast
            risk_score += 0.1
        
        return min(1.0, risk_score)
    
    def _score_to_level(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score <= 0.25:
            return 'low'
        elif score <= 0.5:
            return 'medium'
        elif score <= 0.75:
            return 'high'
        else:
            return 'extreme'
    
    def _calculate_combined_availability(self,
                                       weather_risk: WeatherRisk,
                                       equipment_risk: EquipmentReliability) -> float:
        """Calculate combined system availability."""
        # Combine weather and equipment availability
        weather_availability = weather_risk.expected_availability
        equipment_availability = 1.0 - (equipment_risk.annual_failure_rate * 0.1)  # Simplified
        
        # Series system availability
        combined_availability = weather_availability * equipment_availability
        
        return max(0.5, combined_availability)
    
    def _identify_risk_factors(self,
                             weather_risk: WeatherRisk,
                             equipment_risk: EquipmentReliability,
                             operational_risk: float,
                             environmental_risk: float) -> List[str]:
        """Identify primary risk factors."""
        factors = []
        
        if weather_risk.risk_score > 0.5:
            factors.extend(weather_risk.primary_factors)
        
        if equipment_risk.reliability_score < 0.7:
            factors.append(f"Equipment reliability: {len(equipment_risk.critical_components)} critical components")
        
        if operational_risk > 0.3:
            factors.append("High operational complexity")
        
        if environmental_risk > 0.3:
            factors.append("Challenging environmental conditions")
        
        return factors
    
    def _generate_mitigation_strategies(self,
                                      weather_risk: WeatherRisk,
                                      equipment_risk: EquipmentReliability,
                                      overall_risk_level: str) -> List[str]:
        """Generate comprehensive mitigation strategies."""
        strategies = []
        
        # Weather mitigations
        strategies.extend(weather_risk.mitigation_recommendations)
        
        # Equipment mitigations
        strategies.extend(equipment_risk.maintenance_recommendations)
        
        # Overall risk mitigations
        if overall_risk_level in ['high', 'extreme']:
            strategies.extend([
                "Implement redundant communication paths",
                "Deploy backup power systems",
                "Establish 24/7 monitoring and rapid response procedures"
            ])
        
        if overall_risk_level == 'extreme':
            strategies.extend([
                "Consider alternative communication technologies as backup",
                "Implement automated failover systems",
                "Establish emergency communication protocols"
            ])
        
        return list(set(strategies))  # Remove duplicates
    
    def _generate_deployment_recommendations(self,
                                           overall_risk_level: str,
                                           weather_risk: WeatherRisk,
                                           equipment_risk: EquipmentReliability) -> List[str]:
        """Generate deployment-specific recommendations."""
        recommendations = []
        
        if overall_risk_level == 'low':
            recommendations.append("Deployment recommended with standard configuration")
        elif overall_risk_level == 'medium':
            recommendations.append("Deployment feasible with enhanced monitoring and maintenance")
        elif overall_risk_level == 'high':
            recommendations.append("Deployment challenging - implement comprehensive risk mitigation")
        else:  # extreme
            recommendations.append("Deployment not recommended without major risk mitigation measures")
        
        # Specific recommendations based on risks
        if weather_risk.risk_score > 0.6:
            recommendations.append("Consider seasonal deployment timing to minimize weather impact")
        
        if equipment_risk.reliability_score < 0.6:
            recommendations.append("Upgrade to higher-reliability components before deployment")
        
        if equipment_risk.mtbf_hours < 20000:
            recommendations.append("Implement redundant systems due to low equipment MTBF")
        
        return recommendations
    
    def _calculate_cost_impact(self,
                             overall_risk_level: str,
                             weather_risk: WeatherRisk,
                             equipment_risk: EquipmentReliability) -> Dict[str, float]:
        """Calculate cost impact of risks."""
        base_cost = 100000  # Base deployment cost in USD
        
        # Risk-based cost multipliers
        risk_multipliers = {
            'low': 1.0,
            'medium': 1.2,
            'high': 1.5,
            'extreme': 2.0
        }
        
        # Calculate cost components
        deployment_cost = base_cost * risk_multipliers[overall_risk_level]
        
        # Maintenance cost (annual)
        maintenance_cost = deployment_cost * 0.1 * (2.0 - equipment_risk.reliability_score)
        
        # Downtime cost (annual, based on availability)
        revenue_per_hour = 100  # Simplified revenue impact
        annual_hours = 8760
        downtime_hours = annual_hours * (1.0 - weather_risk.expected_availability)
        downtime_cost = downtime_hours * revenue_per_hour
        
        return {
            'deployment_cost_usd': deployment_cost,
            'annual_maintenance_cost_usd': maintenance_cost,
            'annual_downtime_cost_usd': downtime_cost,
            'total_annual_cost_usd': maintenance_cost + downtime_cost
        }
