#!/usr/bin/env python3
"""
Test script to verify optimization constraint handling and functionality.
"""

import requests
import json
import time

def test_optimization_constraints():
    """Test optimization with various constraint scenarios."""
    
    base_url = "http://localhost:8000"
    
    # Test case 1: Normal constraints that should be satisfiable
    print("=" * 60)
    print("TEST 1: Normal constraints (should find valid solutions)")
    print("=" * 60)
    
    test_data_1 = {
        "lat_tx": 37.7749,
        "lon_tx": -122.4194,
        "lat_rx": 37.7849,
        "lon_rx": -122.4094,
        "avg_fog_density": 0.3,
        "avg_rain_rate": 2.0,
        "avg_surface_temp": 25.0,
        "avg_ambient_temp": 20.0,
        "min_height": 10.0,
        "max_height": 50.0,
        "available_materials": ["white_paint", "aluminum", "steel"],
        "min_received_power_dbm": -40.0,
        "reliability_target": 0.95
    }
    
    response = requests.post(f"{base_url}/optimize", json=test_data_1)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        if result['success']:
            recommendations = result['recommendations']
            print(f"Recommended Tx Height: {recommendations.get('tx_height_m', 'N/A')} m")
            print(f"Recommended Rx Height: {recommendations.get('rx_height_m', 'N/A')} m")
            print(f"Predicted Power: {recommendations.get('expected_rx_power_dbm', 'N/A')} dBm")
            print(f"Constraints Met: {recommendations.get('constraints_met', 'N/A')}")
            print(f"Expected Reliability: {recommendations.get('expected_reliability', 'N/A'):.1%}")
            print(f"Confidence Score: {result.get('confidence_score', 'N/A'):.1%}")
            
            # Verify constraints are met
            tx_height = recommendations.get('tx_height_m', 0)
            rx_height = recommendations.get('rx_height_m', 0)
            predicted_power = recommendations.get('expected_rx_power_dbm', -100)
            
            print("\nConstraint Verification:")
            print(f"  Height constraints: {test_data_1['min_height']} <= {tx_height}, {rx_height} <= {test_data_1['max_height']}")
            print(f"  Power constraint: {predicted_power} >= {test_data_1['min_received_power_dbm']}")
            
            height_ok = (test_data_1['min_height'] <= tx_height <= test_data_1['max_height'] and 
                        test_data_1['min_height'] <= rx_height <= test_data_1['max_height'])
            power_ok = predicted_power >= test_data_1['min_received_power_dbm']
            
            print(f"  ✓ Heights within bounds: {height_ok}")
            print(f"  ✓ Power meets minimum: {power_ok}")
        else:
            print(f"Error: {result.get('error_message', 'Unknown error')}")
    else:
        print(f"Request failed: {response.text}")
    
    print("\n" + "=" * 60)
    print("TEST 2: Strict constraints (may not find solutions)")
    print("=" * 60)
    
    # Test case 2: Very strict constraints that may not be satisfiable
    test_data_2 = {
        "lat_tx": 37.7749,
        "lon_tx": -122.4194,
        "lat_rx": 37.7849,
        "lon_rx": -122.4094,
        "avg_fog_density": 2.0,  # High fog
        "avg_rain_rate": 15.0,   # Heavy rain
        "avg_surface_temp": 25.0,
        "avg_ambient_temp": 20.0,
        "min_height": 5.0,
        "max_height": 15.0,      # Limited height range
        "available_materials": ["concrete"],  # Limited materials
        "min_received_power_dbm": -10.0,     # Very high power requirement
        "reliability_target": 0.999          # Very high reliability
    }
    
    response = requests.post(f"{base_url}/optimize", json=test_data_2)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        if result['success']:
            recommendations = result['recommendations']
            print(f"Recommended Tx Height: {recommendations.get('tx_height_m', 'N/A')} m")
            print(f"Recommended Rx Height: {recommendations.get('rx_height_m', 'N/A')} m")
            print(f"Predicted Power: {recommendations.get('expected_rx_power_dbm', 'N/A')} dBm")
            print(f"Constraints Met: {recommendations.get('constraints_met', 'N/A')}")
            print(f"Expected Reliability: {recommendations.get('expected_reliability', 'N/A'):.1%}")
            print(f"Confidence Score: {result.get('confidence_score', 'N/A'):.1%}")
            
            if recommendations.get('warning'):
                print(f"Warning: {recommendations['warning']}")
        else:
            print(f"Error: {result.get('error_message', 'Unknown error')}")
    else:
        print(f"Request failed: {response.text}")
    
    print("\n" + "=" * 60)
    print("TEST 3: Long distance link")
    print("=" * 60)
    
    # Test case 3: Long distance link
    test_data_3 = {
        "lat_tx": 37.7749,
        "lon_tx": -122.4194,
        "lat_rx": 37.8749,  # Further away
        "lon_rx": -122.3094,
        "avg_fog_density": 0.1,
        "avg_rain_rate": 1.0,
        "avg_surface_temp": 25.0,
        "avg_ambient_temp": 20.0,
        "min_height": 20.0,
        "max_height": 100.0,
        "available_materials": ["white_paint", "aluminum"],
        "min_received_power_dbm": -35.0,
        "reliability_target": 0.90
    }
    
    response = requests.post(f"{base_url}/optimize", json=test_data_3)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Success: {result['success']}")
        if result['success']:
            recommendations = result['recommendations']
            print(f"Recommended Tx Height: {recommendations.get('tx_height_m', 'N/A')} m")
            print(f"Recommended Rx Height: {recommendations.get('rx_height_m', 'N/A')} m")
            print(f"Predicted Power: {recommendations.get('expected_rx_power_dbm', 'N/A')} dBm")
            print(f"Expected Reliability: {recommendations.get('expected_reliability', 'N/A'):.1%}")
            print(f"Confidence Score: {result.get('confidence_score', 'N/A'):.1%}")
            
            # Check if cost features are removed
            if 'cost_impact' in recommendations.get('risk_assessment', {}):
                print("❌ ERROR: Cost features still present in results!")
            else:
                print("✓ Cost features successfully removed")
        else:
            print(f"Error: {result.get('error_message', 'Unknown error')}")
    else:
        print(f"Request failed: {response.text}")

if __name__ == "__main__":
    print("Testing Optimization Constraint Handling")
    print("Make sure backend is running on localhost:8000")
    print()
    
    try:
        test_optimization_constraints()
        print("\n" + "=" * 60)
        print("✓ All tests completed!")
        print("Check the results above to verify:")
        print("1. Constraints are properly respected")
        print("2. Predictions are meaningful and realistic")
        print("3. Cost features have been removed")
        print("4. UI shows consistent design (check browser)")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
