#!/usr/bin/env python3
"""
Simple script to test the optimization API endpoint directly.
"""

import requests
import json

# Replace with your actual backend URL
BACKEND_URL = "https://your-backend-service-name.onrender.com"

def test_optimization():
    """Test the optimization endpoint with a simple request."""
    
    # Simple test data
    test_data = {
        "lat_tx": 37.7749,
        "lon_tx": -122.4194,
        "lat_rx": 37.7849,
        "lon_rx": -122.4094,
        "avg_fog_density": 0.1,
        "avg_rain_rate": 2.0,
        "avg_surface_temp": 25.0,
        "avg_ambient_temp": 20.0,
        "min_height": 5.0,
        "max_height": 50.0,
        "available_materials": ["concrete"],
        "min_received_power_dbm": -30.0,
        "reliability_target": 0.99
    }
    
    print("Testing optimization endpoint...")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Test data: {json.dumps(test_data, indent=2)}")
    print("-" * 50)
    
    try:
        # Test health endpoint first
        print("1. Testing health endpoint...")
        health_response = requests.get(f"{BACKEND_URL}/health", timeout=10)
        print(f"Health status: {health_response.status_code}")
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Models loaded: {health_data.get('models_loaded', {})}")
        print()
        
        # Test optimization endpoint
        print("2. Testing optimization endpoint...")
        opt_response = requests.post(
            f"{BACKEND_URL}/optimize",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Optimization status: {opt_response.status_code}")
        
        if opt_response.status_code == 200:
            result = opt_response.json()
            print("✅ Optimization successful!")
            print(f"Response keys: {list(result.keys())}")
            if 'recommendations' in result:
                print(f"Recommendations keys: {list(result['recommendations'].keys())}")
        else:
            print("❌ Optimization failed!")
            print(f"Error: {opt_response.text[:500]}...")  # First 500 chars
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out - backend might be overloaded")
    except requests.exceptions.ConnectionError:
        print("❌ Connection error - backend might be down")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_optimization()
