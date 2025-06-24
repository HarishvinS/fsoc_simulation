#!/usr/bin/env python3
"""
Test script to verify the simulation engine is working.
"""

import requests
import json

def test_simulation():
    """Test the simulation endpoint"""
    # Test simulation data
    test_data = {
        'lat_tx': 37.7749, 'lon_tx': -122.4194,
        'lat_rx': 37.7849, 'lon_rx': -122.4094,
        'height_tx': 20, 'height_rx': 15,
        'material_tx': 'white_paint', 'material_rx': 'aluminum',
        'fog_density': 0.5, 'rain_rate': 2.0,
        'surface_temp': 25, 'ambient_temp': 20,
        'wavelength_nm': 1550, 'tx_power_dbm': 20
    }

    print("Testing simulation endpoint...")
    response = requests.post('http://localhost:8002/simulate', json=test_data)
    print(f'Simulation Status: {response.status_code}')
    
    if response.status_code == 200:
        result = response.json()
        print(f'Simulation Success: {result["success"]}')
        if result['success']:
            print(f'Received Power: {result["results"]["received_power_dbm"]:.2f} dBm')
            print(f'Link Distance: {result["results"]["link_distance_km"]:.2f} km')
            print("✅ Simulation engine is working correctly!")
        else:
            print(f"❌ Simulation failed: {result.get('error_message', 'Unknown error')}")
    else:
        print(f'❌ HTTP Error: {response.text}')

if __name__ == "__main__":
    test_simulation()