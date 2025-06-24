#!/usr/bin/env python3
"""
System status checker for FSOC Link Optimization system.
"""

import requests
import subprocess
import sys
from datetime import datetime

def check_port(port):
    """Check if a port is in use"""
    try:
        result = subprocess.run(
            ['netstat', '-an'], 
            capture_output=True, 
            text=True, 
            shell=True
        )
        return f":{port}" in result.stdout
    except:
        return False

def check_api_health():
    """Check API server health"""
    try:
        response = requests.get('http://localhost:8002/health', timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection refused"
    except Exception as e:
        return False, str(e)

def check_simulation():
    """Test simulation functionality"""
    test_data = {
        'lat_tx': 37.7749, 'lon_tx': -122.4194,
        'lat_rx': 37.7849, 'lon_rx': -122.4094,
        'height_tx': 20, 'height_rx': 15,
        'material_tx': 'white_paint', 'material_rx': 'aluminum',
        'fog_density': 0.1, 'rain_rate': 1.0,
        'surface_temp': 25, 'ambient_temp': 20,
        'wavelength_nm': 1550, 'tx_power_dbm': 20
    }
    
    try:
        response = requests.post('http://localhost:8002/simulate', json=test_data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            return result.get('success', False), result
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    print("🔍 FSOC Link Optimization System Status Check")
    print("=" * 50)
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if API port is open
    print("🌐 Network Status:")
    api_port_open = check_port(8002)
    frontend_port_open = check_port(5000)
    
    print(f"   API Port 8002: {'✅ OPEN' if api_port_open else '❌ CLOSED'}")
    print(f"   Frontend Port 5000: {'✅ OPEN' if frontend_port_open else '❌ CLOSED'}")
    print()
    
    # Check API health
    print("🏥 API Health Check:")
    api_healthy, health_info = check_api_health()
    
    if api_healthy:
        print("   ✅ API Server: HEALTHY")
        print(f"   📊 Models Loaded: {health_info.get('models_loaded', {})}")
        print(f"   🔧 System Info: {health_info.get('system_info', {})}")
    else:
        print(f"   ❌ API Server: {health_info}")
    print()
    
    # Test simulation
    if api_healthy:
        print("🧪 Simulation Test:")
        sim_working, sim_result = check_simulation()
        
        if sim_working:
            print("   ✅ Simulation Engine: WORKING")
            if isinstance(sim_result, dict) and 'results' in sim_result:
                results = sim_result['results']
                print(f"   📡 Test Result: {results.get('received_power_dbm', 'N/A'):.2f} dBm")
                print(f"   📏 Link Distance: {results.get('link_distance_km', 'N/A'):.2f} km")
        else:
            print(f"   ❌ Simulation Engine: {sim_result}")
    else:
        print("🧪 Simulation Test: SKIPPED (API not healthy)")
    
    print()
    print("📋 Summary:")
    if api_healthy and api_port_open:
        print("   🎉 System is READY for use!")
        print("   🌐 Frontend URL: http://localhost:5000")
        print("   📚 API Docs: http://localhost:8002/docs")
    else:
        print("   ⚠️  System needs attention:")
        if not api_port_open:
            print("      - Start the API server: python start_backend.py")
        if not api_healthy:
            print("      - Check API server logs for errors")

if __name__ == "__main__":
    main()