"""
Flask frontend for FSOC Link Optimization.

Provides a web interface to interact with the FastAPI backend for:
- Link performance visualization
- Deployment parameter optimization
- Simulation results display
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
import json
import os
import time
from datetime import datetime

app = Flask(__name__)

# Check backend connection on startup
def check_backend_connection():
    """Check if backend API is available"""
    try:
        response = requests.get(f"{API_BASE_URL}/ping", timeout=2)
        if response.status_code == 200:
            print(f"Successfully connected to backend API at {API_BASE_URL}")
            return True
    except Exception as e:
        print(f"Warning: Could not connect to backend API: {e}")
    return False

# Try to connect to backend
backend_available = check_backend_connection()

# Configuration
API_BASE_URL = "http://localhost:8001"  # FastAPI backend URL

# Add retry logic for API connections
def api_request(method, endpoint, **kwargs):
    """Make API request with retry logic"""
    import requests
    from requests.exceptions import ConnectionError
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                return requests.get(f"{API_BASE_URL}{endpoint}", **kwargs)
            elif method.lower() == 'post':
                return requests.post(f"{API_BASE_URL}{endpoint}", **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
        except ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise

@app.route('/')
def index():
    """Home page with navigation to different features."""
    # Check backend connection
    backend_status = check_backend_connection()
    return render_template('index.html', backend_status=backend_status)

@app.route('/simulate', methods=['GET', 'POST'])
def simulate():
    """Page for running single link simulations."""
    if request.method == 'POST':
        # Extract form data
        try:
            # Create environment input from form data
            env_data = {
                "lat_tx": float(request.form['lat_tx']),
                "lon_tx": float(request.form['lon_tx']),
                "height_tx": float(request.form['height_tx']),
                "material_tx": request.form['material_tx'],
                
                "lat_rx": float(request.form['lat_rx']),
                "lon_rx": float(request.form['lon_rx']),
                "height_rx": float(request.form['height_rx']),
                "material_rx": request.form['material_rx'],
                
                "fog_density": float(request.form['fog_density']),
                "rain_rate": float(request.form['rain_rate']),
                "surface_temp": float(request.form['surface_temp']),
                "ambient_temp": float(request.form['ambient_temp']),
                
                "wavelength_nm": float(request.form.get('wavelength_nm', 1550)),
                "tx_power_dbm": float(request.form.get('tx_power_dbm', 20))
            }
            
            # Call backend API
            detailed_output = 'detailed_output' in request.form
            response = api_request(
                'post',
                "/simulate",
                json=env_data,
                params={"detailed_output": detailed_output}
            )
            
            if response.status_code == 200:
                result = response.json()
                return render_template('simulation_results.html', result=result)
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                return render_template('simulate.html', error=error_msg)
                
        except Exception as e:
            return render_template('simulate.html', error=str(e))
    
    # GET request - show form
    return render_template('simulate.html')

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    """Page for deployment parameter optimization."""
    if request.method == 'POST':
        try:
            # Create optimization request from form data
            opt_data = {
                "lat_tx": float(request.form['lat_tx']),
                "lon_tx": float(request.form['lon_tx']),
                "lat_rx": float(request.form['lat_rx']),
                "lon_rx": float(request.form['lon_rx']),
                
                "avg_fog_density": float(request.form.get('avg_fog_density', 0.1)),
                "avg_rain_rate": float(request.form.get('avg_rain_rate', 2.0)),
                "avg_surface_temp": float(request.form.get('avg_surface_temp', 25.0)),
                "avg_ambient_temp": float(request.form.get('avg_ambient_temp', 20.0)),
                
                "min_height": float(request.form.get('min_height', 5)),
                "max_height": float(request.form.get('max_height', 100)),
                "available_materials": request.form.getlist('available_materials'),
                
                "min_received_power_dbm": float(request.form.get('min_received_power_dbm', -30)),
                "reliability_target": float(request.form.get('reliability_target', 0.99))
            }
            
            # Call backend API
            response = api_request('post', "/optimize", json=opt_data)
            
            if response.status_code == 200:
                result = response.json()
                return render_template('optimization_results.html', result=result)
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                return render_template('optimize.html', error=error_msg)
                
        except Exception as e:
            return render_template('optimize.html', error=str(e))
    
    # GET request - show form
    return render_template('optimize.html')

@app.route('/health')
def health():
    """Display backend health status."""
    try:
        response = api_request('get', "/health")
        if response.status_code == 200:
            health_data = response.json()
            return render_template('health.html', health=health_data)
        else:
            return render_template('health.html', error=f"API Error: {response.status_code}")
    except Exception as e:
        return render_template('health.html', error=str(e))

@app.route('/examples')
def examples():
    """Load and run example simulations."""
    example_type = request.args.get('type', 'urban')
    
    try:
        # Call API to get example data
        if example_type == 'urban':
            response = api_request(
                'post',
                "/simulate",
                json={
                    "lat_tx": 37.7749, "lon_tx": -122.4194,
                    "lat_rx": 37.7849, "lon_rx": -122.4094,
                    "height_tx": 20, "height_rx": 15,
                    "material_tx": "white_paint", "material_rx": "aluminum",
                    "fog_density": 0.5, "rain_rate": 2.0,
                    "surface_temp": 25, "ambient_temp": 20,
                    "wavelength_nm": 1550, "tx_power_dbm": 20
                }
            )
        else:  # rural
            response = api_request(
                'post',
                "/simulate",
                json={
                    "lat_tx": 40.7128, "lon_tx": -74.0060,
                    "lat_rx": 40.7628, "lon_rx": -73.9560,
                    "height_tx": 50, "height_rx": 45,
                    "material_tx": "steel", "material_rx": "steel",
                    "fog_density": 0.1, "rain_rate": 5.0,
                    "surface_temp": 30, "ambient_temp": 25,
                    "wavelength_nm": 1550, "tx_power_dbm": 25
                }
            )
            
        if response.status_code == 200:
            result = response.json()
            return render_template('simulation_results.html', result=result, example=True)
        else:
            return render_template('index.html', error=f"API Error: {response.status_code}")
            
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)