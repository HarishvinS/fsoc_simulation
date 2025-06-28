# FSOC Link Optimization Frontend

This is a Flask-based frontend for the FSOC Link Optimization system. It provides a web interface to interact with the FastAPI backend for simulating and optimizing Free Space Optical Communication links.

## Features

- Link performance simulation with visualization
- Deployment parameter optimization
- System health monitoring
- Pre-configured example simulations

## Setup

1. Install the required dependencies:
   ```
   pip install -r ../requirements.txt
   ```

2. Make sure the backend API is running:
   ```
   python ../start_api.py
   ```

3. Start the frontend server:
   ```
   python ../start_frontend.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Pages

- **Home**: Navigation to different features
- **Simulate Link**: Run detailed FSOC link simulations with environmental parameters
- **AI-Powered Optimization**: Advanced neural network optimization with uncertainty quantification and comprehensive risk assessment
- **System Health**: Check the status of the backend API and model availability
- **Examples**: Pre-configured example simulations

## API Integration

The frontend communicates with the FastAPI backend running on port 8001. Make sure the backend is running before using the frontend.