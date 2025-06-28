# FSOC Link Optimization System

A comprehensive system for modeling and optimizing Free Space Optical Communication (FSOC) links under various atmospheric conditions.

To run this app locally, clone this repository and run ```python start_app.py```. 

Alternatively, you can also access it via a Render deployment. However, the instance is hosted on the free plan as of now, causing extensive delay. Running the application locally is recommended at this time. 

Render deployment: https://fsoc-global.onrender.com

## Overview

This system models real-world FSOC link degradation from:
- Fog and rain attenuation
- Thermal gradients and scintillation
- Mounting height and surface material effects
- Geographic and atmospheric conditions

The system uses advanced machine learning models to suggest optimal deployment parameters based on environmental conditions and link requirements.

## Machine Learning Models

The system employs multiple sophisticated machine learning models:

1. **PowerPredictorModel**: Predicts received optical power based on environmental and system parameters
   - **Neural Network**: A custom FSocNeuralNetwork architecture with dropout layers for uncertainty quantification
   - **XGBoost**: Gradient boosting for high-accuracy regression with feature importance analysis
   - **Random Forest**: Ensemble learning for robust prediction across varied conditions
   - **LightGBM**: Gradient boosting framework optimized for efficiency and accuracy

2. **DeploymentOptimizerModel**: Suggests optimal deployment parameters
   - Uses reinforcement learning concepts to find parameter combinations that maximize link performance
   - Implements grid search optimization with constraint satisfaction

These models are trained on physics-based simulation data and provide fast inference for real-time deployment optimization.

## Architecture

```
/fsoc-optimize/
├── /backend/
│   ├── /ingest/          # Weather, terrain input handling
│   ├── /physics/         # Beam modeling, atmospheric attenuation
│   ├── /simulation/      # Environment + physics composition
│   ├── /optimizer/       # ML-based deployment optimization
│   ├── /api/             # FastAPI server endpoints
│   └── /data/            # Local test datasets
├── /frontend/            # Flask-based web interface
├── /models/              # Trained ML models
├── /docs/                # Architecture and theory documentation
├── /tests/               # pytest and integration tests
├── requirements.txt
└── Dockerfile
```

## Features

- **Physics-Based Modeling**: Accurate atmospheric propagation simulation
- **Machine Learning Optimization**: ML-powered deployment parameter suggestions
- **Multi-Factor Analysis**: Weather, terrain, mounting considerations
- **Real-time Prediction**: Fast inference for deployment planning
- **Extensible Design**: Ready for real-world API integration

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fsoc-optimize.git
cd fsoc-optimize

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

```bash
# Train prediction models (if needed)
python train_models.py --train

# Start the application
python start_app.py

# Alternatively, use the batch file on Windows
start_app.bat  # Start with existing models
start_with_training.bat  # Train models first, then start
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# The application will be available at:
# - http://localhost:8000
```

## Usage Examples

### Simulation

Use the web interface to simulate FSOC link performance with custom parameters:
- Set environmental conditions (fog, rain, temperature)
- Configure link parameters (height, distance, materials)
- View predicted power and performance metrics

### Optimization

Get machine learning-driven recommendations for optimal deployment:
- Input your environmental constraints
- Specify performance requirements
- Receive optimized height, material, and configuration suggestions

### API Integration

Access the system programmatically:
```python
import requests

# Simulate a link
response = requests.post("http://localhost:8001/simulate", json={
    "environment": {
        "fog_density": 0.5,
        "rain_rate": 10.0,
        "temperature": 25.0
    },
    "link": {
        "distance": 1000,
        "tx_height": 50,
        "rx_height": 50,
        "material": "aluminum"
    }
})
result = response.json()
print(f"Predicted power: {result['power_dbm']} dBm")
```

## Model Training and Evaluation

The system includes tools for training and evaluating machine learning models:

```bash
# Train all model types with custom parameters
python train_models.py --samples 5000 --models neural_network xgboost random_forest

# Evaluate model performance
python evaluate_models.py --test-set real_world_data.csv
```

Performance metrics for each model type:
- R² Score: Coefficient of determination
- RMSE: Root Mean Square Error in dB
- MAE: Mean Absolute Error in dB
- CV Score: Cross-validation score

## Development

### Project Structure

- `backend/`: Core simulation and optimization logic
- `frontend/`: Flask web interface
- `models/`: Trained ML models
- `tests/`: Test suite

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend
```

## General Usage

Clone this repository and run ```python start_app.py```


The system is deployed via Render. Access it here: https://fsoc-global.onrender.com
**This system is hosted using Render's free plan, so you may experience severe delays. Running the application locally is recommended.**

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License

## Acknowledgments

Project created by [Harishvin Sasikumar](https://harishvin.framer.website/)
