#!/bin/bash
# Render build script

set -o errexit

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p backend/data
mkdir -p models

# Generate training data and train models if not exists
python -c "
import os
from pathlib import Path

# Check if models exist
models_dir = Path('models')
if not (models_dir / 'power_predictor_xgboost.pkl').exists():
    print('Training models for production...')
    os.system('python train_models.py --train')
else:
    print('Models already exist, skipping training')
"

echo "Build completed successfully"