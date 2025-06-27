#!/bin/bash

# Build script for Render deployment
echo "🔧 Building FSOC Link Optimization System..."

# Upgrade pip and setuptools first
echo "⬆️ Upgrading build tools..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Train models if they don't exist
echo "🤖 Checking for trained models..."
if [ ! -f "models/power_predictor_random_forest.pkl" ] || [ ! -f "models/power_predictor_xgboost.pkl" ]; then
    echo "🏋️ Training prediction models..."
    python train_models.py --train
else
    echo "✅ Models already exist, skipping training"
fi

echo "✅ Build completed successfully!"
