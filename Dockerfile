FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p models data logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Train models during build if they don't exist
RUN python -c "import os; from pathlib import Path; \
    models_dir = Path('models'); \
    rf_exists = (models_dir / 'power_predictor_random_forest.pkl').exists(); \
    xgb_exists = (models_dir / 'power_predictor_xgboost.pkl').exists(); \
    print(f'RF model exists: {rf_exists}, XGB model exists: {xgb_exists}'); \
    exec(open('train_models.py').read()) if not (rf_exists and xgb_exists) else print('Models already exist')" || echo "Model training failed, continuing..."

# Default command - use the new main.py entry point
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]