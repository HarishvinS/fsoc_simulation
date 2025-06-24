"""
Production configuration for FSOC Link Optimization System.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Environment settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Model settings
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "backend" / "data"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# API settings
API_TITLE = "FSOC Link Optimization API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Production API for Free Space Optical Communication link optimization"