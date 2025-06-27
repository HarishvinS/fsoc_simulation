#!/usr/bin/env python3
"""
Backend-only entry point for FSOC Link Optimization system.
This runs just the FastAPI backend service.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "backend"))

# Set environment variables for production
os.environ["ENVIRONMENT"] = "production"

def load_models():
    """Load trained models into the model manager."""
    logger.info("Loading prediction models...")
    try:
        from train_models import load_models_into_manager
        success = load_models_into_manager()
        if success:
            logger.info("✓ Models loaded successfully")
        else:
            logger.warning("⚠️ Failed to load models. Optimization feature may not work.")
        return success
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        return False

# Load models on startup
load_models()

# Import the FastAPI app
from backend.api.main import app

# Export the app for uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting FSOC Backend API on port {port}")
    logger.info("📚 API docs available at: /docs")
    logger.info("🏥 Health check available at: /health")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
