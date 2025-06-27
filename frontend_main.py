#!/usr/bin/env python3
"""
Frontend-only entry point for FSOC Link Optimization system.
This runs just the Flask frontend service.
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
sys.path.append(str(Path(__file__).parent / "frontend"))

# Set environment variables for production
os.environ["ENVIRONMENT"] = "production"

# Set the backend API URL - this will be your backend service URL
BACKEND_URL = os.environ.get("BACKEND_API_URL", "https://fsoc-backend.onrender.com")
os.environ["API_BASE_URL"] = BACKEND_URL

# Import the Flask app
from frontend.app import app

# Export the app for gunicorn
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting FSOC Frontend on port {port}")
    logger.info(f"🔗 Backend API URL: {BACKEND_URL}")
    logger.info("🌐 Frontend available at: /")
    
    # Use gunicorn for production
    import subprocess
    cmd = [
        "gunicorn",
        "--bind", f"0.0.0.0:{port}",
        "--workers", "2",
        "--timeout", "120",
        "--log-level", "info",
        "frontend_main:app"
    ]
    subprocess.run(cmd)
