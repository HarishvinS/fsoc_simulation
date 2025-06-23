#!/usr/bin/env python3
"""
Combined startup script for FSOC Link Optimization system.
Launches both the FastAPI backend and Flask frontend servers.
"""

import sys
import os
import subprocess
import threading
import time
import logging
from pathlib import Path
import webbrowser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "backend"))
sys.path.append(str(Path(__file__).parent / "frontend"))

def start_backend():
    """Start the FastAPI backend server"""
    print("Starting FSOC Link Optimization API...")
    print("Documentation available at: http://localhost:8001/docs")
    
    # Import and run the FastAPI app
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )

def start_frontend():
    """Start the Flask frontend server"""
    print("Starting FSOC Link Optimization Frontend...")
    print("Frontend available at: http://localhost:5000")
    
    # Import and run the Flask app
    from frontend.app import app
    app.run(host="0.0.0.0", port=5000, debug=False)

def open_browser():
    """Open the browser after a short delay"""
    time.sleep(2)  # Wait for servers to start
    webbrowser.open("http://localhost:5000")

def load_models():
    """Load trained models into the model manager."""
    logger.info("Loading prediction models...")
    try:
        # Import the model loading function
        from train_models import load_models_into_manager
        
        # Load models
        success = load_models_into_manager()
        if success:
            logger.info("Models loaded successfully")
            return True
        else:
            logger.warning("Failed to load models. Optimization feature may not work.")
            return False
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

if __name__ == "__main__":
    print("Starting FSOC Link Optimization System...")
    
    # Load models first
    load_models()
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Wait a moment for backend to initialize
    time.sleep(1)
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start frontend in the main thread
    start_frontend()