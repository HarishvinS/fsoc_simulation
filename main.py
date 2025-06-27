#!/usr/bin/env python3
"""
Main entry point for FSOC Link Optimization system on Render.
This creates a single FastAPI application that serves both API and frontend.
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
sys.path.append(str(Path(__file__).parent / "frontend"))

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

# Import and configure the FastAPI app
from backend.api.main import app
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi import Request

# Mount static files from frontend
static_path = Path(__file__).parent / "frontend" / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Add a simple frontend route
@app.get("/frontend", response_class=HTMLResponse)
async def serve_frontend():
    """Serve a simple frontend page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FSOC Link Optimization System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .api-link { background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; }
            .api-link:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 FSOC Link Optimization System</h1>
            <p>Welcome to the Free Space Optical Communication Link Optimization System!</p>
            
            <h2>🔗 Quick Links</h2>
            <ul>
                <li><a href="/docs" class="api-link">📚 API Documentation</a></li>
                <li><a href="/health" class="api-link">🏥 Health Check</a></li>
                <li><a href="/ping" class="api-link">📡 Ping Test</a></li>
            </ul>
            
            <h2>🎯 Features</h2>
            <ul>
                <li><strong>Physics-Based Modeling:</strong> Accurate atmospheric propagation simulation</li>
                <li><strong>AI Optimization:</strong> ML-powered deployment parameter suggestions</li>
                <li><strong>Multi-Factor Analysis:</strong> Weather, terrain, mounting considerations</li>
                <li><strong>Real-time Prediction:</strong> Fast inference for deployment planning</li>
            </ul>
            
            <h2>🚀 Getting Started</h2>
            <p>Visit the <a href="/docs">API Documentation</a> to explore available endpoints and try out the system.</p>
            
            <h2>📊 Example API Calls</h2>
            <pre>
# Health Check
GET /health

# Run Simulation
POST /simulate
{
  "location": {"latitude": 37.7749, "longitude": -122.4194},
  "link_distance_km": 1.0,
  "wavelength_nm": 1550,
  "tx_power_dbm": 20,
  "weather": {"visibility_km": 10, "temperature_c": 20}
}

# Get Optimization Recommendations
POST /optimize
{
  "location": {"latitude": 37.7749, "longitude": -122.4194},
  "requirements": {"min_availability": 0.99, "max_distance_km": 5}
}
            </pre>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Root route redirect to frontend
@app.get("/", response_class=HTMLResponse)
async def root_redirect():
    """Redirect root to frontend page."""
    return await serve_frontend()

# Export the app for uvicorn
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting FSOC system on port {port}")
    logger.info("📱 Frontend available at: /")
    logger.info("📚 API docs available at: /docs")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
