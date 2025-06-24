#!/usr/bin/env python3
"""
Production entry point for FSOC Link Optimization System.
Optimized for Render deployment.
"""

import os
import sys
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import the FastAPI app
from backend.api.main import app

# For Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)