#!/usr/bin/env python3
"""
Startup script for FSOC Link Optimization API server.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

import uvicorn

if __name__ == "__main__":
    print("Starting FSOC Link Optimization API...")
    print("Documentation available at: http://localhost:8002/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )