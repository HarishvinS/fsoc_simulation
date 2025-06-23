#!/usr/bin/env python3
"""
Startup script for FSOC Link Optimization Frontend server.
"""

import sys
from pathlib import Path

# Add frontend to path
sys.path.append(str(Path(__file__).parent / "frontend"))

if __name__ == "__main__":
    print("Starting FSOC Link Optimization Frontend...")
    print("Frontend available at: http://localhost:5000")
    print("Make sure the backend API is running (python start_api.py)")
    print("Press Ctrl+C to stop the server")
    
    from frontend.app import app
    app.run(host="0.0.0.0", port=5000, debug=True)