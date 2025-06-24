#!/usr/bin/env python3
"""
Simple script to start the FSOC simulation backend API server.
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

import uvicorn

def main():
    print("🚀 Starting FSOC Link Optimization API...")
    print("📚 API Documentation: http://localhost:8002/docs")
    print("🔍 Health Check: http://localhost:8002/health")
    print("📡 Ping Test: http://localhost:8002/ping")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "backend.api.main:app",
            host="0.0.0.0",
            port=8002,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())