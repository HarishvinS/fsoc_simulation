# FSOC Link Optimization System

A comprehensive system for modeling and optimizing Free Space Optical Communication (FSOC) links under various atmospheric conditions.

## Overview

This system models real-world FSOC link degradation from:
- Fog and rain attenuation
- Thermal gradients and scintillation
- Mounting height and surface material effects
- Geographic and atmospheric conditions

The AI-powered optimizer suggests optimal deployment parameters based on environmental conditions and link requirements.

## Architecture

```
/fsoc-optimize/
├── /backend/
│   ├── /ingest/          # Weather, terrain input handling
│   ├── /physics/         # Beam modeling, atmospheric attenuation
│   ├── /simulation/      # Environment + physics composition
│   ├── /optimizer/       # AI-based deployment optimization
│   ├── /api/            # FastAPI server endpoints
│   └── /data/           # Local test datasets
├── /frontend/           # React dashboard (future)
├── /models/            # Trained ML models
├── /docs/              # Architecture and theory documentation
├── /tests/             # pytest and integration tests
├── requirements.txt
└── Dockerfile
```

## Features

- **Physics-Based Modeling**: Accurate atmospheric propagation simulation
- **AI Optimization**: ML-powered deployment parameter suggestions
- **Multi-Factor Analysis**: Weather, terrain, mounting considerations
- **Real-time Prediction**: Fast inference for deployment planning
- **Extensible Design**: Ready for real-world API integration

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Train prediction models (if needed)
python train_models.py --train

# Start the application (both backend and frontend)
python start_app.py

# Alternatively, use the batch file on Windows
start_app.bat  # Start with existing models
start_with_training.bat  # Train models first, then start
```

The application will be available at:
- Frontend: http://localhost:5000
- Backend API: http://localhost:8001
- API Documentation: http://localhost:8001/docs

## Development Phases

- [x] Phase 0: Specification and setup
- [x] Phase 1: Input layer and validation
- [x] Phase 2: Beam physics engine
- [x] Phase 3: Data generation and AI training
- [x] Phase 4: API and interfaces
- [x] Phase 5: Frontend visualization
- [ ] Phase 6: Real-world API integration

## License

MIT License