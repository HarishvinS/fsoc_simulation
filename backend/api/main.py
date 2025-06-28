"""
FastAPI web service for FSOC link optimization.

Provides REST API endpoints for:
- Link performance prediction
- Deployment parameter optimization
- Batch simulation requests
- Model status and health checks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import json
import time
import asyncio
from datetime import datetime
import logging
import traceback
from pathlib import Path

# Import our modules
from backend.ingest.input_schema import (
    EnvironmentInput, BatchSimulationInput, OptimizationRequest,
    MaterialType, EXAMPLE_URBAN_LINK, EXAMPLE_RURAL_LINK, EXAMPLE_REAL_WEATHER_LINK
)
from backend.simulation.engine import FSocSimulationEngine
from backend.optimizer.models import ModelManager, PowerPredictorModel
from backend.ingest.mock_weather import MockWeatherAPI
from backend.ingest.weather_mapper import WeatherDataMapper
from backend.risk.assessment import ComprehensiveRiskAssessment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FSOC Link Optimization API",
    description="API for Free Space Optical Communication link performance prediction and deployment optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
# Configure CORS based on environment
import os
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

if ENVIRONMENT == "production":
    # More restrictive CORS for production
    allowed_origins = [
        "https://*.onrender.com",
        "https://localhost:*",
        "http://localhost:*"
    ] + ALLOWED_ORIGINS
else:
    # Allow all for development
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global instances
simulation_engine = FSocSimulationEngine()
weather_api = MockWeatherAPI()  # Keep for backward compatibility
weather_mapper = WeatherDataMapper()
model_manager = ModelManager()
risk_assessor = ComprehensiveRiskAssessment()

# Background task tracking
active_tasks = {}


# Startup event to load models
@app.on_event("startup")
async def startup_event():
    """Load trained models on startup."""
    logger.info("Loading trained models on startup...")

    try:
        # Get the models directory path
        models_dir = Path(__file__).parent.parent.parent / "models"

        # Check if model files exist
        model_types = ["neural_network", "random_forest", "xgboost"]
        models_loaded = 0

        for model_type in model_types:
            model_path = models_dir / f"power_predictor_{model_type}.pkl"

            if model_path.exists():
                try:
                    model = PowerPredictorModel(model_type)
                    model.load_model(str(model_path))
                    model_manager.power_predictors[model_type] = model
                    models_loaded += 1
                    logger.info(f"✓ {model_type.replace('_', ' ').title()} model loaded successfully")
                except Exception as e:
                    logger.error(f"✗ Failed to load {model_type} model: {e}")
            else:
                logger.warning(f"✗ {model_type.replace('_', ' ').title()} model not found at {model_path}")

        if models_loaded > 0:
            logger.info(f"🎉 Successfully loaded {models_loaded} prediction models")
            logger.info("🚀 Optimization feature is now available!")
        else:
            logger.warning("⚠️  No prediction models loaded. Optimization feature will not work.")
            logger.info("💡 To train models, run: python train_models.py --train")

    except Exception as e:
        logger.error(f"❌ Error during model loading: {e}")
        logger.error(traceback.format_exc())


# Response models
class SimulationResponse(BaseModel):
    """Response model for simulation results."""
    success: bool
    simulation_id: str
    timestamp: datetime
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None


class OptimizationResponse(BaseModel):
    """Response model for optimization recommendations."""
    success: bool
    optimization_id: str
    timestamp: datetime
    recommendations: Optional[Dict[str, Any]] = None
    confidence_score: Optional[float] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    api_version: str
    models_loaded: Dict[str, bool]
    system_info: Dict[str, Any]


class BatchTaskResponse(BaseModel):
    """Response for batch simulation tasks."""
    task_id: str
    status: str  # "queued", "running", "completed", "failed"
    progress: float
    estimated_completion: Optional[datetime] = None
    results_url: Optional[str] = None


# Dependency functions
async def get_simulation_engine():
    """Dependency to get simulation engine."""
    return simulation_engine


async def get_model_manager():
    """Dependency to get model manager."""
    return model_manager


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "FSOC Link Optimization API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        models_loaded = {
            "power_predictors": len(model_manager.power_predictors) > 0,
            "simulation_engine": simulation_engine is not None,
            "weather_api": weather_api is not None
        }
        
        system_info = {
            "active_tasks": len(active_tasks),
            "last_simulation_time": getattr(simulation_engine, 'last_simulation_time', 0),
            "api_startup_time": time.time()
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            api_version="1.0.0",
            models_loaded=models_loaded,
            system_info=system_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now(),
            api_version="1.0.0",
            models_loaded={"error": True},
            system_info={"error": str(e)}
        )

@app.get("/ping")
async def ping():
    """Simple ping endpoint for connection testing."""
    return {"status": "ok", "timestamp": str(datetime.now())}


@app.post("/simulate", response_model=SimulationResponse)
async def simulate_link(
    environment: EnvironmentInput,
    detailed_output: bool = Query(False, description="Include detailed layer-by-layer results"),
    engine: FSocSimulationEngine = Depends(get_simulation_engine)
):
    """
    Simulate FSOC link performance for given environmental conditions.
    
    Args:
        environment: Complete environment specification
        detailed_output: Include detailed layer results
        
    Returns:
        Comprehensive simulation results
    """
    simulation_id = f"sim_{int(time.time() * 1000)}"
    start_time = time.time()
    
    try:
        logger.info(f"Starting simulation {simulation_id}")
        
        # Run simulation
        results = engine.simulate_single_link(environment, detailed_output)
        
        execution_time = time.time() - start_time
        
        logger.info(f"Simulation {simulation_id} completed in {execution_time:.2f}s")
        
        return SimulationResponse(
            success=True,
            simulation_id=simulation_id,
            timestamp=datetime.now(),
            results=results,
            execution_time_seconds=execution_time
        )
        
    except Exception as e:
        logger.error(f"Simulation {simulation_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return SimulationResponse(
            success=False,
            simulation_id=simulation_id,
            timestamp=datetime.now(),
            error_message=str(e),
            execution_time_seconds=time.time() - start_time
        )


@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_deployment(
    request: OptimizationRequest,
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Get optimal deployment parameter recommendations.
    
    Args:
        request: Optimization request with location and constraints
        
    Returns:
        Optimization recommendations
    """
    optimization_id = f"opt_{int(time.time() * 1000)}"
    
    try:
        logger.info(f"Starting optimization {optimization_id}")
        logger.info(f"Request data: lat_tx={request.lat_tx}, lat_rx={request.lat_rx}")

        # Check if we have trained models
        if not manager.power_predictors:
            logger.error("No trained prediction models available")
            raise HTTPException(
                status_code=503,
                detail="No trained prediction models available. Please train models first."
            )

        logger.info(f"Available models: {list(manager.power_predictors.keys())}")
        
        # Create deployment optimizer
        optimizer = manager.create_deployment_optimizer()
        
        # Prepare base conditions
        base_conditions = {
            'input_lat_tx': request.lat_tx,
            'input_lon_tx': request.lon_tx,
            'input_lat_rx': request.lat_rx,
            'input_lon_rx': request.lon_rx,
            'input_fog_density': request.avg_fog_density,
            'input_rain_rate': request.avg_rain_rate,
            'input_surface_temp': request.avg_surface_temp,
            'input_ambient_temp': request.avg_ambient_temp,
            'input_wavelength_nm': 1550,  # Default
            'input_tx_power_dbm': 20,     # Default
            'input_height_tx': 20,        # Default height
            'input_height_rx': 20,        # Default height
            'input_material_tx': 'white_paint',  # Default material
            'input_material_rx': 'white_paint'   # Default material
        }
        
        # Prepare constraints
        constraints = {
            'min_height': request.min_height,
            'max_height': request.max_height,
            'available_materials': [m.value for m in request.available_materials],
            'min_received_power': request.min_received_power_dbm
        }

        # Add constraints to base conditions for optimization algorithm
        base_conditions['min_received_power'] = request.min_received_power_dbm
        base_conditions['reliability_target'] = request.reliability_target
        
        # Run optimization
        recommendations = optimizer.optimize_deployment(
            base_conditions, constraints, "max_power"
        )
        
        # Convert numpy types to native Python types for serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        recommendations = convert_numpy_types(recommendations)
        
        # Add field name mappings for frontend compatibility
        if 'height_tx' in recommendations:
            recommendations['tx_height_m'] = recommendations['height_tx']
        if 'height_rx' in recommendations:
            recommendations['rx_height_m'] = recommendations['height_rx']
        if 'material_tx' in recommendations:
            recommendations['tx_material'] = recommendations['material_tx']
        if 'material_rx' in recommendations:
            recommendations['rx_material'] = recommendations['material_rx']
        
        # Add fixed parameters from base conditions for frontend
        recommendations['tx_power_dbm'] = base_conditions.get('input_tx_power_dbm', 20)
        recommendations['wavelength_nm'] = base_conditions.get('input_wavelength_nm', 1550)
        recommendations['expected_rx_power_dbm'] = recommendations.get('predicted_power_dbm', -30)
        recommendations['link_margin_db'] = max(0, recommendations.get('predicted_power_dbm', -30) - request.min_received_power_dbm)
        recommendations['lat_tx'] = base_conditions.get('input_lat_tx')
        recommendations['lon_tx'] = base_conditions.get('input_lon_tx')
        recommendations['lat_rx'] = base_conditions.get('input_lat_rx')
        recommendations['lon_rx'] = base_conditions.get('input_lon_rx')
        recommendations['avg_fog_density'] = base_conditions.get('input_fog_density')
        recommendations['avg_rain_rate'] = base_conditions.get('input_rain_rate')
        recommendations['avg_surface_temp'] = base_conditions.get('input_surface_temp')
        recommendations['avg_ambient_temp'] = base_conditions.get('input_ambient_temp')

        # Perform comprehensive risk assessment
        try:
            location = {
                'lat': base_conditions.get('input_lat_tx'),
                'lon': base_conditions.get('input_lon_tx')
            }

            conditions = {
                'fog_density': base_conditions.get('input_fog_density', 0.1),
                'rain_rate': base_conditions.get('input_rain_rate', 2.0),
                'surface_temp': base_conditions.get('input_surface_temp', 25.0),
                'ambient_temp': base_conditions.get('input_ambient_temp', 20.0),
                'avg_temp': (base_conditions.get('input_surface_temp', 25.0) +
                           base_conditions.get('input_ambient_temp', 20.0)) / 2,
                'humidity': 60  # Default humidity
            }

            equipment_config = {
                'wavelength_nm': base_conditions.get('input_wavelength_nm', 1550),
                'tx_power_dbm': base_conditions.get('input_tx_power_dbm', 20),
                'height_tx': recommendations.get('height_tx', 20),
                'height_rx': recommendations.get('height_rx', 20)
            }

            operational_factors = {
                'link_distance_km': link_distance_km if 'link_distance_km' in locals() else 1.0,
                'height_tx': recommendations.get('height_tx', 20),
                'height_rx': recommendations.get('height_rx', 20),
                'accessibility': 'normal'
            }

            risk_assessment = risk_assessor.assess_deployment_risk(
                location=location,
                conditions=conditions,
                equipment_config=equipment_config,
                operational_factors=operational_factors
            )

            # Add risk assessment to recommendations (excluding cost information)
            recommendations['risk_assessment'] = {
                'overall_risk_level': risk_assessment.overall_risk_level,
                'overall_risk_score': float(risk_assessment.overall_risk_score),
                'weather_risk_level': risk_assessment.weather_risk.risk_level,
                'weather_risk_score': float(risk_assessment.weather_risk.risk_score),
                'equipment_reliability_score': float(risk_assessment.equipment_risk.reliability_score),
                'expected_availability': float(risk_assessment.combined_availability),
                'mtbf_hours': float(risk_assessment.equipment_risk.mtbf_hours),
                'annual_failure_rate': float(risk_assessment.equipment_risk.annual_failure_rate),
                'maintenance_interval_months': risk_assessment.equipment_risk.maintenance_interval_months,
                'risk_factors': risk_assessment.risk_factors,
                'mitigation_strategies': risk_assessment.mitigation_strategies[:5],  # Top 5 strategies
                'deployment_recommendations': risk_assessment.deployment_recommendations
                # Removed cost_impact as it's redundant
            }

            # Use risk assessment for confidence calculation
            risk_confidence = 1.0 - risk_assessment.overall_risk_score

        except Exception as e:
            logger.warning(f"Risk assessment failed: {e}")
            # Fallback to basic confidence calculation
            risk_confidence = 0.7
            recommendations['risk_assessment'] = {
                'overall_risk_level': 'medium',
                'overall_risk_score': 0.3,
                'error': 'Risk assessment unavailable'
            }

        # Calculate confidence score based on optimization quality and constraint satisfaction
        predicted_power = recommendations.get('predicted_power_dbm', -50)
        constraints_met = recommendations.get('constraints_met', True)

        # Check if optimization found valid configurations
        if not constraints_met or recommendations.get('warning'):
            # Low confidence if constraints weren't met
            confidence_score = 0.2
            recommendations['expected_reliability'] = 0.5  # Conservative estimate
            logger.warning(f"Optimization {optimization_id} found no configurations meeting all constraints")
        else:
            # Use the confidence level from the optimization if available
            if 'confidence_level' in recommendations:
                confidence_score = float(recommendations['confidence_level'])
            else:
                # Fallback confidence calculation
                # Power quality factor (0-1, where -10dBm = 1.0, -50dBm = 0.0)
                power_factor = max(0.0, min(1.0, (predicted_power + 50) / 40))

                # Environmental difficulty factor (higher fog/rain = lower confidence)
                fog_density = base_conditions.get('input_fog_density', 0.1)
                rain_rate = base_conditions.get('input_rain_rate', 2.0)
                env_difficulty = min(1.0, (fog_density * 2 + rain_rate * 0.1))  # Normalize to 0-1
                env_factor = max(0.2, 1.0 - env_difficulty)  # Keep minimum 20% confidence

                # Link distance factor (longer links = lower confidence)
                lat1, lon1 = base_conditions.get('input_lat_tx', 0), base_conditions.get('input_lon_tx', 0)
                lat2, lon2 = base_conditions.get('input_lat_rx', 0), base_conditions.get('input_lon_rx', 0)
                import math
                lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
                lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                a = (math.sin(dlat/2)**2 +
                     math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
                c = 2 * math.asin(math.sqrt(a))
                link_distance_km = 6371.0 * c
                distance_factor = max(0.3, 1.0 - (link_distance_km / 50))  # 50km = 30% confidence

                # Combined confidence score
                confidence_score = float(power_factor * env_factor * distance_factor)

            # Use estimated reliability from optimization if available
            if 'estimated_reliability' in recommendations:
                recommendations['expected_reliability'] = float(recommendations['estimated_reliability'])
            else:
                recommendations['expected_reliability'] = min(1.0, max(0.5, confidence_score))

        # Incorporate risk assessment if available
        if 'risk_assessment' in recommendations and 'error' not in recommendations['risk_assessment']:
            risk_factor = 1.0 - recommendations['risk_assessment']['overall_risk_score']
            availability_factor = recommendations['risk_assessment']['expected_availability']
            confidence_score = float(confidence_score * risk_factor * availability_factor)

        confidence_score = max(0.1, min(1.0, confidence_score))  # Clamp between 10% and 100%
        
        logger.info(f"Optimization {optimization_id} completed")
        
        return OptimizationResponse(
            success=True,
            optimization_id=optimization_id,
            timestamp=datetime.now(),
            recommendations=recommendations,
            confidence_score=confidence_score
        )
        
    except Exception as e:
        logger.error(f"Optimization {optimization_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return OptimizationResponse(
            success=False,
            optimization_id=optimization_id,
            timestamp=datetime.now(),
            error_message=str(e)
        )


@app.post("/batch-simulate", response_model=BatchTaskResponse)
async def batch_simulate(
    request: BatchSimulationInput,
    background_tasks: BackgroundTasks,
    engine: FSocSimulationEngine = Depends(get_simulation_engine)
):
    """
    Submit batch simulation request for processing.
    
    Args:
        request: Batch simulation parameters
        background_tasks: FastAPI background tasks
        
    Returns:
        Task tracking information
    """
    task_id = f"batch_{int(time.time() * 1000)}"
    
    # Add task to background processing
    background_tasks.add_task(
        process_batch_simulation,
        task_id,
        request,
        engine
    )
    
    # Track active task
    active_tasks[task_id] = {
        "status": "queued",
        "progress": 0.0,
        "start_time": time.time(),
        "estimated_completion": None
    }
    
    return BatchTaskResponse(
        task_id=task_id,
        status="queued",
        progress=0.0
    )


@app.get("/batch-status/{task_id}", response_model=BatchTaskResponse)
async def get_batch_status(task_id: str):
    """Get status of batch simulation task."""
    if task_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = active_tasks[task_id]
    
    return BatchTaskResponse(
        task_id=task_id,
        status=task_info["status"],
        progress=task_info["progress"],
        estimated_completion=task_info.get("estimated_completion"),
        results_url=task_info.get("results_url")
    )


@app.get("/weather/{lat}/{lon}")
async def get_weather(lat: float, lon: float, use_real_weather: bool = Query(default=True)):
    """Get current weather conditions for a location."""
    try:
        weather_summary = simulation_engine.get_weather_for_location(lat, lon, use_real_weather)
        return {
            "success": True,
            "location": {"latitude": lat, "longitude": lon},
            "weather_source": "real_api" if use_real_weather else "mock",
            "weather": weather_summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/test")
async def test_weather_connectivity():
    """Test connectivity to weather data sources."""
    try:
        test_results = simulation_engine.test_weather_connectivity()
        return {
            "success": True,
            "test_results": test_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/weather/sources")
async def get_weather_sources():
    """Get information about available weather data sources."""
    return {
        "success": True,
        "sources": {
            "mock": {
                "name": "Mock Weather API",
                "description": "Synthetic weather data for testing and development",
                "requires_api_key": False,
                "always_available": True
            },
            "openmeteo": {
                "name": "OpenMeteo API",
                "description": "Real weather data from OpenMeteo service",
                "requires_api_key": False,
                "website": "https://open-meteo.com",
                "features": ["Current weather", "Forecasts", "Historical data"]
            }
        }
    }


@app.get("/examples")
async def get_examples():
    """Get example configurations for testing."""
    return {
        "urban_link": EXAMPLE_URBAN_LINK.model_dump(),
        "rural_link": EXAMPLE_RURAL_LINK.model_dump(),
        "real_weather_link": EXAMPLE_REAL_WEATHER_LINK.model_dump(),
        "materials": [material.value for material in MaterialType]
    }


@app.post("/train-models")
async def train_models(
    background_tasks: BackgroundTasks,
    num_samples: int = Query(1000, ge=100, le=10000),
    manager: ModelManager = Depends(get_model_manager)
):
    """
    Train prediction models on simulation data.
    
    Args:
        num_samples: Number of simulation samples to generate for training
        
    Returns:
        Training task information
    """
    task_id = f"train_{int(time.time() * 1000)}"
    
    # Add training task to background processing
    background_tasks.add_task(
        train_models_background,
        task_id,
        num_samples,
        manager
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": f"Model training with {num_samples} samples queued"
    }


@app.get("/models/status")
async def get_model_status(manager: ModelManager = Depends(get_model_manager)):
    """Get status of trained models."""
    model_info = {}
    
    for model_type, model in manager.power_predictors.items():
        if model.metrics:
            model_info[model_type] = {
                "r2_score": model.metrics.r2_score,
                "rmse": model.metrics.rmse,
                "mae": model.metrics.mae,
                "training_samples": model.metrics.training_samples,
                "feature_importance": model.get_feature_importance()
            }
    
    return {
        "models_trained": len(manager.power_predictors),
        "model_details": model_info
    }


@app.post("/predict-deployment")
async def predict_deployment(input_data: EnvironmentInput):
    """
    Predict expected power and suggest optimal deployment parameters using AI models.
    """
    from backend.optimizer.infer import suggest_deployment
    result = suggest_deployment(input_data.dict())
    return result


# Background task functions
async def process_batch_simulation(task_id: str, 
                                 request: BatchSimulationInput,
                                 engine: FSocSimulationEngine):
    """Process batch simulation in background."""
    try:
        active_tasks[task_id]["status"] = "running"
        
        # Generate simulation dataset
        results = engine.batch_simulate(
            request.parameter_ranges,
            request.base_config,
            request.num_samples
        )
        
        # Save results
        results_filename = f"batch_results_{task_id}.csv"
        engine.save_simulation_results(results, results_filename)
        
        # Update task status
        active_tasks[task_id]["status"] = "completed"
        active_tasks[task_id]["progress"] = 1.0
        active_tasks[task_id]["results_url"] = f"/download/{results_filename}"
        
    except Exception as e:
        active_tasks[task_id]["status"] = "failed"
        active_tasks[task_id]["error"] = str(e)
        logger.error(f"Batch task {task_id} failed: {str(e)}")


async def train_models_background(task_id: str, 
                                num_samples: int,
                                manager: ModelManager):
    """Train models in background."""
    try:
        logger.info(f"Starting model training task {task_id}")
        
        # Generate training data
        from ..simulation.engine import create_training_dataset
        dataset_file = create_training_dataset(num_samples=num_samples)
        
        # Load training data
        import pandas as pd
        training_data = pd.read_csv(dataset_file)
        
        # Train models including neural networks
        results = manager.train_power_predictor(
            training_data,
            model_types=["neural_network", "xgboost", "random_forest"]
        )
        
        logger.info(f"Model training task {task_id} completed")
        
    except Exception as e:
        logger.error(f"Model training task {task_id} failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )


# Run the server
if __name__ == "__main__":
    import uvicorn
    
    print("Starting FSOC Link Optimization API...")
    print("Documentation available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )