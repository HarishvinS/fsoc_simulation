# FSOC Link Optimization - Production Deployment

## Render Deployment

### Quick Deploy
1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use these settings:
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker app:app`
   - **Environment**: `production`

### Environment Variables
Set these in Render dashboard:
- `ENVIRONMENT=production`
- `DEBUG=false`

### Manual Deployment
```bash
# 1. Clone repository
git clone <your-repo-url>
cd taara_simulation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (first time only)
python train_models.py --train

# 4. Start production server
gunicorn --bind 0.0.0.0:8000 --workers 2 --worker-class uvicorn.workers.UvicornWorker app:app
```

## Docker Deployment

```bash
# Build image
docker build -t fsoc-optimization .

# Run container
docker run -p 8000:8000 -e ENVIRONMENT=production fsoc-optimization
```

## Health Checks

- **Health**: `GET /health`
- **Ping**: `GET /ping`
- **Models Status**: `GET /models/status`

## API Endpoints

- **Simulate**: `POST /simulate`
- **Optimize**: `POST /optimize`
- **Weather**: `GET /weather/{lat}/{lon}`
- **Examples**: `GET /examples`

## Production Features

✅ **Automatic model loading on startup**
✅ **Production-grade error handling**
✅ **Comprehensive health checks**
✅ **Environment-based configuration**
✅ **Optimized for Render deployment**
✅ **Background task processing**
✅ **CORS enabled for frontend**

## Monitoring

The API includes built-in monitoring endpoints:
- System health and model status
- Request/response logging
- Error tracking and reporting