# Deployment Guide for FSOC Link Optimization System

This guide explains how to deploy the FSOC Link Optimization System to Render as a single web service.

## 🚀 Quick Deploy to Render

### Option 1: Using Render Dashboard (Recommended)

1. **Fork/Clone this repository** to your GitHub account

2. **Create a new Web Service** on [Render Dashboard](https://dashboard.render.com)

3. **Connect your repository** and provide these settings:
   - **Language**: `Python 3`
   - **Build Command**: `./build.sh`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`

4. **Set Environment Variables** (optional):
   ```
   ENVIRONMENT=production
   PYTHONDONTWRITEBYTECODE=1
   PYTHONUNBUFFERED=1
   ```

5. **Deploy!** - Your app will be available at `https://your-app-name.onrender.com`

### Option 2: Using render.yaml (Infrastructure as Code)

1. **Push the included `render.yaml`** to your repository
2. **Create a new Blueprint** on Render Dashboard
3. **Connect your repository** - Render will automatically use the `render.yaml` configuration

## 🔧 What Happens During Deployment

1. **Build Phase** (`build.sh`):
   - Installs Python dependencies from `requirements.txt`
   - Trains ML models if they don't exist
   - Prepares the application for production

2. **Runtime Phase** (`main.py`):
   - Loads trained models into memory
   - Starts FastAPI server with both API and frontend
   - Serves on the port provided by Render (`$PORT`)

## 📡 Available Endpoints

Once deployed, your app will have:

- **Frontend**: `https://your-app.onrender.com/` - Simple web interface
- **API Docs**: `https://your-app.onrender.com/docs` - Interactive API documentation
- **Health Check**: `https://your-app.onrender.com/health` - System status
- **API Endpoints**: All simulation and optimization endpoints

## 🏗️ Architecture

```
Single Render Web Service
├── FastAPI Backend (Port $PORT)
│   ├── /docs - API Documentation
│   ├── /health - Health Check
│   ├── /simulate - Run Simulations
│   ├── /optimize - Get Recommendations
│   └── /predict - AI Predictions
├── Static Frontend (Embedded)
│   ├── / - Main Page
│   └── /static/* - Static Assets
└── ML Models (Loaded in Memory)
    ├── Random Forest Predictor
    └── XGBoost Predictor
```

## 🔒 Production Considerations

### Security
- CORS is configured for production domains
- Environment variables are used for sensitive configuration
- No hardcoded secrets in the codebase

### Performance
- Models are loaded once at startup
- Health checks ensure service availability
- Uvicorn provides production-grade ASGI serving

### Monitoring
- Built-in health check endpoint
- Structured logging for debugging
- Error handling with graceful degradation

## 🛠️ Local Development vs Production

### Local Development
```bash
# Start both services separately
python start_app.py  # Runs on localhost:5000 (frontend) + localhost:8002 (backend)
```

### Production (Render)
```bash
# Single service
uvicorn main:app --host 0.0.0.0 --port $PORT  # Everything on one port
```

## 🔄 Updating Your Deployment

1. **Push changes** to your connected Git repository
2. **Render automatically rebuilds** and deploys
3. **Zero-downtime deployment** - old version runs until new one is ready

## 🆘 Troubleshooting

### Build Failures
- Check that `build.sh` has execute permissions: `chmod +x build.sh`
- Verify all dependencies are in `requirements.txt`
- Check build logs in Render Dashboard

### Runtime Issues
- Check `/health` endpoint for system status
- Review application logs in Render Dashboard
- Verify environment variables are set correctly

### Model Training Issues
- Models are trained during build if they don't exist
- Training can take 2-3 minutes on first deploy
- Check logs for training progress and errors

## 📊 Free Tier Limitations

Render's free tier includes:
- ✅ 750 hours/month of runtime
- ✅ Automatic SSL certificates
- ✅ Custom domains
- ⚠️ Services sleep after 15 minutes of inactivity
- ⚠️ 512MB RAM limit

For production use, consider upgrading to a paid plan for:
- Always-on services
- More RAM and CPU
- Faster builds
- Priority support

## 🎯 Next Steps

After deployment:
1. **Test the API** using the interactive docs at `/docs`
2. **Run simulations** to verify everything works
3. **Monitor performance** using Render's metrics
4. **Set up custom domain** (optional)
5. **Configure monitoring alerts** (optional)

Your FSOC Link Optimization System is now ready for production use! 🎉
