@echo off
echo FSOC Link Optimization System
echo ============================
echo.
echo Step 1: Training prediction models...
python train_models.py --train
if %ERRORLEVEL% NEQ 0 (
    echo Error: Model training failed.
    pause
    exit /b 1
)

echo.
echo Step 2: Starting application...
python start_app.py
pause