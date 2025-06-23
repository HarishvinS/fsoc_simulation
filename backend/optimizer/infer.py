from backend.optimizer.model_train import PowerPredictorModel, DeploymentOptimizerModel
import pandas as pd

def suggest_deployment(input_data: dict):
    # Load models (paths are placeholders)
    power_model = PowerPredictorModel()
    power_model.load('models/power_predictor.joblib')
    optimizer_model = DeploymentOptimizerModel()
    optimizer_model.load('models/deployment_optimizer.joblib')

    X = pd.DataFrame([input_data])
    power_pred = power_model.predict(X)[0]
    optimal_design = optimizer_model.predict(X)[0]
    return {
        "expected_power": power_pred,
        "suggested_mount_height": optimal_design[0],
        "suggested_material": input_data.get("material_tx", "concrete")
    }
