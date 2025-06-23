import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

class PowerPredictorModel:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, df: pd.DataFrame):
        X = df[[
            "lat_tx", "lon_tx", "height_tx", "material_tx",
            "lat_rx", "lon_rx", "height_rx", "material_rx",
            "fog_density", "rain_rate", "surface_temp", "ambient_temp", "wavelength_nm"
        ]]
        y = df["power_final"]
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class DeploymentOptimizerModel:
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, df: pd.DataFrame):
        X = df[[
            "lat_tx", "lon_tx", "fog_density", "rain_rate", "surface_temp", "ambient_temp", "wavelength_nm"
        ]]
        y = df[["height_tx"]]  # Could be extended to multi-output
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
