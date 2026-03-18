import xgboost as xgb
import numpy as np

class XGBPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, features):

        predictions = self.model.predict(features)

        return float(predictions.tolist())
