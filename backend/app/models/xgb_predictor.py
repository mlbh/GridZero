import xgboost as xgb
import numpy as np
import pandas as pd
class XGBPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, features: pd.DataFrame):

        numeric_features = features.select_dtypes(include=['number'])

        dmatrix = xgb.DMatrix(numeric_features)

        prediction = self.model.predict(dmatrix)

        return prediction
