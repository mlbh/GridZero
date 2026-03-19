import xgboost as xgb
import numpy as np
import pandas as pd
class XGBPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, features_df):

        predictions = []

        for i in range(len(features_df)):
            row = features_df.iloc[i].copy()

            # 48 predictions overide 
            # xgb predictions not historical API values
            # carbon_lag_17520 from API
            if i >= 48:
                row["carbon_lag_48"] = predictions[i - 48]

            # predict for a single row
            X = pd.DataFrame([row])
            pred = self.model.predict(X)
            predictions.append(float(pred[0]))



        predictions = self.model.predict(features)

        return float(predictions.tolist())
