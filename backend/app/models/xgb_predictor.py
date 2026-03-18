import xgboost as xgb
import numpy as np

class XGBPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, features_df):

        """
        features_df: 48 rows of weather + generation features (no lags yet)
        Returns: list of 48 carbon intensity predictions
        """

        predictions = []

        for i in range (len(features_df)):
            row = features_df.iloc[i],copy()

        


        predictions = self.model.predict(features)

        return float(predictions.tolist())
