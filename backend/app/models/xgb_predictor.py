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

            # 48 predictions from lstm override carbon_lag_48
            # with our xgb predictions not historical API values
            # carbon_lag_17520 from API
            if i >= 48:
                row["carbon_lag_48"] = predictions[i - 48]

            # predict for a single row
            X = pd.DataFrame([row])
            pred = self.model.predict(X)
            predictions.append(float(pred[0]))



        predictions = self.model.predict(features)

        return float(predictions.tolist())
