class XGBPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, features):

        if features.ndim == 1:
            features = features.reshape(1, -1)

        predictions = self.model.predict(features)

        return float(predictions.tolist())
