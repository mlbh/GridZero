class XGBPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, features):

        prediction = self.model.predict(features)

        return float(prediction[0])
