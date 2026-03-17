import torch


class LSTMPredictor:

    def __init__(self, model):
        self.model = model

    def predict(self, features):

        tensor = torch.tensor(features).float()

        self.model.eval()

        with torch.no_grad():
            prediction = self.model(tensor)

        return prediction
