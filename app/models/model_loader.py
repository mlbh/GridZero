import joblib
import torch

from app.cloud.gcs_loader import GCSModelLoader


class ModelStore:

    def __init__(self, bucket_name):

        self.gcs = GCSModelLoader(bucket_name)

        self.pipeline = None
        self.lstm = None
        self.xgb = None

    def load(self):

        pipeline_path = self.gcs.download_file("models/preprocessing_pipeline.joblib")
        lstm_path = self.gcs.download_file("models/lstm_model.pt")
        xgb_path = self.gcs.download_file("models/xgb_model.joblib")

        self.pipeline = joblib.load(pipeline_path)
        self.lstm = torch.load(lstm_path, map_location="cpu")
        self.xgb = joblib.load(xgb_path)
