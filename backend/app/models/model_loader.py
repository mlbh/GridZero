import joblib
import torch

from app.cloud.gcs_loader import GCSModelLoader
from app.config import GCS_BUCKET, MODEL_VERSION


class ModelStore:

    def __init__(self):

        self.pipeline = None
        self.lstm = None
        self.xgb = None

    def load_models(self):

        gcs = GCSModelLoader(GCS_BUCKET)

        pipeline_path = gcs.download_model(
            f"models/{MODEL_VERSION}/preprocessing_pipeline.joblib"
        )

        lstm_path = gcs.download_model(
            f"models/{MODEL_VERSION}/lstm_model.pt"
        )

        xgb_path = gcs.download_model(
            f"models/{MODEL_VERSION}/xgb_model.joblib"
        )

        self.pipeline = joblib.load(pipeline_path)
        self.lstm = torch.load(lstm_path, map_location="cpu")
        self.xgb = joblib.load(xgb_path)


model_store = ModelStore()
