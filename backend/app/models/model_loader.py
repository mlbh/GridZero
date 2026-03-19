from app.cloud.gcs_loader import GCSModelLoader
from app.config import GCS_BUCKET, XGB_MODEL_PATH, LSTM_MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH, MODEL_BUCKET
import joblib
import xgboost as xgb
from tensorflow import keras
import json
import xgboost as xgb
from google.cloud import storage

class ModelStore:

    def __init__(self):

        # self.pipeline = None
        self.lstm = None
        self.xgb = None
        #scaling issue
        self.x_scaler = None
        self.y_scaler = None

    def load_models(self):

        gcs = GCSModelLoader(GCS_BUCKET)

        # Download models
        lstm_path = gcs.download_file(LSTM_MODEL_PATH)
        xgb_path = gcs.download_file(XGB_MODEL_PATH)

        #need to pull scaler paths
        x_scaler_local = gcs.download_file(X_SCALER_PATH)
        y_scaler_local = gcs.download_file(Y_SCALER_PATH)


        #load LSTM .keras
        self.lstm = keras.models.load_model(lstm_path)

        #load xgboost .json
        self.xgb = xgb.Booster()
        self.xgb.load_model(str(xgb_path))
        # self.xgb_model = self.load_xgb_model_from_gcs(bucket_name="grid_zero_bucket", blob_path=XGB_MODEL_PATH)


        # 4. Load Scalers (.pkl) using joblib
        self.x_scaler = joblib.load(x_scaler_local)
        self.y_scaler = joblib.load(y_scaler_local)

    # def load_xgb_model_from_gcs(bucket_name: str, blob_path: str):
    #     """
    #     Load an XGBoost model from a Google Cloud Storage bucket (JSON format).

    #     Args:
    #         bucket_name: Name of the GCS bucket (e.g., 'my-bucket')
    #         blob_path:   Path to the model file within the bucket (e.g., 'models/xgb_model.json')

    #     Returns:
    #         A loaded xgb.Booster instance.
    #     """
    #     client = storage.Client()
    #     bucket = client.bucket(bucket_name)
    #     blob = bucket.blob(blob_path)

    #     model_bytes = blob.download_as_bytes()
    #     model_json = json.loads(model_bytes)

    #     model = xgb.Booster()
    #     model.load_model(bytearray(json.dumps(model_json).encode("utf-8")))

    #     return model

model_store = ModelStore()
