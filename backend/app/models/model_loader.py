from app.cloud.gcs_loader import GCSModelLoader
from app.config import GCS_BUCKET, XGB_MODEL_PATH, LSTM_MODEL_PATH, X_SCALER_PATH, Y_SCALER_PATH
import joblib
import xgboost as xgb
from tensorflow import keras


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

        # 4. Load Scalers (.pkl) using joblib
        self.x_scaler = joblib.load(x_scaler_local)
        self.y_scaler = joblib.load(y_scaler_local)


model_store = ModelStore()
