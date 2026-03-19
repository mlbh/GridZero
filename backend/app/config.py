import os
from dotenv import load_dotenv

load_dotenv()

GCS_BUCKET = os.getenv("MODEL_BUCKET")
MODEL_BUCKET = os.getenv("MODEL_BUCKET")


XGB_MODEL_PATH = os.getenv("XGB_MODEL_PATH")
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH")

X_SCALER_PATH = os.getenv("X_SCALER_PATH", "weights/JM_lstm_x_scaler.pkl")
Y_SCALER_PATH = os.getenv("Y_SCALER_PATH", "weights/JM_lstm_y_scaler.pkl")

OPEN_METEO_URL = os.getenv("OPEN_METEO_URL")
ELEXON_URL = os.getenv("ELEXON_URL")
