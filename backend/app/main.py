from fastapi import FastAPI
from services.weather_service import weather_preproc, fetch_forecast
from app.models.model_loader import model_store
from app.models.lstm_predictor import LSTMPredictor
from app.models.xgb_predictor import XGBPredictor
from app.utils.feature_builder import build_lstm_features, build_xgb_features


app = FastAPI()
#uvicornfast:app --reload

@app.on_event("startup")
def startup():

    model_store.load_models()

    global lstm_predictor, xgb_predictor

    lstm_predictor = LSTMPredictor(model_store.lstm)
    xgb_predictor = XGBPredictor(model_store.xgb)



@app.get("/predict")
def predict():

    weather_raw = fetch_forecast()

    weather_clean = weather_preproc(weather_raw)

    lstm_features = build_lstm_features(weather_clean)

    generation_prediction = lstm_predictor.predict(lstm_features)

    xgb_features = build_xgb_features(weather_clean, generation_prediction)

    carbon = xgb_predictor.predict(xgb_features)

    return {
        "generation_prediction": generation_prediction.tolist(),
        "carbon_intensity": carbon
    }
