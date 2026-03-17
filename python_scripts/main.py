from fastapi import FastAPI
from forecastAPI import weather_preproc_no_fill, fetch_forecast




app = FastAPI()

#uvicornfast:app --reload



@app.get('/')
def index():
    return {'ok': True}

@app.get('/predict')
def predict(dsolar_percentage, gas_percentage, wind_percentage, nuclear_percentage):
    solar_input = int(solar_percentage)
    gas_input = int(gas_percentage)

    forecast = fetch_forecast()
    forecast = weather_preproc_no_fill(forecast)


    
    pred = model.predict()
    battery_func = #demand minus outputs (if total percentage <100)
    #if total % > 100 either function to prevent it or use battery func to mention 'surplus'


    return {'carbon intensity': pred,
            'how much battery needed to suppliment' : battery_func }
