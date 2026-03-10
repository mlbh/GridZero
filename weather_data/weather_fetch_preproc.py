# imports required
from google.cloud import bigquery
import pandas as pd
import requests


def fetch_weather(start_date, end_date, latitude=51.5, longitude=-0.1):
    '''fetch api data from open-meteo archive,
    returns selected parameters hourly for London (based on long and lat).
    Dates in string format. See weather_data_notes.txt for information on selected params.'''

    url = 'https://archive-api.open-meteo.com/v1/archive'

    selected_params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': [
            'temperature_2m',
            'wind_speed_100m',
            'wind_gusts_10m',
            'cloud_cover',
            'shortwave_radiation',
            'direct_radiation',
            'diffuse_radiation',
            'pressure_msl',
            'snowfall'
        ]
    }
    response = requests.get(url, params=selected_params).json()
    return pd.DataFrame(response["hourly"])


def weather_preproc(df):
    ''' preprocess weather dataframe, resample, rename, check quality'''
    # datetime and set index
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # sample half hourly
    df = df.resample('30min').ffill()

    # reset index and rename to datetime
    df = df.reset_index()
    df = df.rename(columns={'time': 'datetime'})

    # rename columns w/ units
    df = df.rename(columns={
        'temperature_2m': 'temperature_2m_c',
        'wind_speed_100m': 'wind_speed_100m_ms',
        'wind_gusts_10m': 'wind_gusts_10m_ms',
        'cloud_cover': 'cloud_cover_pct',
        'shortwave_radiation': 'shortwave_radiation_wm2',
        'direct_radiation': 'direct_radiation_wm2',
        'diffuse_radiation': 'diffuse_radiation_wm2',
        'pressure_msl': 'pressure_msl_hpa',
        'snowfall': 'snowfall_cm'
    })

    return df
