import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_sequences(X, y, lookback):

    Xs, ys = [], []

    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])

    return np.array(Xs), np.array(ys)


def lstm_train(df):
    feature_cols = [
        # weather
        'temperature_2m_c',
        'wind_speed_100m_ms',
        'wind_gusts_10m_ms',
        'cloud_cover_pct',
        'shortwave_radiation_wm2',
        'direct_radiation_wm2',
        'diffuse_radiation_wm2',
        'pressure_msl_hpa',
        'precipitation_mm',

        # time
        'hour_sin','hour_cos',
        'dow_sin','dow_cos',
        'doy_sin','doy_cos',

        # past generation (important)
        'biomass',
        'fossil_gas',
        'fossil_hard_coal',
        'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage',
        'nuclear',
        'other',
        'solar',
        'wind_offshore',
        'wind_onshore'
    ]

    target_cols = [
        'biomass',
        'fossil_gas',
        'fossil_hard_coal',
        'hydro_pumped_storage',
        'hydro_run_of_river_and_poundage',
        'nuclear',
        'other',
        'solar',
        'wind_offshore',
        'wind_onshore'
    ]

    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X = X_scaler.fit_transform(df[feature_cols])
    y = y_scaler.fit_transform(df[target_cols])




    lookback = 336

    # def create_sequences(X, y, lookback):

    #     Xs, ys = [], []

    #     for i in range(len(X) - lookback):
    #         Xs.append(X[i:i+lookback])
    #         ys.append(y[i+lookback])

    #     return np.array(Xs), np.array(ys)

    X_seq, y_seq = create_sequences(X, y, lookback)


    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]

    X_val = X_seq[train_size:train_size+val_size]
    y_val = y_seq[train_size:train_size+val_size]

    X_test = X_seq[train_size+val_size:]
    y_test = y_seq[train_size+val_size:]



    model = Sequential()

    model.add(LSTM(128, return_sequences=True,
                input_shape=(lookback, len(feature_cols))))

    model.add(Dropout(0.2))

    model.add(LSTM(64))

    model.add(Dense(64, activation="relu"))

    model.add(Dense(len(target_cols)))
