import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from python_scripts.data_to_bigquery import load_from_bigquery
from python_scripts.hgb_pipeline import hgb_train_preproc


def evaluate_model(y_true, preds):
    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds) ** 0.5
    r2 = r2_score(y_true, preds)
    return mae, rmse, r2


def compare_weather_feature_sets(
    project_id: str,
    dataset_id: str,
    table_id: str,
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime",
    test_size: float = 0.2,
    random_state: int = 42,
):
    df = load_from_bigquery(project_id, dataset_id, table_id)

    df = hgb_train_preproc(
        df,
        target_col=target_col,
        datetime_col=datetime_col
    )

    weather_features = [
        "temperature_2m_c",
        "wind_speed_100m_ms",
        "wind_gusts_10m_ms",
        "cloud_cover_pct",
        "shortwave_radiation_wm2",
        "direct_radiation_wm2",
        "diffuse_radiation_wm2",
        "pressure_msl_hpa",
        "precipitation_mm"
    ]

    generation_features = [
        "biomass",
        "fossil_gas",
        "fossil_hard_coal",
        "hydro_pumped_storage",
        "hydro_run_of_river_and_poundage",
        "nuclear",
        "other",
        "solar",
        "wind_offshore",
        "wind_onshore",
        "totaloutput_mw"
    ]

    time_features = [
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "doy_sin",
        "doy_cos"
    ]

    lag_rolling_features = [
        "carbon_lag_48",
        "carbon_lag_336",
        "carbon_roll_24h",
        "carbon_roll_168h"
    ]

    full_features = weather_features + generation_features + time_features + lag_rolling_features
    no_weather_features = generation_features + time_features + lag_rolling_features
    weather_only_features = weather_features

    y = df[target_col].copy()

    X_sets = {
        "Full Model": df[full_features].copy(),
        "No-Weather Model": df[no_weather_features].copy(),
        "Weather-Only Model": df[weather_only_features].copy()
    }

    split_index = int(len(df) * (1 - test_size))
    results = []
    preds_dict = {}

    for name, X in X_sets.items():
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae, rmse, r2 = evaluate_model(y_test, preds)

        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        preds_dict[name] = preds

    comparison_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    return comparison_df, preds_dict, y_test
