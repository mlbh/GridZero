import numpy as np
import pandas as pd


# ---------------------------
# 1. Create baseline input
# ---------------------------
def create_default_simulation_input(feature_cols: list) -> pd.DataFrame:
    """
    Create a default baseline row for simulation.
    """
    default_values = {
        "temperature_2m_c": 15.0,
        "wind_speed_100m_ms": 6.0,
        "wind_gusts_10m_ms": 8.0,
        "cloud_cover_pct": 50.0,
        "shortwave_radiation_wm2": 200.0,
        "direct_radiation_wm2": 120.0,
        "diffuse_radiation_wm2": 80.0,
        "pressure_msl_hpa": 1013.0,
        "precipitation_mm": 0.0,
        "biomass": 1200.0,
        "fossil_gas": 8000.0,
        "fossil_hard_coal": 1500.0,
        "hydro_pumped_storage": 500.0,
        "hydro_run_of_river_and_poundage": 700.0,
        "nuclear": 6000.0,
        "other": 300.0,
        "solar": 2500.0,
        "wind_offshore": 4000.0,
        "wind_onshore": 3500.0,
        "totaloutput_mw": 30000.0,
        "hour_sin": 0.0,
        "hour_cos": 1.0,
        "dow_sin": 0.0,
        "dow_cos": 1.0,
        "doy_sin": 0.0,
        "doy_cos": 1.0,
        "carbon_lag_48": 180.0,
        "carbon_lag_336": 190.0,
        "carbon_lag_17520": 200.0,
        "carbon_roll_24h": 185.0,
        "carbon_roll_168h": 188.0,
    }

    row = {col: default_values.get(col, 0.0) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


# ---------------------------
# 2. Prepare input
# ---------------------------
def prepare_simulation_input(df_input: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Ensure correct feature order and type.
    """
    X = df_input.copy()
    X = X.reindex(columns=feature_cols)
    X = X.astype(float)
    return X


# ---------------------------
# 3. Predict
# ---------------------------
def predict_carbon_intensity(model, df_input: pd.DataFrame, feature_cols: list) -> float:
    """
    Predict carbon intensity.
    """
    X = prepare_simulation_input(df_input, feature_cols)
    pred = model.predict(X)[0]
    return float(pred)


# ---------------------------
# 4. Apply scenario changes
# ---------------------------
def apply_simulation_changes(base_df: pd.DataFrame, changes: dict) -> pd.DataFrame:
    """
    Apply what-if changes.
    """
    sim_df = base_df.copy()

    for col, value in changes.items():
        if col not in sim_df.columns:
            raise ValueError(f"Column '{col}' not found in simulation input.")
        sim_df.at[0, col] = value

    return sim_df


# ---------------------------
# 5. Compare results
# ---------------------------
def compare_simulation(model, base_df: pd.DataFrame, sim_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Compare baseline vs simulated prediction.
    """
    baseline_pred = predict_carbon_intensity(model, base_df, feature_cols)
    simulated_pred = predict_carbon_intensity(model, sim_df, feature_cols)

    diff = simulated_pred - baseline_pred
    pct_change = (diff / baseline_pred * 100) if baseline_pred != 0 else np.nan

    return pd.DataFrame({
        "baseline_prediction": [baseline_pred],
        "simulated_prediction": [simulated_pred],
        "absolute_change": [diff],
        "percent_change": [pct_change]
    })


# ---------------------------
# 6. Full pipeline (MAIN FUNCTION)
# ---------------------------
def run_simulation(model, feature_cols: list, changes: dict, baseline_df: pd.DataFrame | None = None):
    """
    Full simulation pipeline.

    Returns:
        baseline_df, scenario_df, comparison_df
    """
    if baseline_df is None:
        baseline_df = create_default_simulation_input(feature_cols)

    scenario_df = apply_simulation_changes(baseline_df, changes)
    comparison_df = compare_simulation(model, baseline_df, scenario_df, feature_cols)

    return baseline_df, scenario_df, comparison_df
