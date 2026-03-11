from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from data_to_bigquery import load_from_bigquery
from feature_engineering import engineer_features, drop_lag_nulls

# load merged data set
df = load_from_bigquery('gridzero-489711', 'merged_set', 'Fully_merged_dataset_2025')

# feature engineering of the df
df = engineer_features(df, target_col='carbon_intensity')
validate_features(df)
df = drop_lag_nulls(df)

# features
feature_cols = [
    'temperature_2m_c', 'wind_speed_100m_ms', 'wind_gusts_10m_ms',
    'cloud_cover_pct', 'shortwave_radiation_wm2', 'direct_radiation_wm2',
    'diffuse_radiation_wm2', 'pressure_msl_hpa', 'snowfall_cm',
    'Fossil Gas', 'Nuclear', 'Wind Offshore', 'Wind Onshore',
    'Solar', 'Biomass', 'TotalOutput-MW',
    'lag_48', 'lag_336', 'hour', 'month', 'is_weekend'
]

target_col = 'carbon_intensity'

X = df[feature_cols]
y = df[target_col]

# ── TEMPORAL SPLIT ───────────────────────────────────────────────────
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ── TRAIN ────────────────────────────────────────────────────────────
model = make_pipeline(
    XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
)
model.fit(X_train, y_train)

# ── EVALUATE ─────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} gCO2/kWh")
print(f"R²:  {r2_score(y_test, y_pred):.4f}")

# ── PLOT ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
plt.figure(figsize=(15,4))
plt.plot(y_test.values[:200], label='Actual', alpha=0.7)
plt.plot(y_pred[:200], label='Predicted', alpha=0.7)
plt.title('Carbon Intensity — Actual vs Predicted')
plt.ylabel('gCO2/kWh')
plt.legend()
plt.show()
