from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# python script imports
from data_to_bigquery import load_from_bigquery
from feature_engineering import engineer_features, drop_lag_nulls

# load merged data set
df = load_from_bigquery('gridzero-489711', 'merged_set', 'Fully_merged_dataset_2025')

# feature engineering of the df
# note that there is a bug if 'target_col='carbon_instensity' arg is explicity included in the function call
# target_col = 'carbon_intensity' is set to default
df = engineer_features(df, target_col='carbon_intensity')
validate_features(df)
df = drop_lag_nulls(df)

# features
feature_cols = [
    'temperature_2m_c', 'wind_speed_100m_ms', 'wind_gusts_10m_ms',
    'cloud_cover_pct', 'shortwave_radiation_wm2', 'direct_radiation_wm2',
    'diffuse_radiation_wm2', 'pressure_msl_hpa', 'snowfall_cm', 'rain_mm', 'precipitation_mm',
    'Fossil Gas', 'Nuclear', 'Wind Offshore', 'Wind Onshore',
    'Solar', 'Biomass', 'TotalOutput-MW',
    'lag_48', 'lag_336', 'hour', 'month', 'is_weekend'
]

target_col = 'carbon_intensity'

X = df[feature_cols]
y = df[target_col]

# split for single year (baseline)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#

# split for full dataset (ensure minimum of 2 years for test)
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# check split ouputs
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# train

# evaluate

# plot
