import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# python script imports
from data_to_bigquery import load_from_bigquery
from feature_engineering import engineer_features, drop_lag_nulls, validate_features

# preproc that runs if features need to be engineered
def baseline_preproc(df, target_col='carbon_intensity', add_year_lag=False):
    '''
    Apply feature engineering only if engineered columns are missing.
    Drop lag-created nulls if lag columns are present.
    '''

    df = df.copy()

    # rename time column if needed
    if 'time' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'time': 'datetime'})

    required_features = [
        'lag_48',
        'lag_336',
        'hour',
        'day_of_week',
        'month',
        'is_weekend'
    ]

    if add_year_lag:
        required_features.append('lag_17520')

    # only engineer features if they are missing
    missing_features = [col for col in required_features if col not in df.columns]

    if missing_features:
        df = engineer_features(df, target_col=target_col, add_year_lag=add_year_lag)

    # drop nulls only if lag columns exist
    lag_cols = [col for col in ['lag_48', 'lag_336', 'lag_17520'] if col in df.columns]

    if lag_cols:
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

    return df


    # Function to predict with baseline model simple split
def baseline_model_xgb(PROJECT: str='gridzero-489711',
                    DATASET: str='merged_set',
                    TABLE: str='Fully_merged_dataset_2025',
                    test_size:int = 0.3
                    ):

    # pull df from BQ
    df = load_from_bigquery(PROJECT=PROJECT, DATASET=DATASET, TABLE=TABLE)

    # rename if col 'time' is present
    df = df.rename(columns={'time' : 'datetime'})

    # # engineer the features
    # df = engineer_features(df, target_col='carbon_intensity')
    # df = baseline_preproc(df)

    # #print checks
    # validate_features(df)
    # df = drop_lag_nulls(df)

    # preproc if applicable
    df = baseline_preproc(df)

    # set features and target variables
    X = df.drop(columns=['carbon_intensity', 'datetime'])
    y = df['carbon_intensity']

    # test train split for single year
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # build simple xgb bood model
    pipeline = make_pipeline(
                    StandardScaler(),
                    XGBRegressor()
                 )

    # training model
    pipeline.fit(X_train, y_train)

    return pipeline


#the idea was to end up with something like this...
def make_prediction(X_new):
    baseline_model = baseline_model_xgb()

    y_pred = baseline_model.predict(X_new)
    return y_pred



# daniels feature engineering function below, updating the correct col
def engineer_features(df, target_col="carbon_intensity_gCO2_kWh", add_year_lag=False):
    """
    Create lag and calendar features for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a datetime column and target column.

    target_col : str, default="carbon_intensity"
        Name of the target column to create lag features from.

    add_year_lag : bool, default=False
        Whether to add a 1-year lag feature.
        Only use this if the dataframe contains at least 2 years of half-hourly data.

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered lag and calendar features.
    """

    df = df.copy()

    # Ensure datetime is in datetime format
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Drop rows with invalid datetime values
    df = df.dropna(subset=["datetime"])

    # Sort by datetime
    df = df.sort_values("datetime").reset_index(drop=True)

    # Lag features
    # 48 half-hour periods = 24 hours
    df["lag_48"] = df[target_col].shift(48)

    # 336 half-hour periods = 7 days
    df["lag_336"] = df[target_col].shift(336)

    # Optional yearly lag
    # 17520 half-hour periods = 365 days
    if add_year_lag:
        df["lag_336"] = df[target_col].shift(17520)

    # Calendar features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    return df
