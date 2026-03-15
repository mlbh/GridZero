import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_error, max_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# python script imports
from data_to_bigquery import load_from_bigquery
from feature_engineering import engineer_features, drop_lag_nulls, validate_features


# xgb training preproc fucntion that runs if features need to be engineered
# imports Daniels feature_engineering to engineer cols if not present
# py scripts listed in from/imports
# drop datetime later in model step, not here
def xgb_train_preproc(
    df: pd.DataFrame,
    target_col: str = 'carbon_intensity_gCO2_kWh',
    add_year_lag: bool =False
    ) -> pd.DataFrame:
    '''
    Apply feature engineering ONLY if engineered columns are missing.
    Otherwise return df unchanged (except optional 'time' to 'datetime' rename).
    '''
    df = df.copy()
    # ensure datetime col name if needed
    if 'time' in df.columns and 'datetime' not in df.columns:
        df = df.rename(columns={'time': 'datetime'})
     # engineer features only if missing
    required_features = [
        'lag_48',
        'lag_336',
        'hour',
        'day_of_week',
        'month',
        'is_weekend'
    ]

    # if year = TRUE
    if add_year_lag:
        required_features.append('lag_17520')
    # embedded function fuzzy match
    # suffix matcher if features renamed
    def feature_exists(df_cols, feature):
        return any(col.endswith(feature) for col in df_cols)
    # only engineer features if ANY required features missing
    if not all(feature_exists(df.columns, col) for col in required_features):
        df = engineer_features(df, target_col=target_col, add_year_lag=add_year_lag)
    # simple drop null to make sure can parse saucey tables and raw tables
    df = df.dropna()
    return df


# Function to predict with baseline model simple/non-temporal split
# imports load_from bigquery fucntion to load df
# outputs: model, X_train, X_test, y_train, y_test = baseline_model_
def baseline_model_xgb(
                        PROJECT: str='gridzero-489711',
                        DATASET: str='merged_set',
                        TABLE: str='test_merge_2017_onward_raw',
                        target_col='carbon_intensity_gCO2_kWh',
                        test_size:int = 0.3
                        ) -> tuple[XGBRegressor,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   ]:

    # load df from BQ
    df = load_from_bigquery(PROJECT=PROJECT, DATASET=DATASET, TABLE=TABLE)

    # rename if col 'time' is present
    df = df.rename(columns={'time' : 'datetime'})

    # preproc if applicable
    df = xgb_train_preproc(df, target_col=target_col)

    # drop lag null rows
    lag_cols = [col for col in ['lag_48', 'lag_336', 'lag_17520'] if col in df.columns]
    if lag_cols:
        df = df.dropna(subset=lag_cols)

    # define features and target
    X = df.drop(columns=[target_col, 'datetime'], errors='ignore').copy()
    # drop all non-numeric cols
    X = X.select_dtypes(include='number')
    # converting from Int64 (if applicable)so can also use the sauncey merged data tables
    X = X.astype(float)
    y = df[target_col]

    # test train split for single year
    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                        y,
                                        test_size=test_size,
                                        random_state=42
                                        )

    # build simple xgb bood model
    model = XGBRegressor(random_state=42)

    # training model
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

# follow with this
# feature_cols = X_train.columns.tolist()

# trained model evaluation fucntion
def evaluate_trained_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    return {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred),
        'MaxError': max_error(y_test, y_pred)
    }

# preping prediction features to be used with frontend
# Not needed atm but might be needed if new stream of data added
def pred_preproc_xgb(df_new: pd.DataFrame,
                    feature_cols: list,
                    target_col: str='carbon_intensity_gCO2_kWh'
                ) -> pd.DataFrame:

    X_new = df_new.drop(columns=[target_col, 'datetime'], errors='ignore').copy()
    X_new = X_new.select_dtypes(include='number')
    X_new = X_new.astype(float)
    X_new = X_new.reindex(columns=feature_cols)

    return X_new

# predictio function
def xgb_prediction(
            model,
            df_new: pd.DataFrame,
            feature_cols: list,
            target_col: str='carbon_intensity_gCO2_kWh'
            ):
# calls fancy frontend function
    X_new = pred_preproc_xgb(
            df_new=df_new,
            feature_cols=feature_cols,
            target_col=target_col
        )
    # predicting
    y_pred = model.predict(X_new)
    return y_pred



# BBoujie af LLM CHECK to help debugging
# tables = ['test_merge_2017_onward_raw', 'test_merge_with_the_sauce']

# for table in tables:
#     print(f"\n--- TESTING PREPROC: {table} ---")

#     df = load_from_bigquery(
#         PROJECT='gridzero-489711',
#         DATASET='merged_set',
#         TABLE=table
#     )

#     df_pre = baseline_preproc(df)

#     print("raw shape:", df.shape)
#     print("preproc shape:", df_pre.shape)
#     print("columns after preproc:", df_pre.columns.tolist())

#     target_candidates = [c for c in df_pre.columns if 'carbon_intensity' in c]
#     print("target candidates:", target_candidates)

#     if target_candidates:
#         target_col = target_candidates[0]
#         print("target NA count:", df_pre[target_col].isna().sum())

#     lag_cols = [c for c in ['lag_48', 'lag_336', 'lag_17520'] if c in df_pre.columns]
#     print("lag cols present:", lag_cols)

#     if lag_cols:
#         print("lag NA counts:")
#         print(df_pre[lag_cols].isna().sum())

#     print("total NA count in preproc df:", df_pre.isna().sum().sum())
