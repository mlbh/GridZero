import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_error, max_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# python script imports
from data_to_bigquery import load_from_bigquery
from feature_engineering import engineer_features, drop_lag_nulls, validate_features


# baseline_preproc fucntion that runs if features need to be engineered
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

    if add_year_lag:
        required_features.append('lag_17520')

    # only engineer features if ANY required features missing
    if not all(col in df.columns for col in required_features):
        df = engineer_features(df, target_col=target_col, add_year_lag=add_year_lag)

    # # drop rows where target missing
    # df = df.dropna(subset=[target_col])

    # # drop lag null rows
    # lag_cols = [col for col in ['lag_48', 'lag_336', 'lag_17520'] if col in df.columns]
    # if lag_cols:
    #     df = df.dropna(subset=lag_cols)

    # simple drop null to make sure can parse saucey tables and raw tables
    df = df.dropna()

    return df

# Function to predict with baseline model simple/non-temporal split
# imports load_from bigquery fucntion to load df
def baseline_model_xgb(
                        PROJECT: str='gridzero-489711',
                        DATASET: str='merged_set',
                        TABLE: str='test_merge_2017_onward_raw',
                        target_col='carbon_intensity_gCO2_kWh',
                        test_size:int = 0.3
                        ) -> XGBRegressor:

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

    ''' Debugging line, uncomment if errores relating to dtypes'''
    # print("NON NUMERIC:", X.select_dtypes(exclude='number').columns.tolist())

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

    return model, X.columns.tolist()

# preping prediction features to be used with frontend
# Not needed atm but might be needed if new stream of data added
def prepare_prediction_features(
                        df_new: pd.DataFrame,
                        feature_cols: list,
                        target_col: str='carbon_intensity_gCO2_kWh'
                    ) -> pd.DataFrame:

    # rename if col 'time' is present
    df_new = df_new.rename(columns={'time': 'datetime'}).copy()
    df_new = baseline_preproc(df_new, target_col=target_col)

    X_new = df_new.drop(columns=[target_col, 'datetime'], errors='ignore').copy()
    X_new = X_new.select_dtypes(include='number')
    X_new = X_new.astype(float)

    # keep only training columns
    X_new = X_new.reindex(columns=feature_cols, fill_value=0)

    return X_new


def make_prediction(
            model,
            df_new: pd.DataFrame,
            feature_cols: list,
            target_col: str='carbon_intensity_gCO2_kWh'
        ):

# calls fancy frontend function
    X_new = prepare_prediction_features(
    df_new=df_new,
    feature_cols=feature_cols,
    target_col=target_col
)

    y_pred = model.predict(X_new)

    return y_pred


#the idea was to end up with something like this...
def make_prediction(X_new):
    baseline_model = baseline_model_xgb()

    y_pred = baseline_model.predict(X_new)
    return y_pred

def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    return {
        'model': model_name,
        'model_fit_t': cv_results['fit_time'].mean(),
        'mae_mean': -cv_results['test_mae'].mean(),
        'rmse_mean': -cv_results['test_rmse'].mean(),
        'r2_mean': cv_results['test_r2'].mean(),
        'max_err_max': -cv_results['test_max_err'].max(),
    }



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
