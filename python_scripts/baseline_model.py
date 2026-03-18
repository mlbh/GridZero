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
    target_col: str = 'carbon_intensity_gco2_kwh',
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
                        target_col='carbon_intensity_gco2_kwh',
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

# xgboost with hyper params:
def baseline_model_xgb_1(
                        PROJECT: str='gridzero-489711',
                        DATASET: str='merged_set',
                        TABLE: str='test_merge_2017_onward_raw',
                        target_col='carbon_intensity_gco2_kwh',
                        test_size:int = 0.3
                        ) -> tuple[XGBRegressor,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   ]:

    # load df from BQ
    df = load_from_bigquery(PROJECT=PROJECT, DATASET=DATASET, TABLE=TABLE)
    # sort by datetime and reset index ooooo
    df = df.sort_values('datetime').reset_index(drop=True)
    target_col = 'carbon_intensity_gco2_kwh'
    # temporal split
    #option 1 by year
    # train_df = df[df['datetime'].dt.year == 2025]
    # test_df  = df[df['datetime'].dt.year >= 2025]
    # option 2
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=[target_col, 'datetime'])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, 'datetime'])
    y_test = test_df[target_col]

    # keep only num col to make xgboost happy
    feature_cols = X_train.select_dtypes(include='number').columns.tolist()

    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    # build simple xgb bood model
    model = XGBRegressor(random_state=42)

    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# xgboost with hyper params:
def baseline_model_xgb_2(
                        PROJECT: str='gridzero-489711',
                        DATASET: str='merged_set',
                        TABLE: str='test_merge_2017_onward_raw',
                        target_col='carbon_intensity_gco2_kwh',
                        test_size:int = 0.3
                        ) -> tuple[XGBRegressor,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   ]:

    # load df from BQ
    df = load_from_bigquery(PROJECT=PROJECT, DATASET=DATASET, TABLE=TABLE)
    # sort by datetime and reset index ooooo
    df = df.sort_values('datetime').reset_index(drop=True)
    target_col = 'carbon_intensity_gco2_kwh'
    # temporal split
    #option 1 by year
    # train_df = df[df['datetime'].dt.year == 2025]
    # test_df  = df[df['datetime'].dt.year >= 2025]
    # option 2
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=[target_col, 'datetime'])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, 'datetime'])
    y_test = test_df[target_col]

    # keep only num col to make xgboost happy
    feature_cols = X_train.select_dtypes(include='number').columns.tolist()

    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    # build simple xgb bood model
    model = XGBRegressor(
            random_state=42,
            n_estimators=1450,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=6,
            gamma=0,
            subsample=1,
            colsample_bytree=1,
            reg_alpha=0,
            reg_lambda=2
            )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test



# trained model evaluation fucntion
def evaluate_trained_model(model, X_train, X_test, y_train, y_test) -> dict:
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    train_max_err = max_error(y_train, y_train_pred)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    test_max_err = max_error(y_test, y_test_pred)


    model_eval= pd.DataFrame({
    "dataset": ["train", "test"],
    "mae": [train_mae, test_mae],
    "rmse": [train_rmse, test_rmse],
    "r2": [train_r2, test_r2],
    "max_error": [train_max_err, test_max_err]
})
    return model_eval



# preping prediction features to be used with frontend
# Not needed atm but might be needed if new stream of data added
def pred_preproc_xgb(df_new: pd.DataFrame,
                    feature_cols: list,
                    target_col: str='carbon_intensity_gco2_kwh'
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
            target_col: str='carbon_intensity_gco2_kwh'
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

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor



# grdisearch model function
def gridsearch_model_xgb(
                        param_grid: dict,
                        PROJECT: str='gridzero-489711',
                        DATASET: str='merged_set',
                        TABLE: str='test_merge_2017_onward_raw',
                        target_col='carbon_intensity_gco2_kwh',
                        test_size:int = 0.3,
                        refit_metric: str='mae',
                        n_splits: int=5
                        ):

    # load df from BQ
    df = load_from_bigquery(PROJECT=PROJECT, DATASET=DATASET, TABLE=TABLE)

    # sort by datetime
    df = df.sort_values('datetime').reset_index(drop=True)

    # temporal split
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=[target_col, 'datetime'])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, 'datetime'])
    y_test = test_df[target_col]

    # numeric only
    feature_cols = X_train.select_dtypes(include='number').columns.tolist()

    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    # scoring
    scoring = {
        'mae': 'neg_mean_absolute_error',
        'rmse': 'neg_root_mean_squared_error',
        'r2': 'r2',
        'max_err': 'neg_max_error'
    }

    # time series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # base model (NO params → grid will set)
    model = XGBRegressor(
            random_state=42,
            n_jobs=-1
            )

    grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit_metric,
            cv=tscv,
            verbose=2,
            n_jobs=-1,
            return_train_score=False
            )

    grid.fit(X_train, y_train)

    return grid, X_train, X_test, y_train, y_test


def baseline_model_xgb_3(
                        PROJECT: str='gridzero-489711',
                        DATASET: str='merged_set',
                        TABLE: str='test_merge_2017_onward_raw',
                        target_col='carbon_intensity_gco2_kwh',
                        test_size:int = 0.3
                        ) -> tuple[XGBRegressor,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   pd.DataFrame,
                                   ]:

    # load df from BQ
    df = load_from_bigquery(PROJECT=PROJECT, DATASET=DATASET, TABLE=TABLE)
    # sort by datetime and reset index ooooo
    df = df.sort_values('datetime').reset_index(drop=True)
    target_col = 'carbon_intensity_gco2_kwh'
    # temporal split
    #option 1 by year
    # train_df = df[df['datetime'].dt.year == 2025]
    # test_df  = df[df['datetime'].dt.year >= 2025]
    # option 2
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx].copy()
    test_df  = df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=[target_col, 'datetime'])
    y_train = train_df[target_col]

    X_test = test_df.drop(columns=[target_col, 'datetime'])
    y_test = test_df[target_col]

    # keep only num col to make xgboost happy
    feature_cols = X_train.select_dtypes(include='number').columns.tolist()

    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    # build simple xgb bood model
    model = XGBRegressor(
            random_state=42,
            n_estimators=1450,
            learning_rate=0.04,
            max_depth=6,
            min_child_weight=6,
            gamma=0,
            subsample=1,
            colsample_bytree=1,
            reg_alpha=0,
            reg_lambda=2
            )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test
