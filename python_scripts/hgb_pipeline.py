import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    max_error
)

# project imports
from python_scripts.data_to_bigquery import load_from_bigquery

def hgb_train_preproc(
    df: pd.DataFrame,
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime",
    drop_year_lag_na: bool = True,
    add_rolling_features: bool = True
) -> pd.DataFrame:
    """
    Prepare dataframe for HGB training.

    Steps:
    - ensure datetime column is present and parsed
    - sort chronologically
    - drop rows with missing lag features
    - create leakage-safe rolling features using shift(1)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from BigQuery
    target_col : str
        Name of target column
    datetime_col : str
        Name of datetime column
    drop_year_lag_na : bool
        If True, drop rows where carbon_lag_17520 is null
    add_rolling_features : bool
        If True, add rolling mean features

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for train/test split
    """
    df = df.copy()

    if "time" in df.columns and datetime_col not in df.columns:
        df = df.rename(columns={"time": datetime_col})

    if datetime_col not in df.columns:
        raise ValueError(f"Missing datetime column: {datetime_col}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
    df = df.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)

    lag_cols = ["carbon_lag_48", "carbon_lag_336"]
    if drop_year_lag_na and "carbon_lag_17520" in df.columns:
        lag_cols.append("carbon_lag_17520")

    lag_cols = [col for col in lag_cols if col in df.columns]

    if lag_cols:
        df = df.dropna(subset=lag_cols).reset_index(drop=True)

    if add_rolling_features:
        df["carbon_roll_24h"] = df[target_col].shift(1).rolling(24).mean()
        df["carbon_roll_168h"] = df[target_col].shift(1).rolling(168).mean()
        df = df.dropna(subset=["carbon_roll_24h", "carbon_roll_168h"]).reset_index(drop=True)

    return df


def get_hgb_feature_cols(
    df: pd.DataFrame,
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime"
) -> list:
    """
    Return numeric feature columns excluding target and datetime.
    """
    X = df.drop(columns=[target_col, datetime_col], errors="ignore").copy()
    X = X.select_dtypes(include="number")
    return X.columns.tolist()


def temporal_split(
    df: pd.DataFrame,
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime",
    test_size: float = 0.2,
    feature_cols: list | None = None
):
    """
    Chronological train/test split.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    df = df.sort_values(datetime_col).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = get_hgb_feature_cols(df, target_col=target_col, datetime_col=datetime_col)

    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df[target_col]

    X_test = test_df[feature_cols].astype(float)
    y_test = test_df[target_col]

    return train_df, test_df, X_train, X_test, y_train, y_test, feature_cols


def hgb_model_train(
    PROJECT: str = "gridzero-489711",
    DATASET: str = "merged_set",
    TABLE: str = "full_feature_engineered_data_test",
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime",
    test_size: float = 0.2,
    drop_year_lag_na: bool = True
):
    """
    Full reusable HGB training pipeline from BigQuery.

    Returns
    -------
    tuple
        model, train_df, test_df, X_train, X_test, y_train, y_test, feature_cols
    """
    df = load_from_bigquery(PROJECT=PROJECT, DATASET=DATASET, TABLE=TABLE)

    df = hgb_train_preproc(
        df=df,
        target_col=target_col,
        datetime_col=datetime_col,
        drop_year_lag_na=drop_year_lag_na,
        add_rolling_features=True
    )

    train_df, test_df, X_train, X_test, y_train, y_test, feature_cols = temporal_split(
        df=df,
        target_col=target_col,
        datetime_col=datetime_col,
        test_size=test_size
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=400,
        max_depth=8,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, train_df, test_df, X_train, X_test, y_train, y_test, feature_cols


def evaluate_trained_model(model, X_test, y_test) -> dict:
    """
    Evaluate a trained HGB model.
    """
    y_pred = model.predict(X_test)

    return {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
        "MaxError": max_error(y_test, y_pred)
    }


def pred_preproc_hgb(
    df_new: pd.DataFrame,
    feature_cols: list,
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime"
) -> pd.DataFrame:
    """
    Prepare new data for inference by matching training feature columns.
    """
    X_new = df_new.drop(columns=[target_col, datetime_col], errors="ignore").copy()
    X_new = X_new.select_dtypes(include="number")
    X_new = X_new.astype(float)
    X_new = X_new.reindex(columns=feature_cols)

    return X_new


def hgb_prediction(
    model,
    df_new: pd.DataFrame,
    feature_cols: list,
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime"
):
    """
    Generate predictions on new data using trained HGB model.
    """
    X_new = pred_preproc_hgb(
        df_new=df_new,
        feature_cols=feature_cols,
        target_col=target_col,
        datetime_col=datetime_col
    )

    y_pred = model.predict(X_new)
    return y_pred
