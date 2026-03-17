import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    max_error
)

from python_scripts.data_to_bigquery import load_from_bigquery
from python_scripts.hgb_pipeline import hgb_train_preproc


def lgbm_model_train(
    project_id: str,
    dataset_id: str,
    table_id: str,
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train LightGBM model using the same preprocessing pipeline as HGB.
    """

    df = load_from_bigquery(project_id, dataset_id, table_id)

    assert target_col in df.columns, "Target column missing"
    assert datetime_col in df.columns, "Datetime column missing"

    df = hgb_train_preproc(df, target_col=target_col, datetime_col=datetime_col)

    X = df.drop(columns=[target_col, datetime_col])
    y = df[target_col]

    split_index = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

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

    print("Training LightGBM...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": root_mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "Max Error": max_error(y_test, preds),
    }

    print("\nLightGBM Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    feature_importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("\nTop 10 Features:")
    print(feature_importance.head(10))

    return model, metrics, preds, X_test, y_test


def lgbm_prediction(model, input_df: pd.DataFrame):
    """
    Make predictions using trained LightGBM model.
    """
    return model.predict(input_df)
