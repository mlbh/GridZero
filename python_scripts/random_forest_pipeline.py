import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    max_error
)

# project imports
from python_scripts.data_to_bigquery import load_from_bigquery
from python_scripts.hgb_pipeline import hgb_train_preproc  # reuse preprocessing


def rf_model_train(
    project_id: str,
    dataset_id: str,
    table_id: str,
    target_col: str = "carbon_intensity_gco2_kwh",
    datetime_col: str = "datetime",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train Random Forest model using same pipeline as HGB.
    """

    # Load data
    df = load_from_bigquery(project_id, dataset_id, table_id)

    # Preprocess (reuse your existing logic )
    df = hgb_train_preproc(df, target_col=target_col, datetime_col=datetime_col)

    # Split features & target
    X = df.drop(columns=[target_col, datetime_col])
    y = df[target_col]

    # Time-based split (important for time series)
    split_index = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Random Forest Model (customizable)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": root_mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "Max Error": max_error(y_test, preds),
    }

    print("\nRandom Forest Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return model, metrics, preds, X_test, y_test


def rf_prediction(model, input_df: pd.DataFrame):
    """
    Make predictions using trained RF model
    """
    return model.predict(input_df)
