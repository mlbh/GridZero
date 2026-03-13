import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_error, max_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

# python script imports
from data_to_bigquery import load_from_bigquery
from feature_engineering import engineer_features, drop_lag_nulls, validate_features

# model comparison code
models = {
    'dummy': make_pipeline(
        DummyRegressor(strategy='mean')
    ),
    'linear_regression': make_pipeline(
        StandardScaler(),
        LinearRegression()
    ),
    'random_forest': make_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    ),
    'xgboost_default': make_pipeline(
            XGBRegressor()
    ),

    'xgboost': make_pipeline(
        XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            objective='reg:squarederror'
        )
    ),
    'xgboost_scl': make_pipeline(
        StandardScaler(),
        XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
    ),
    'xgboost_opt' : make_pipeline(
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            #L1
            reg_alpha=0.1,
            #L2
            reg_lambda=0,
            # histo method (faster than default greedy algo)
            tree_method='hist',
            random_state=42
        )
    ),

    'xgboost_opt_scl' : make_pipeline(
        StandardScaler(),
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=42
        )

    )
}

# scoring
scoring = {
    'mae': 'neg_mean_absolute_error',
    'rmse': 'neg_root_mean_squared_error',
    'r2': 'r2',
    'max_err':'neg_max_error'
}

# loop through models
# results
results = []

for model_name, pipeline in models.items():
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )

# eval metrics selected and named
    model_results = {
        'model': model_name,
        'model_fit_t': cv_results['fit_time'].mean(),
        'mae_mean': -cv_results['test_mae'].mean(),
        'rmse_mean': -cv_results['test_rmse'].mean(),
        'r2_mean': cv_results['test_r2'].mean(),
        'max_err_max': -cv_results['test_max_err'].max(),
    }
    results.append(model_results)

# comparison table with names
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('mae_mean')
display(results_df)
