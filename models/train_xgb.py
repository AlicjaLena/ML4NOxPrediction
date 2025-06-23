import pandas as pd
import numpy as np
import joblib
import json
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from features.preprocess import load_and_split_data
from features.feature_engineering import add_engineered_features, select_features
from utils.metrics import evaluate_model


def train_model(X_train, y_train):
    """
    Trains an XGBoost model using GridSearchCV in a pipeline.
    Returns the best model and the pipeline.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBRegressor(objective='reg:squarederror', random_state=42))
    ])

    param_grid = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 6, 10],
        'xgb__learning_rate': [0.01, 0.1, 0.2],
        'xgb__subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


def main():
    # Load and preprocess
    train_df, val_df, _ = load_and_split_data()
    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)

    X_train = select_features(train_df)
    y_train = train_df["NOX"]

    X_val = select_features(val_df)
    y_val = val_df["NOX"]

    # Train and evaluate
    best_model, best_params = train_model(X_train, y_train)
    scaler = best_model.named_steps['scaler']

    mae, rmse, r2 = evaluate_model(best_model, scaler, X_val, y_val)

    # Save model and scaler
    joblib.dump(best_model, "./artifacts/best_model_xgb.joblib")
    joblib.dump(scaler, "./artifacts/scaler_xgb.joblib")

    # Output
    print(f"\n✅ Best Parameters: {best_params}")
    print(f"📊 MAE:  {mae:.3f}")
    print(f"📊 RMSE: {rmse:.3f}")
    print(f"📊 R²:   {r2:.3f}")

    # Save metrics for comparison
    metrics = {
        "Model": "MLP",
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 3)
    }

    with open("./results/metrics_xgb.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
