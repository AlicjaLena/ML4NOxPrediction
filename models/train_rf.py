import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.metrics import evaluate_model


from features.preprocess import load_and_split_data
from features.feature_engineering import add_engineered_features, select_features


def train_model(X_train, y_train):
    """
    Trains a Random Forest Regressor using GridSearchCV.
    Returns the best model and the fitted scaler.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    rf = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    return grid_search.best_estimator_, scaler, grid_search.best_params_

def main():
    # Load and preprocess data
    train_df, val_df, _ = load_and_split_data()
    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)

    X_train = select_features(train_df)
    y_train = train_df["NOX"]

    X_val = select_features(val_df)
    y_val = val_df["NOX"]

    # Train model
    best_model, scaler, best_params = train_model(X_train, y_train)

    # Evaluate
    mae, rmse, r2 = evaluate_model(best_model, scaler, X_val, y_val)

    # Save model and scaler
    joblib.dump(best_model, "./artifacts/best_model_rf.joblib")
    joblib.dump(scaler, "./artifacts/scaler_rf.joblib")

    # Print results
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

    with open("./results/metrics_rf.json", "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
