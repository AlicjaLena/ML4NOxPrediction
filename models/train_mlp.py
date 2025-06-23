import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from utils.metrics_mlp import evaluate_model_mlp

from features.preprocess import load_and_split_data
from features.feature_engineering import add_engineered_features, select_features


def build_model(input_dim):
    """
    Builds and compiles an MLP model for regression.
    """
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim, kernel_regularizer='l2'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Regression output
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model


def train_model(model, X_train, y_train):
    """
    Trains the MLP model with early stopping.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=300,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )
    return history


def plot_learning_curve(history):
    """
    Plots the training and validation loss curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('MLP Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./results/figures/mlp_learning_curve.png', dpi=300)
    plt.show()


def main():
    # Load and prepare data
    train_df, val_df, _ = load_and_split_data()
    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)

    X_train = select_features(train_df)
    y_train = train_df["NOX"].values.reshape(-1, 1)

    X_val = select_features(val_df)
    y_val = val_df["NOX"].values.reshape(-1, 1)

    # Scaling
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val)

    # Build and train
    model = build_model(X_train_scaled.shape[1])
    history = train_model(model, X_train_scaled, y_train_scaled)

    # Evaluate
    mae, rmse, r2 = evaluate_model_mlp(model, scaler_y, X_val_scaled, y_val_scaled)

    print(f"\n📊 MLP MAE:  {mae:.2f}")
    print(f"📊 MLP RMSE: {rmse:.2f}")
    print(f"📊 MLP R²:   {r2:.3f}")

    # Save metrics for comparison
    metrics = {
        "Model": "MLP",
        "MAE": round(float(mae), 3),
        "RMSE": round(float(rmse), 3),
        "R2": round(float(r2), 3)
    }

    with open("./results/metrics_mlp.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save model and scalers
    model.save("./artifacts/best_model_mlp.keras")
    joblib.dump(scaler_X, "./artifacts/scaler_X_mlp.joblib")
    joblib.dump(scaler_y, "./artifacts/scaler_y_mlp.joblib")

    # Plot loss
    plot_learning_curve(history)


if __name__ == "__main__":
    main()
