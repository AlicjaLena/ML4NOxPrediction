from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, scaler, X_val, y_val):
    """
    Applies a trained model and scaler to validation data,
    then calculates and returns MAE, RMSE, and R².
    """
    X_val_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_val_scaled)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    return mae, rmse, r2
