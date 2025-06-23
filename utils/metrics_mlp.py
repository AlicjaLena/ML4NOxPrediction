from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model_mlp(model, scaler_y, X_val_scaled, y_val_scaled):
    """
    Predicts and evaluates MLP model using inverse scaling.
    """
    y_pred_scaled = model.predict(X_val_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_val = scaler_y.inverse_transform(y_val_scaled)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    return mae, rmse, r2
