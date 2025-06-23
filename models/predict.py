import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

from features.feature_engineering import add_engineered_features, select_features

def load_model_and_data():
    model = joblib.load("./artifacts/best_model_xgb.joblib")
    # scaler = joblib.load("./artifacts/scaler_xgb.joblib")  # optional

    test_df = pd.read_csv("./data/test_data.csv")
    return model, test_df


def predict_nox(model, test_df):
    test_df = add_engineered_features(test_df)
    X_test = select_features(test_df)
    y_pred = model.predict(X_test)

    test_df["NOX_predicted"] = y_pred
    test_df[["NOX_predicted"]].to_csv("./results/NOX_predictions_from_test.csv", index=False)

    return test_df, y_pred


def plot_prediction_distribution(y_pred, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=30, color='skyblue', edgecolor='black')
    plt.title("Histogram of Predicted NOX Values")
    plt.xlabel("Predicted NOX")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_comparison_with_train(train_df, y_pred, save_path):
    plt.figure(figsize=(10, 6))
    plt.hist(train_df["NOX"], bins=30, alpha=0.5, label="Train NOX", color='lightgreen', edgecolor='black')
    plt.hist(y_pred, bins=30, alpha=0.5, label="Predicted NOX (Test)", color='skyblue', edgecolor='black')
    plt.title("NOX Distribution: Train vs Predicted (Test)")
    plt.xlabel("NOX")
    plt.ylabel("Sample Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    os.makedirs("./results/figures", exist_ok=True)

    model, test_df = load_model_and_data()
    test_df, y_pred = predict_nox(model, test_df)

    # Optional: compare to training distribution
    train_df = pd.read_csv("./data/train_data.csv")

    plot_prediction_distribution(
        y_pred,
        save_path="./results/figures/hist_predicted_nox.png"
    )

    plot_comparison_with_train(
        train_df,
        y_pred,
        save_path="./results/figures/compare_train_test_nox.png"
    )

    print("✅ Prediction and plots completed. Files saved in /results.")


if __name__ == "__main__":
    main()
