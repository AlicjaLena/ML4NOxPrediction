import pandas as pd

def load_metrics():
    """
    Load manually entered or previously saved model evaluation metrics.
    Replace with JSON/CSV reading if exporting metrics during training.
    """
    return {
        'Model': ['Random Forest', 'XGBoost', 'MLP'],
        'MAE': [2.280, 2.205, 2.516],
        'RMSE': [3.563, 3.424, 3.929],
        'R²': [0.885, 0.894, 0.860]
    }

def create_comparison_table(metrics):
    df = pd.DataFrame(metrics)
    return df.round(3)

def main():
    metrics = load_metrics()
    df = create_comparison_table(metrics)

    print("\n📊 Model Comparison:")
    print(df)

    df.to_csv("./results/model_comparison.csv", index=False)

if __name__ == "__main__":
    main()
