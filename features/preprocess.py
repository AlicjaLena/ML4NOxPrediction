import pandas as pd
import os

def load_and_split_data(data_dir="../data"):
    """Loads data from years and divides it into train, val, test chronologically."""
    # Wczytaj dane roczne
    df_2011 = pd.read_csv(os.path.join(data_dir, "gt_2011.csv"))
    df_2012 = pd.read_csv(os.path.join(data_dir, "gt_2012.csv"))
    df_2013 = pd.read_csv(os.path.join(data_dir, "gt_2013.csv"))
    df_2014 = pd.read_csv(os.path.join(data_dir, "gt_2014.csv"))
    df_2015 = pd.read_csv(os.path.join(data_dir, "gt_2015.csv"))

    # Podział chronologiczny
    train_df = pd.concat([df_2011, df_2012], ignore_index=True)
    val_df = df_2013.copy()
    test_df = pd.concat([df_2014, df_2015], ignore_index=True)

    return train_df, val_df, test_df

def prepare_features_and_targets(train_df, val_df):
    """Separates input features and target variable."""
    X_train = train_df.drop(columns=["NOX"])
    y_train = train_df["NOX"]

    X_val = val_df.drop(columns=["NOX"])
    y_val = val_df["NOX"]

    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    train_df, val_df, test_df = load_and_split_data()

    # Zapisz zbiory do CSV (opcjonalnie)
    os.makedirs("../data", exist_ok=True)
    train_df.to_csv("../data/train_data.csv", index=False)
    val_df.to_csv("../data/val_data.csv", index=False)
    test_df.to_csv("../data/test_data.csv", index=False)

    X_train, X_val, y_train, y_val = prepare_features_and_targets(train_df, val_df)

    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {test_df.shape}")
