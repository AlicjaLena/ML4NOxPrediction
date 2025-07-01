import os
import pandas as pd

def load_and_split_data(data_dir="./data"):
    """Loads data from 2011–2015 and divides it chronologically."""
    df_2011 = pd.read_csv(os.path.join(data_dir, "gt_2011.csv"))
    df_2012 = pd.read_csv(os.path.join(data_dir, "gt_2012.csv"))
    df_2013 = pd.read_csv(os.path.join(data_dir, "gt_2013.csv"))
    df_2014 = pd.read_csv(os.path.join(data_dir, "gt_2014.csv"))
    df_2015 = pd.read_csv(os.path.join(data_dir, "gt_2015.csv"))

    train_df = pd.concat([df_2011, df_2012], ignore_index=True)
    val_df = df_2013.copy()
    test_df = pd.concat([df_2014, df_2015], ignore_index=True)

    os.makedirs(data_dir, exist_ok=True)
    train_df.to_csv(os.path.join(data_dir, "train_data.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val_data.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test_data.csv"), index=False)

    return train_df, val_df, test_df


def add_engineered_features(df):
    """Adds engineering features to the dataset."""
    df = df.copy()
    df['Pressure_Ratio'] = df['CDP'] / df['AP']
    df['Combustion_Efficiency'] = df['TIT'] - df['TAT']
    df['Energy_Efficiency'] = df['TEY'] / df['GTEP']
    df['Humidity_Temperature_Ratio'] = df['AH'] / df['AT']
    return df


def select_features(df, feature_list=None):
    """Returns selected attributes."""
    if feature_list is None:
        feature_list = [
            'AT',
            'CO',
            'AP',
            'Energy_Efficiency',
            'Humidity_Temperature_Ratio',
            'Combustion_Efficiency'
        ]
    return df[feature_list]


def main():
    train_df, val_df, test_df = load_and_split_data()

    # Add features
    train_df = add_engineered_features(train_df)
    val_df = add_engineered_features(val_df)
    test_df = add_engineered_features(test_df)

    # Select features for learning and prediction
    train_df = select_features(train_df)
    val_df = select_features(val_df)
    test_df = select_features(test_df)

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = main()
    print("Data loaded and prepared")
