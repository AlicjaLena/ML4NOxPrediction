import pandas as pd

def add_engineered_features(df):
    """
    Adds engineered features to the given DataFrame based on domain knowledge.

    Parameters:
    - df (pd.DataFrame): DataFrame with raw features.

    Returns:
    - df (pd.DataFrame): DataFrame with new engineered features.
    """
    df = df.copy()
    
    # Feature: Pressure ratio (compressor vs atmospheric)
    df['Pressure_Ratio'] = df['CDP'] / df['AP']
    
    # Feature: Combustion efficiency (temperature drop across the turbine)
    df['Combustion_Efficiency'] = df['TIT'] - df['TAT']
    
    # Feature: Energy efficiency (energy per exhaust pressure unit)
    df['Energy_Efficiency'] = df['TEY'] / df['GTEP']
    
    # Feature: Humidity to temperature ratio
    df['Humidity_Temperature_Ratio'] = df['AH'] / df['AT']
    
    return df


def select_features(df, feature_list=None):
    """
    Selects a subset of important features from the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with all features.
    - feature_list (list): Optional custom list of features to keep.

    Returns:
    - df_selected (pd.DataFrame): DataFrame with selected features only.
    """
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


if __name__ == "__main__":
    # Example usage for testing
    df = pd.read_csv("../data/train_data.csv")
    df = add_engineered_features(df)
    df_selected = select_features(df)

    print("Feature-engineered dataframe preview:")
    print(df_selected.head())
