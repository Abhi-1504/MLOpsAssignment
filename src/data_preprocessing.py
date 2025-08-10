import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(raw_data_path, processed_data_path):
    """
    Loads raw data, performs preprocessing, and saves the
    train/test splits to the processed data folder.
    """
    print("Starting data preprocessing...")
    
    # Load data
    housing = pd.read_csv(raw_data_path)
    
    # Separate target variable
    y = housing['median_house_value']
    X = housing.drop('median_house_value', axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Impute missing 'total_bedrooms' with the median of the training set
    median_bedrooms = X_train['total_bedrooms'].median()
    X_train['total_bedrooms'].fillna(median_bedrooms, inplace=True)
    X_test['total_bedrooms'].fillna(median_bedrooms, inplace=True)

    # Create new ratio features
    X_train['rooms_per_household'] = X_train['total_rooms'] / X_train['households']
    X_train['bedrooms_per_room'] = X_train['total_bedrooms'] / X_train['total_rooms']
    
    X_test['rooms_per_household'] = X_test['total_rooms'] / X_test['households']
    X_test['bedrooms_per_room'] = X_test['total_bedrooms'] / X_test['total_rooms']

    print("Feature engineering complete.")

    # Create processed data directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)

    # Save the processed dataframes
    X_train.to_csv(os.path.join(processed_data_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'y_test.csv'), index=False)

    print(f"Processed data saved to {processed_data_path}")

if __name__ == "__main__":
    preprocess_data('data/raw/housing.csv', 'data/processed')
