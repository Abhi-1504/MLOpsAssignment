import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(raw_data_path, processed_data_path):
    """
    Loads raw data, performs train-test split, imputes missing values,
    engineers new features, and saves the processed dataframes.
    """
    print("--- Starting data preprocessing stage ---")
    
    housing = pd.read_csv(raw_data_path)
    
    y = housing['median_house_value']
    X = housing.drop('median_house_value', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    median_bedrooms = X_train['total_bedrooms'].median()
    X_train['total_bedrooms'].fillna(median_bedrooms, inplace=True)
    X_test['total_bedrooms'].fillna(median_bedrooms, inplace=True)

    X_train['rooms_per_household'] = X_train['total_rooms'] / X_train['households']
    X_train['bedrooms_per_room'] = X_train['total_bedrooms'] / X_train['total_rooms']
    X_test['rooms_per_household'] = X_test['total_rooms'] / X_test['households'] 
    X_test['bedrooms_per_room'] = X_test['total_bedrooms'] / X_test['total_rooms']  

    print("Feature engineering complete.")

    os.makedirs(processed_data_path, exist_ok=True)

    X_train.to_csv(os.path.join(processed_data_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'y_test.csv'), index=False)

    print(f"Processed data saved to {processed_data_path}")

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = ['ocean_proximity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Fit and save the preprocessor
    preprocessor.fit(X_train)
    os.makedirs(processed_data_path, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(processed_data_path, 'preprocessor.pkl'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--processed_data", type=str, required=True)
    args = parser.parse_args()
    preprocess_data(args.raw_data, args.processed_data)
