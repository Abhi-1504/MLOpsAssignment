import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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
    
    # Fill missing values
    median_bedrooms = X_train['total_bedrooms'].median()
    X_train['total_bedrooms'].fillna(median_bedrooms, inplace=True)
    X_test['total_bedrooms'].fillna(median_bedrooms, inplace=True)
    
    # Feature engineering
    X_train['rooms_per_household'] = X_train['total_rooms'] / X_train['households']
    X_train['bedrooms_per_room'] = X_train['total_bedrooms'] / X_train['total_rooms']
    X_test['rooms_per_household'] = X_test['total_rooms'] / X_test['households']
    X_test['bedrooms_per_room'] = X_test['total_bedrooms'] / X_test['total_rooms']
    
    print("Feature engineering complete.")
    
    # Save processed data
    os.makedirs(processed_data_path, exist_ok=True)
    X_train.to_csv(os.path.join(processed_data_path, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_data_path, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_path, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_path, 'y_test.csv'), index=False)
    
    print(f"Processed data saved to {processed_data_path}")
    
    # Create preprocessor that matches experimentation notebook
    # Use the same feature selection as Decision Tree experimentation
    dt_numeric_features = X_train.select_dtypes(include=np.number).columns.tolist() + \
                         ['rooms_per_household', 'bedrooms_per_room']
    dt_categorical_features = ['ocean_proximity']
    
    # Create the same transformers as experimentation
    numeric_transformer_dt = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer_dt = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create the exact same preprocessor as experimentation
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_dt, dt_numeric_features),
            ('cat', categorical_transformer_dt, dt_categorical_features)
        ],
        remainder='drop'  # This drops unused features and prevents string-to-float errors
    )
    
    # Fit and save the preprocessor
    preprocessor.fit(X_train)
    os.makedirs(processed_data_path, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--processed_data", type=str, required=True)
    args = parser.parse_args()
    
    preprocess_data(args.raw_data, args.processed_data)
