import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

def load_and_preprocess_data():
    # Load the iris dataset
    iris = load_iris()

    # Create the dataframe
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Convert numeric targets to species names using iris.target_names
    # This is more robust than manual mapping
    df['target'] = df['target'].map({i: name for i, name in enumerate(iris.target_names)})

    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/iris.csv', index=False)

    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)
    print("Sample training labels:")
    print(y_train.head())
    print("Sample testing labels:")
    print(y_test.head())
    print("Data preprocessing complete. Raw data saved to 'data/raw/iris.csv'.")
    
    # Additional validation
    print(f"\nClass distribution in training set:")
    print(y_train.value_counts().sort_index())
    print(f"\nClass distribution in testing set:")
    print(y_test.value_counts().sort_index())