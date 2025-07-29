from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def save_data():
    # Load built-in dataset from sklearn
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Create folder if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)

    # Save dataset as CSV
    df.to_csv("data/raw/housing.csv", index=False)
    print("✅ Data saved at data/raw/housing.csv")

if __name__ == "__main__":
    save_data()
