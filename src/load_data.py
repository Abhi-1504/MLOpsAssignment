import pandas as pd

def load_data(data_path):
    """
    Loads the housing data from the specified path.
    This is a simple placeholder for the DVC pipeline.
    """
    print(f"Loading data from {data_path}...")
    return pd.read_csv(data_path)

if __name__ == "__main__":
    df = load_data('data/raw/housing.csv')
    print("Data loaded successfully.")
    print(df.head())
