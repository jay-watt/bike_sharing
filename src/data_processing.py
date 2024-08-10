import pandas as pd
from config import RAW_TRAIN_PATH, RAW_TEST_PATH


def load_data(filepath):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    - filepath (str): The path to the CSV file.

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    # Load the CSV file into a DataFrame
    df = pd.read_csv(filepath, parse_dates=['datetime'])

    return df


def load_train_and_test_data():
    """
    Load training and testing data from CSV files.

    Returns:
    - tuple: A tuple containing the training and testing DataFrames.
    """
    train_data = load_data(RAW_TRAIN_PATH)
    test_data = load_data(RAW_TEST_PATH)

    return train_data, test_data
