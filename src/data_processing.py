import pandas as pd


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


def load_train_and_test_data(train_filepath, test_filepath):
    """
    Load training and testing data from CSV files.

    Parameters:
    - train_filepath (str): The path to the training CSV file.
    - test_filepath (str): The path to the testing CSV file.

    Returns:
    - tuple: A tuple containing the training and testing DataFrames.
    """
    train_data = load_data(train_filepath)
    test_data = load_data(test_filepath)

    return train_data, test_data
