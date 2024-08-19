import pandas as pd
import numpy as np
from config import RAW_TRAIN_PATH, RAW_TEST_PATH, CATEGORICAL_FEATURES, TARGET_VARIABLES


def load_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['datetime'])

    return df


def load_train_and_test_data():
    train_data = load_data(RAW_TRAIN_PATH)
    test_data = load_data(RAW_TEST_PATH)

    return train_data, test_data


def perform_categorical_conversion(data):
    for feature in CATEGORICAL_FEATURES:
        data[feature] = data[feature].astype('category')

    return data


def convert_datetime_to_timestamp(data):
    data['datetime'] = data['datetime'].astype(
        'int64') // (10**9 * 3600)  # Convert to hours

    return data

def transform_target_variable_data(data):
    data[TARGET_VARIABLES] = data[TARGET_VARIABLES].apply(lambda x: np.log(x + 1))
    
    return data


def preprocess_data(data):
    perform_categorical_conversion(data)
    convert_datetime_to_timestamp(data)


def preprocess_train_and_test_data(train_data, test_data):
    preprocess_data(train_data)
    preprocess_data(test_data)
