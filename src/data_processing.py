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
    converted_data = data.copy()
    for feature in CATEGORICAL_FEATURES:
        converted_data[feature] = converted_data[feature].astype('category')

    return converted_data


def convert_datetime_to_timestamp(data):
    converted_data = data.copy()
    converted_data['datetime'] = converted_data['datetime'].astype(
        'int64') // (10**9 * 3600)

    return converted_data

def transform_target_variable_data(data):
    transformed_data = data.copy()
    transformed_data[TARGET_VARIABLES] = transformed_data[TARGET_VARIABLES].apply(lambda x: np.log(x + 1))
    
    return transformed_data


def preprocess_data(data):
    processed_data = data.copy()
    processed_data = perform_categorical_conversion(processed_data)
    processed_data = convert_datetime_to_timestamp(processed_data)
    processed_data = transform_target_variable_data(processed_data)
    
    return processed_data


def preprocess_train_and_test_data(train_data, test_data):
    processed_train_data = train_data.copy()
    processed_test_data = test_data.copy()
    processed_train_data = preprocess_data(processed_train_data)
    processed_test_data = preprocess_data(processed_test_data)
