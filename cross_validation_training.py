import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

import read_dataset

def split_and_build_class(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    train_data = read_dataset.microData()
    train_data.get_data(X_train)
    train_data.set_output(y_train)
    test_data = read_dataset.microData()
    test_data.get_data(X_test)
    test_data.set_output(y_test)
    return [train_data, test_data]

def main():
    # Read the dataset.
    dataset_path = './dataset/'
    dataset_file_path = './dataset_file_path.csv'
    df_path = pd.read_csv(dataset_file_path)
    all_file_param = read_dataset.read_all_dataset(df_path)
    file_amount = len(all_file_param)
    plot_files = ['Training set Microclimate (2 hour intervals)']
    yield_file = 'Target Variable Water Yield'
    # plot_files = ['Training set Microclimate (2 hour intervals)',
    #               'Training set Microclimate (5 minute intervals)']

    # Set up features for micro files.
    micro_features = read_dataset.set_features(dataset_path, plot_files)

    # Read yield file for micro training file.
    yield_df = None
    for k in xrange(file_amount):
        file = all_file_param[k]
        if file.data_name == yield_file:
            yield_path = dataset_path + file.file_path
            yield_df = pd.read_csv(yield_path)
            break

    # Traverse all the dataset.
    for k in xrange(file_amount):
        file = all_file_param[k]

        if file.data_name not in plot_files:
            continue
        print '==========' + file.data_name + '=========='

        path = dataset_path + file.file_path
        df = pd.read_csv(path)

        # Split the micro training file into training dataset and test dataset.
        [train_data, test_data] = split_and_build_class(df.values, yield_df.values)

main()