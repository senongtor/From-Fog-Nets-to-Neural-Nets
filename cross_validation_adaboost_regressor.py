import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

import read_dataset

def split_and_build_class(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print X_train.shape
    print X_test.shape

    # Normalize the input data.
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    fixed_X_train = X_train[:, 1:]
    imp.fit(fixed_X_train)
    fixed_X_train = imp.transform(fixed_X_train)
    preprocessing.normalize(fixed_X_train, copy=False)
    X_train[:, 1:] = fixed_X_train

    fixed_X_test = X_test[:, 1:]
    imp.fit(fixed_X_test)
    fixed_X_test = imp.transform(fixed_X_test)
    preprocessing.normalize(fixed_X_test, copy=False)
    X_test[:, 1:] = fixed_X_test

    train_data = read_dataset.microData()
    train_data.get_data(X_train)
    y_train = train_data.set_output(y_train)
    test_data = read_dataset.microData()
    test_data.get_data(X_test)
    y_test = test_data.set_output(y_test)

    return [X_train, X_test, y_train, y_test, train_data, test_data]

def run_regression(X, y):
    clf = ensemble.AdaBoostRegressor(n_estimators=1000)
    clf.fit(X, y)
    return clf

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
        X_train, X_test, y_train, y_test, train_data, test_data = split_and_build_class(df.values, yield_df.values)
        # [train_data, test_data] = split_and_build_class(df.values, yield_df.values)

        # Run Ridge Regression.
        clf = run_regression(X_train[:, 1:], y_train)
        y_hat_test = clf.predict(X_test[:, 1:])

        cmap = plt.get_cmap('jet_r')
        plt.figure(figsize=(10, 10))

        interval = file.interval
        intervel_minute = read_dataset.get_interval_minute(interval)

        test_size = y_hat_test.shape[0]
        plt.plot([i for i in xrange(test_size)], y_hat_test)
        plt.plot([i for i in xrange(test_size)], y_test)
        plt.legend(['Prediction', 'Real'])
        plt.suptitle('Cross validation + Adaboost Regressor')
        plt.savefig('Cross validation + Adaboost Regressor.png', bbox_inches='tight')

        loss = np.sqrt(mean_squared_error(y_test, y_hat_test))
        print 'Cross validation + Adaboost Regressor loss =', loss
        
main()