import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import svm

import write_submission
import read_dataset
import Binary_Classification

def split_and_build_class(X, y):
    X_train = X[: 4061]
    X_test = X[4061:]
    y_train = y[: 4061]
    y_test = y[4061:]
    print X_train.shape
    print X_test.shape

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

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
    clf = svm.LinearSVC()
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

        y_train_binary = Binary_Classification.transform_to_binary(y_train)

        # Run SVM.
        clf = run_regression(X_train[:, 1:], y_train_binary)
        y_hat_test_binary = clf.predict(X_test[:, 1:])
        print 'Number of Class 1 in Training Data:', np.count_nonzero(y_train_binary)
        print 'Number of Class 1 in Test Data:', np.count_nonzero(y_test)
        print 'Number of Class 1 in Predicted Data:', np.count_nonzero(y_hat_test_binary)

        # Run Ridge Regression.
        X_train_regression = []
        y_train_regression = []
        X_test_regression = []
        y_hat_test_regression = []

        for i in xrange(len(y_train_binary)):
            if y_train_binary[i] != 0:
                X_train_regression.append(X_train[i])
                y_train_regression.append(y_train[i])
        X_train_regression = np.array(X_train_regression)
        clf_regression = linear_model.Ridge()
        clf_regression.fit(X_train_regression[:, 1:], y_train_regression)

        for i in xrange(len(y_hat_test_binary)):
            if y_hat_test_binary[i] != 0:
                X_test_regression.append(X_test[i])
        X_test_regression = np.array(X_test_regression)
        if len(X_test_regression):
            y_hat_test_regression = clf_regression.predict(X_test_regression[:, 1:])

        j = 0
        y_hat_test = []
        if len(X_test_regression):
            for i in xrange(len(y_hat_test_binary)):
                if y_hat_test_binary[i] == 0:
                    y_hat_test.append(y_hat_test_binary[i])
                else:
                    y_hat_test.append(y_hat_test_regression[j])
                    j += 1
            y_hat_test = np.array(y_hat_test)
        else:
            y_hat_test = y_hat_test_binary

        cmap = plt.get_cmap('jet_r')
        plt.figure(figsize=(10, 10))

        interval = file.interval
        intervel_minute = read_dataset.get_interval_minute(interval)

        test_size = y_hat_test.shape[0]
        plt.plot([i for i in xrange(test_size)], y_hat_test)
        plt.plot([i for i in xrange(test_size)], y_test)
        plt.legend(['Prediction', 'Real'])
        plt.suptitle('Time series of all points.')
        plt.savefig('time_series_all_points_SVM_and_ridge_regression.png', bbox_inches='tight')

        # print 'Time series loss =', clf.score(X_test[:, 1:], y_test)
        print 'Time series loss =', mean_squared_error(y_test, y_hat_test)

        '''
        =======================================================================
        '''

        # Predict test and write submission
        submission_file_name = 'Submission format'
        submission_file = None
        test_file_name = 'Test set Microclimate (2 hour intervals)'
        test_file = None

        for k in xrange(file_amount):
            file = all_file_param[k]
            if file.data_name == submission_file_name:
                submission_file = file
                break
        submission_path = dataset_path + submission_file.file_path
        df_submission = pd.read_csv(submission_path, index_col=0, parse_dates=[0])

        for k in xrange(file_amount):
            file = all_file_param[k]
            if file.data_name == test_file_name:
                test_file = file
                break
        test_path = dataset_path + test_file.file_path
        df_test = pd.read_csv(test_path, index_col=0, parse_dates=[0])

        X_combined = write_submission.combine_table(df_submission, df_test)
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        fixed_X = X_combined.values[:, 0:]
        imp.fit(fixed_X)
        X_combined.values[:, 0:] = imp.transform(fixed_X)
        y_submission = write_submission.write_submission(
            X_combined, clf, df_submission, 'Binary Classifier SVM Submission')

main()