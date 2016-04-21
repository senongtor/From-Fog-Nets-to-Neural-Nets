import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

import write_submission
import read_dataset

def run_regression(X, y):
    clf = linear_model.Ridge(normalize=True)
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

        # Run Ridge Regression.
        X_train_all = df.values[:, 1:]
        y_train_all = yield_df.values[:, 1:]
        imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(X_train_all)
        fixed_X_training = imp.transform(X_train_all)

        clf = run_regression(fixed_X_training, y_train_all)

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
            X_combined, clf, df_submission,
            'Ridge Regression Submission with All Training Data')

main()