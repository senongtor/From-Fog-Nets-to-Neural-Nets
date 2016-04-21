import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

def combine_table(left, right):
    X = left.join(right, how='left')  # left join onto the format
    X = X[right.columns]  # now just subset back down to the input columns
    return X

def write_submission(X, estimator, file, new_name):
    y = estimator.predict(X)
    file['yield'] = y
    file.to_csv(new_name + '.csv')
    return y

def write_submission_binary_classifier_and_regression(X, binary, regression, file, new_name):
    y_hat_binary = binary.predict(X)

    X_regression = []
    y_hat_regression = []

    for i in xrange(len(y_hat_binary)):
        if y_hat_binary[i] != 0:
            X_regression.append(X.values[i])
    X_regression = np.array(X_regression)
    if len(X_regression):
        y_hat_regression = regression.predict(X_regression)

    j = 0
    y = []
    if len(X_regression):
        for i in xrange(len(y_hat_binary)):
            if y_hat_binary[i] == 0:
                y.append(y_hat_binary[i])
            else:
                y.append(y_hat_regression[j])
                j += 1
        y = np.array(y)
    else:
        y = y_hat_binary

    file['yield'] = y
    file.to_csv(new_name + '.csv')
    return y