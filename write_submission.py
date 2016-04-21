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