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

def transform_to_binary(y):
    y_binary = []
    for i in xrange(len(y)):
        if y[i] > 0:
            y_binary.append(1)
        else:
            y_binary.append(0)
    return y_binary