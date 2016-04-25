import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import parser
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

def readdata(filepath):
    raw_in=pd.read_csv(filepath)
    X=raw_in.values[:]
    return X

def findprevday(y,time):
    defaulty=0;
    for i in xrange(len(y)):
        if (parser.parse(time) - timedelta(days=1))==parser.parse(y[i][0]):
            return y[i][1]
        elif time==y[i][0]:
            defaulty=y[i][0]

    return defaulty

def run_regression(X, y):
    clf = linear_model.Ridge(normalize=True)
    # clf = ensemble.BaggingRegressor(n_estimators=1000)
    clf.fit(X, y)
    return clf


def split_and_build_class(X, y):
    X_train = X[: 4061]
    X_test = X[4061:]
    y_train = y[: 4061]
    y_test = y[4061:]

    prevyield_train = np.zeros((len(y_train), 1))
    for k in xrange(12):
        prevyield_train[k]=y_train[k][1]

    for i in xrange(12, len(y_train)):
    #The desired previous time spot will not be more than 12 ahead.
        prevyield_train[i] = findprevday(y_train[i-12:i+1,:],X_train[i][0])
    X2_train = np.append(X_train, prevyield_train, axis=1)

    prevyield_test = np.zeros((len(y_test), 1))

    for i in xrange(len(y_test)):
        prevyield_test[i] = findprevday(y_test[i - 12:i + 1, :], X_test[i][0])
    X2_test = np.append(X_test, prevyield_test, axis=1)

    y_train=np.delete(y_train, np.s_[0], 1)
    y_test=np.delete(y_test, np.s_[0], 1)


    # Normalize the input data.
    imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp2 = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)

    fixed_X_train = X_train[:, 1:]
    fixed_X2_train=X2_train[:,1:]
    imp.fit(fixed_X_train)
    imp2.fit(fixed_X2_train)
    fixed_X_train = imp.transform(fixed_X_train)
    fixed_X2_train=imp2.transform(fixed_X2_train)
    # preprocessing.normalize(fixed_X_train, copy=False)
    X_train[:, 1:] = fixed_X_train
    X2_train[:,1:] = fixed_X2_train

    fixed_X_test = X_test[:, 1:]
    fixed_X2_test = X2_test[:, 1:]
    imp.fit(fixed_X_test)
    imp2.fit(fixed_X2_test)
    fixed_X_test = imp.transform(fixed_X_test)
    fixed_X2_test = imp2.transform(fixed_X2_test)
    # preprocessing.normalize(fixed_X_test, copy=False)
    X_test[:, 1:] = fixed_X_test
    X2_test[:, 1:] = fixed_X2_test

    return [X_train, X_test, y_train, y_test, X2_train, X2_test]


def main():
    filepathX='./dataset/Training set Microclimate (2 hour intervals).csv'
    filepathy='./dataset/Target Variable Water Yield.csv'
    X=readdata(filepathX)
    y=readdata(filepathy)
    X_train,X_test,y_train,y_test,X2_train,X2_test=split_and_build_class(X,y)

    clf = run_regression(X_train[:, 1:], y_train)
    y_hat_test = clf.predict(X_test[:, 1:])

    print 'Time series loss w/o yield as feature=', mean_squared_error(y_test, y_hat_test)
    clf2 = run_regression(X2_train[:, 1:], y_train)
    y2_hat_test = clf2.predict(X2_test[:, 1:])

    print 'Time series loss with yield as feature=', mean_squared_error(y_test, y2_hat_test)

main()