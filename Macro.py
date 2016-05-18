import numpy as np
import pandas as pd
import math
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

def convert(direction):
    if direction=='variable wind direction':
        return 8
    elif direction=='Calm, no wind':
        return 0
    else:
        print direction[22:]
        direction=direction[22:]

        dict={'north':2,'northeast':4,'northwest':6,'south':8,'southeast':10,'southwest':12,'west':14,'east':16, }
        if('-' in direction):
            return (dict[direction[0: direction.index('-')]]+dict[direction[direction.index('-')+1: ]])/2
        else:
            return dict[direction]
def convertVV(val):

    if val=='10.0 and more':
        return 10
    return float(val)

def convertcloud(val):
    if val=='No Significant Clouds':
        return 0
    elif 'few clouds' in val.lower():
        return 0.2
    elif 'broken clouds' in val.lower():
        return 0.7
    elif 'scattered clouds' in val.lower():
        return 0.55
    elif 'overcast' in val.lower():
        return 1.0

def main():
    targettime = set()
    for i in range(0, 24, 2):
        targettime.add(str(i))

    filepathX='/Users/Hongtao/From-Fog-Nets-to-Neural-Nets/dataset/Macroclimate Guelmim Airport.csv'
    filepathMX='/Users/Hongtao/From-Fog-Nets-to-Neural-Nets/dataset/Training set Microclimate (2 hour intervals).csv'
    filepathy='/Users/Hongtao/From-Fog-Nets-to-Neural-Nets/dataset/Target Variable Water Yield.csv'
    X=readdata(filepathX)
    y=readdata(filepathy)
    MX=readdata(filepathMX)
    dayavg=[]

    X = np.delete(X, 7, 1)
    X = np.delete(X, 7, 1)
    X = np.delete(X, 7, 1)

    # Get all the cloud conditions
    # l=set()
    # for ii in xrange(X.shape[0]):
    #     if pd.isnull(X[ii, 7]) == False:
    #         l.add(X[ii, 7])
    # print l
    count=0

    total=np.zeros(9)
    for k in xrange(X.shape[0]):
        if pd.isnull(X[k,1])==False:
            X[k, 1]=float(X[k, 1])
            total[0]+=float(X[k,1])
        if pd.isnull(X[k,2]) == False:
            X[k,2]=float(X[k, 2])
            total[1]+=float(X[k,2])
        if pd.isnull(X[k, 3]) == False:
            X[k,3]=float(X[k, 3])
            total[2]+= float(X[k, 3])
        if pd.isnull(X[k, 4]) == False:
            total[3]+= float(X[k, 4])
        if pd.isnull(X[k,5]) == False:
            X[k, 5]=convert(X[k,5])
            total[4]+= convert(X[k,5])
        if pd.isnull(X[k,6]) == False:
            total[5]+=int(X[k,6])
        if pd.isnull(X[k,7]) == False:
            X[k,7]=convertcloud(X[k, 7])
            total[6]+= convertcloud(X[k,7])
        if pd.isnull(X[k,8]) == False:
            X[k,8]=convertVV(X[k,8])
            total[7]+= convertVV(X[k,8])
        if pd.isnull(X[k,9]) == False:
            total[8]+=int(X[k,9])
    total=np.multiply(total,1.0/X.shape[0])

    print total
    # Get rid of useless time spots
    alldates = y[:, 0]
    for j in xrange(alldates.shape[0]):
        alldates[j] = parser.parse(alldates[j])

    indices = [i for i in xrange(X.shape[0]) if parser.parse(X[i,0]) not in alldates]
    X=np.delete(X,indices,0)



    validdates=[]
    # X_sub=[]

    #
    # count=0
    #
    # for i in xrange(X.shape[0]):
    #
    #     if(parser.parse(X[i,0]) in alldates):
    #         count+=1
    #         validdates.append(X[i,0])




    # targetdates=np.zeros(count,2)
    # for j in xrange(count):
    #     targetdates[j,:]=X[j,:]
    #
    # print targetdates
    # X_Adj=X[np.where(str(parser.parse(X[:, 0]).time().hour) in targettime)]
    # X_Adj=np.all(str(parser.parse(X[:, 0]).time().hour) in targettime, axis=1)


    # for i in xrange(X.shape[0]):
    #     if str(parser.parse(X[i, 0]).time().hour) not in targettime:
    #         np.delete()
    #     elif str(parser.parse(X[i, 0]).time().hour) in targettime:
    #         Line=np.zeroes(13)
    #         Line=X[i,:]
    #
    #         if pd.isnull(Line[5])==False:
    #             Line[5]=str(convert(Line[5]))
    #         X_Adj.append(Line)




        # X_train,X_test,y_train,y_test,X2_train,X2_test=split_and_build_class(X,y)
    #
    # clf = run_regression(X_train[:, 1:], y_train)
    # y_hat_test = clf.predict(X_test[:, 1:])
    #
    # print 'Time series loss w/o yield as feature=', mean_squared_error(y_test, y_hat_test)
    # clf2 = run_regression(X2_train[:, 1:], y_train)
    # y2_hat_test = clf2.predict(X2_test[:, 1:])
    #
    # print 'Time series loss with yield as feature=', mean_squared_error(y_test, y2_hat_test)


main()