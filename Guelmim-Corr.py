# Correlation between each column of Macro-Guelmim and

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import splev, splrep
from datetime import datetime, timedelta
from dateutil import parser
from scipy.stats.stats import pearsonr
from sklearn.preprocessing import Imputer

def readdata(filepath):
    raw_in=pd.read_csv(filepath)
    X=raw_in.values[:]
    return X
# Col 10
def convertcloud(val):
    re=0
    l = val.lower().split(" ")

    if val=='No Significant Clouds':
        return 0
    if 'few clouds' in val.lower():
        re+=200.0/int(l[l.index('few')+3])
    if 'broken clouds' in val.lower():
        re+=700.0/int(l[l.index('broken')+3])
    if 'scattered clouds' in val.lower():
        re+=550.0/int(l[l.index('scattered')+3])
    if 'overcast' in val.lower():
        re+=1000.0/int(l[l.index('overcast')+2])
    return re
# col 5
def convertdir(direction):
    if direction=='variable wind direction':
        return 8
    elif direction=='Calm, no wind':
        return 0
    else:
        direction=direction[22:]

        dict={'north':2,'northeast':4,'northwest':6,'south':8,'southeast':10,'southwest':12,'west':14,'east':16, }
        if('-' in direction):
            return (dict[direction[0: direction.index('-')]]+dict[direction[direction.index('-')+1: ]])/2
        else:
            return dict[direction]

# col 11
def convertVV(val):

    if val=='10.0 and more':
        return 10
    return float(val)

filepathX='./dataset/Macroclimate Guelmim Airport.csv'
filepathy='./dataset/Target Variable Water Yield.csv'
filepathpred='./dataset/Submission format.csv'


yhat=readdata(filepathpred)
allthats=yhat[:,0]
for l in xrange(allthats.shape[0]):
    allthats[l] = parser.parse(allthats[l])

y=readdata(filepathy)
allydates=y[:,0]
for j in xrange(allydates.shape[0]):
    allydates[j] = parser.parse(allydates[j])


X=readdata(filepathX)
allxdates=X[:,0]
for k in xrange(allxdates.shape[0]):
    allxdates[k]= parser.parse(allxdates[k])
    if pd.isnull(X[k,5])==False:
        X[k,5]=convertdir(X[k,5])
    if pd.isnull(X[k,10])==False:
        X[k,10]=convertcloud(X[k,10])

print allxdates.shape
print allydates.shape

#Build a set holding all the intersected date
# find the intersection of macro data and target yield data
s=set()
s=[val for val in allxdates if val in allydates]
l=len(s)
#See the intersection of macro data and submission data
shat=[val for val in allxdates if val in allthats]
print 'LOOKK',len(shat)

indices=[i for i in xrange(X.shape[0]) if X[i,0] not in s]
X=np.delete(X,indices,0)
print X.shape

indices2=[i for i in xrange(y.shape[0]) if y[i,0] not in s]
y=np.delete(y,indices2,0)
print y.shape

validindex=[]
for i in xrange(13):
    if(i==7 or i==8 or i==9):
        continue
    validindex.append(i)
print validindex
for i in xrange(len(validindex)):

    X_=X[:,validindex[i]].reshape(X.shape[0],1)
    y_=y[:,1].reshape(y.shape[0],1)

    indicelist=[i for i in xrange(len(X_)) if pd.isnull(X_[i])]

    X_=np.delete(X_,indicelist,0)
    y_=np.delete(y_,indicelist,0)
    print 'st feature has correlation', scipy.stats.spearmanr(X_, y_)


# print X_
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imp.fit(X_)
# X_p=imp.transform(X_)
# print X_
# print y_



# clouds correlation 0.42
# T corr -0.33
# Wind dir 0.33. pay attention to conversion
# E corr 0.33
# Ff -0.35