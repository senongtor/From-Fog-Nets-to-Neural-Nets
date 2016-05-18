import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil import parser
def readdata(filepath):
    raw_in=pd.read_csv(filepath)

    X=raw_in.values[:]
    return X

def main():
    # filep='/Users/Hongtao/From-Fog-Nets-to-Neural-Nets/dataset/Target Variable Water Yield.csv'
    # XX=readdata(filep)
    # x=[]
    # print XX[:,0]
    # for i in xrange(XX.shape[0]):
    #     x.append(parser.parse(XX[i,0]))
    #
    #
    # plt.plot(x, XX[:,1],'ro',color='blue')
    # plt.title('Yield-Date time')
    # plt.gcf().autofmt_xdate()
    # plt.xlabel('Date time')
    # plt.ylabel('Yield')
    # plt.show()
    x=[]
    y=[]
    x.append(0.332)
    x.append(0.126)
    x.append(0.065)
    x.append(0.331)
    x.append(0.36)
    x.append(0.138)
    x.append(0.425)
    x.append(0.034)
    x.append(0.0736)
    for i in xrange(9):
        y.append(i)
    plt.plot(y,x,'ro',color='blue')
    plt.title('Macro Features Correlation(Absolute Value)')
    plt.xlim(-1, 9)
    plt.show()
    # x = []
    # y = []
    # t = []
    # fig = plt.figure()
    # rect = fig.patch
    # rect.set_facecolor('#31312e')
    # readFile = open('data.txt', 'r')
    # sepFile = readFile.read().split('\n')
    # readFile.close()
    # for idx, plotPair in enumerate(sepFile):
    #     if plotPair in '. ':
    #         # skip. or space
    #         continue
    #     if idx > 1:  # to skip the first line
    #         xAndY = plotPair.split(',')
    #         time_string = xAndY[0]
    #         time_string1 = datetime.strptime(time_string, '%d/%m/%Y %H:%M')
    #         t.append(time_string1)
    #         y.append(float(xAndY[1]))
    # ax1 = fig.add_subplot(1, 1, 1, axisbg='white')
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y %H:%M'))
    # ax1.plot(t, y, 'c', linewidth=3.3)
    # plt.title('IRRADIANCE')
    # plt.xlabel('TIME')
    # fig.autofmt_xdate(rotation=45)
    # fig.tight_layout()
    # fig.show()



main()