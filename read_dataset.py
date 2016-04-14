import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

'''
Create parameter object for each file.
'''
class fileParam:
    def __init__(self):
        self.data_name = ''
        self.file_path = ''
        self.start_time = ''
        self.end_time = ''
        self.interval = ''

    def get_param(self, param):
        self.data_name = param[0]
        self.file_path = param[1]
        self.start_time = param[2]
        self.end_time = param[3]
        self.interval = param[4]

'''
Create data object for micro file.
'''
class microData:
    def __init__(self):
        self.data_time = []
        self.percip_mm = []
        self.humidity = []
        self.temp = []
        self.leafwet450_min = []
        self.leafwet460_min = []
        self.leafwet_lwscnt = []
        self.gusts_ms = []
        self.wind_dir = []
        self.wind_ms = []

    def get_data(self, data):
        raw_data_time = data[:, 0]
        fixed_data_time = []
        for time in raw_data_time:
            fixed_time = assign_time(time)
            fixed_data_time.append(fixed_time)
        self.data_time = fixed_data_time
        self.percip_mm = data[:, 1]
        self.humidity = data[:, 2]
        self.temp = data[:, 3]
        self.leafwet450_min = data[:, 4]
        self.leafwet460_min = data[:, 5]
        self.leafwet_lwscnt = data[:, 6]
        self.gusts_ms = data[:, 7]
        self.wind_dir = data[:, 8]
        self.wind_ms = data[:, 9]

    def read_feature(self, feature):
        if feature == 'percip_mm':
            return self.percip_mm
        elif feature == 'humidity':
            return self.humidity
        elif feature == 'temp':
            return self.temp
        elif feature == 'leafwet450_min':
            return self.leafwet450_min
        elif feature == 'leafwet460_min':
            return self.leafwet460_min
        elif feature == 'leafwet_lwscnt':
            return self.leafwet_lwscnt
        elif feature == 'gusts_ms':
            return self.gusts_ms
        elif feature == 'wind_dir':
            return self.wind_dir
        elif feature == 'wind_ms':
            return self.wind_ms
        else:
            print 'Wrong feature', feature
            return []

'''
Assign the time from raw string in the file to datetime object.
@param {string} time
@return {!datetime}
'''
def assign_time(time):
    tmp = datetime.datetime.strptime('', '')
    if '-' in time:
        tmp = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    elif '/' in time:
        date_and_time = time.split(' ')
        date = date_and_time[0].split('/')
        date[2] = '20' + date[2]
        time = '/'.join(date) + ' ' + date_and_time[1]
        tmp = datetime.datetime.strptime(time, '%m/%d/%Y %H:%M')
    else:
        print 'invalid time format'
    return tmp

'''
Get the interval from raw string in the file to number of minute.
@param {string} interval
@return {number}
'''
def get_interval_minute(interval):
    minute = 0
    if interval[-1] == 'h':
        minute = int(interval[:-1]) * 60
    elif interval[-1] == 'm':
        minute = int(interval[:-1])
    else:
        print 'invalid parameter'
    return minute

'''
Read all dataset and save parameters for each file.
@param {string} path
@return {!Array<!fileParam>}
'''
def read_all_dataset(path):
    all_file_param = []
    for row in path.values:
        file_param = fileParam()
        file_param.get_param(row)
        all_file_param.append(file_param)
    return all_file_param
