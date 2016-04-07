import numpy as np
import pandas as pd
import datetime

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

def get_interval_minute(interval):
    minute = 0
    if interval[-1] == 'h':
        minute = int(interval[:-1]) * 60
    elif interval[-1] == 'm':
        minute = int(interval[:-1])
    else:
        print 'invalid parameter'
    return minute


dataset_path = './dataset/'
dataset_file_path = './dataset_file_path.csv'
df_path = pd.read_csv(dataset_file_path)
all_file_param = []
for row in df_path.values:
    file_param = fileParam()
    file_param.get_param(row)
    all_file_param.append(file_param)

for file in all_file_param:
    # if file.data_name != 'Macroclimate Agadir Airport':
    #     continue

    print '==========' + file.data_name + '=========='
    path = dataset_path + file.file_path
    df = pd.read_csv(path)
    data_time = df.values[:, 0]

    interval = file.interval
    start_time = file.start_time
    end_time = file.end_time
    missing_data = []

    prev_time = assign_time(data_time[0])
    current_time = None

    for i in range(1, data_time.shape[0]):
        tmp = data_time[i]
        current_time = assign_time(tmp)
        diff = current_time - prev_time
        if diff.days or diff.seconds / 60 > get_interval_minute(interval):
            missing_data.append((prev_time, current_time))
        prev_time = current_time

    # print len(missing_data)
    for item in missing_data:
        print item[0].strftime('%Y-%m-%d %H:%M:%S'), \
            item[1].strftime('%Y-%m-%d %H:%M:%S')
