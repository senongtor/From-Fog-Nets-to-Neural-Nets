import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import read_dataset

'''
Fix the interval with start time and end time.
@param {!datetime} start_time
@param {!datetime} end_time
@param {!Array<!datetime>} interval
@return {!Array<!Array<!datetime>>}
'''
def fix_interval(start_time, end_time, interval):
    result = []
    start_time_of_today = datetime.datetime(
        interval[0].year, interval[0].month, interval[0].day,
        start_time.hour, start_time.minute)
    start_time_of_tomorrow = start_time_of_today + datetime.timedelta(days=1)
    end_time_of_today = datetime.datetime(
        interval[0].year, interval[0].month, interval[0].day,
        end_time.hour, end_time.minute)
    if start_time == end_time:
        end_time_of_today += datetime.timedelta(days=1)

    if interval[1] <= interval[0]:
        return result
    elif interval[1] <= end_time_of_today:
        return [[max(interval[0], start_time_of_today), interval[1]]]
    elif interval[0] < end_time_of_today:
        result.append([interval[0], end_time_of_today])
        result += fix_interval(start_time, end_time, [start_time_of_tomorrow, interval[1]])
        return result
    else:
        result += fix_interval(start_time, end_time, [start_time_of_tomorrow, interval[1]])
        return result

def main():
    # Read the dataset.
    dataset_path = './dataset/'
    dataset_file_path = './dataset_file_path.csv'
    df_path = pd.read_csv(dataset_file_path)
    all_file_param = read_dataset.read_all_dataset(df_path)
    file_amount = len(all_file_param)

    # Initiate the plot.
    cmap = plt.get_cmap('jet_r')
    plt.figure(figsize=(10, 10))
    plot_for_legend = plt.subplot()

    # Traverse all the dataset.
    for k in xrange(file_amount):
        file = all_file_param[k]

        # Use this block to set which dataset you want to find missing intervals.
        # if file.data_name != 'Macroclimate Guelmim Airport':
            # continue

        print '==========' + file.data_name + '=========='
        path = dataset_path + file.file_path
        df = pd.read_csv(path)
        data_time = df.values[:, 0]

        interval = file.interval
        start_time = datetime.datetime.strptime(file.start_time,'%H:%M')
        end_time = datetime.datetime.strptime(file.end_time,'%H:%M')
        missing_data = []

        prev_time = read_dataset.assign_time(data_time[0])
        current_time = None

        color = cmap(float(k) / file_amount)

        for i in range(1, data_time.shape[0]):
            tmp = data_time[i]
            current_time = read_dataset.assign_time(tmp)
            diff = current_time - prev_time

            # Find out missing intervals with gap larger that the default interval.
            if diff.days or diff.seconds / 60 > read_dataset.get_interval_minute(interval):
                missing_interval = [prev_time, current_time]
                missing_data += fix_interval(start_time, end_time, missing_interval)
            prev_time = current_time

        # Output all the missing intervals.
        for item in missing_data:
            print item[0].strftime('%Y-%m-%d %H:%M:%S'), \
                item[1].strftime('%Y-%m-%d %H:%M:%S')
        print len(missing_data), ' missing intervals are found.'

        # Plot the missing intervals.
        for item in missing_data:
            plt.plot(item, [(k + 1) for j in xrange(2)], c=color)
        plt.ylim([0, file_amount + 1])
        plot_for_legend.plot([], [], c=color, label=file.data_name)

    # Set the position, legend, and subtitle of the plot.
    box = plot_for_legend.get_position()
    plot_for_legend.set_position([box.x0, box.y0, box.width, box.height * 0.6])
    plot_for_legend.set_position([box.x0, box.y0 + box.height * 0.2,
                                  box.width, box.height * 0.8])
    legend = plot_for_legend.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                                    fancybox=True, shadow=True, ncol=2)
    plt.suptitle('Missing Intervals for the Dataset')
    plt.savefig('missing_intervals_for_the_dataset.png')

main()
