import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import read_dataset

def plot_one_feature(plot_files, all_file_param, dataset_path, feature):
    file_amount = len(all_file_param)

    # Initiate the plot.
    cmap = plt.get_cmap('jet_r')
    plt.figure(figsize=(25, 10))
    # plot_for_legend = plt.subplot()

    # Color setup for single file.
    color = cmap(float(9) / file_amount)

    # Traverse all the dataset.
    print '==========' + feature + '=========='
    for k in xrange(file_amount):
        file = all_file_param[k]

        # Color setup for multiple files.
        # color = cmap(float(k) / file_amount)

        # Use this block to set which dataset you want to find missing intervals.
        if file.data_name not in plot_files:
            continue

        print 'Ploting ' + file.data_name + '...'
        path = dataset_path + file.file_path
        df = pd.read_csv(path)
        all_data = read_dataset.microData()
        all_data.get_data(df.values)

        interval = file.interval
        intervel_minute = read_dataset.get_interval_minute(interval)
        prev_time = all_data.data_time[0]
        for i in xrange(1, len(all_data.data_time)):
            current_time = all_data.data_time[i]
            diff = current_time - prev_time
            if not diff.days and diff.seconds / 60 <= intervel_minute:
                plt.plot(
                    [prev_time, current_time],
                    all_data.read_feature(feature)[i - 1: i + 1],
                    c=color
                )
            prev_time = current_time

            if i % 1000 == 0:
                print 'Plotted ', i, ' lines...'

        print 'Plotted done!'

    plt.legend(plot_files)
    plt.suptitle(feature + ' in ' + str(plot_files))
    plt.savefig('./micro_features_plot/' + feature + '.png', bbox_inches='tight')

def main():
    # Read the dataset.
    dataset_path = './dataset/'
    dataset_file_path = './dataset_file_path.csv'
    df_path = pd.read_csv(dataset_file_path)
    all_file_param = read_dataset.read_all_dataset(df_path)
    plot_files = ['Training set Microclimate (2 hour intervals)']
    # plot_files = ['Training set Microclimate (2 hour intervals)',
    #               'Training set Microclimate (5 minute intervals)']

    # Set up features for micro files.
    micro_features = read_dataset.set_features(dataset_path, plot_files)

    for feature in micro_features:
        plot_one_feature(plot_files, all_file_param, dataset_path, feature)

main()
