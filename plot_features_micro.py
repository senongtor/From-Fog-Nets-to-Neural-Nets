import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import read_dataset

def main():
    # Read the dataset.
    dataset_path = './dataset/'
    dataset_file_path = './dataset_file_path.csv'
    df_path = pd.read_csv(dataset_file_path)
    all_file_param = read_dataset.read_all_dataset(df_path)
    file_amount = len(all_file_param)

    # Initiate the plot.
    cmap = plt.get_cmap('jet_r')
    plt.figure(figsize=(15, 15))
    # plot_for_legend = plt.subplot()

    # Traverse all the dataset.
    for k in xrange(file_amount):
        file = all_file_param[k]

        # Use this block to set which dataset you want to find missing intervals.
        if 'Training' not in file.data_name:
            continue

        print '==========' + file.data_name + '=========='
        path = dataset_path + file.file_path
        df = pd.read_csv(path)
        allData = read_dataset.microData()
        allData.get_data(df.values)

    # plt.show()

main()
