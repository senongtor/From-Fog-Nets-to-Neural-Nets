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
    plot_files = ['Training set Microclimate (2 hour intervals)']
    # plot_files = ['Training set Microclimate (2 hour intervals)',
    #               'Training set Microclimate (5 minute intervals)']

    # Set up features for micro files.
    micro_features = read_dataset.set_features(dataset_path, plot_files)


main()