# -*- coding: utf-8 -*-

from __future__ import print_function  # do not delete this line if you want to save your log file.
from collections import Counter
from scipy.io import loadmat
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import random
import numpy as np

random_seed = 2333
train_data_root = '../data/Training/'
valid_data_root = '../data/Validation/'
train_csv_root = '../data/REFERENCE-v3-training-nodup.csv'
valid_csv_root = '../data/REFERENCE-v3-validation.csv'
used_train_csv_root = 'training-nodup.csv'
used_valid_csv_root = 'infer.csv'


def main():
    # Get overlap
    class_names = ['N', 'A', 'O', '~']
    train_df = pd.read_csv(train_csv_root)
    valid_df = pd.read_csv(valid_csv_root)
    for idx, item in enumerate(class_names):
        train_df.Label[train_df['Label'] == item] = idx
        valid_df.Label[valid_df['Label'] == item] = idx
    train_df['DataPath'] = train_data_root + train_df['ID'] + '.mat'
    valid_df['DataPath'] = valid_data_root + valid_df['ID'] + '.mat'
    train_data_list = train_df['DataPath'].tolist()
    valid_data_list = valid_df['DataPath'].tolist()
    trainset = np.zeros((len(train_data_list), 18000))
    validset = np.zeros((len(valid_data_list), 18000))

    for idx, item in enumerate(train_data_list):
        print('Loading record {}'.format(item))
        mat_data = loadmat(item)
        data = mat_data['val'].squeeze()
        data = np.nan_to_num(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        trainset[idx, :min(18000, len(data))] = data[:min(18000, len(data))]
    np.save("trainset.npy", trainset)

    for idx, item in enumerate(valid_data_list):
        print('Loading record {}'.format(item))
        mat_data = loadmat(item)
        data = mat_data['val'].squeeze()
        data = np.nan_to_num(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        validset[idx, :min(18000, len(data))] = data[:min(18000, len(data))]
    np.save("validset.npy", validset)

    train_df.drop(['DataPath'], axis=1, inplace=True)
    valid_df.drop(['DataPath'], axis=1, inplace=True)
    train_df.to_csv(used_train_csv_root, index=False)
    valid_df.to_csv(used_valid_csv_root, index=False)


if __name__ == "__main__":
    random.seed(random_seed)
    np.random.seed(random_seed)
    main()

