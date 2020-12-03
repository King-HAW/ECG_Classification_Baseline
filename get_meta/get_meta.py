# -*- coding: utf-8 -*-

from __future__ import print_function  # do not delete this line if you want to save your log file.
from collections import Counter
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
import random
import numpy as np
sns.set(style="darkgrid", context="notebook")

random_seed = 2333
train_data_root = '../data/Training/'
valid_data_root = '../data/Validation/'
train_csv_root = '../data/REFERENCE-v3-training.csv'
valid_csv_root = '../data/REFERENCE-v3-validation.csv'


def vis_data_distribution(data_path, data_path_list, stage_type):
    name_list = []
    sfreq_list = []
    length_list = []
    for file in data_path_list:
        f = open(os.path.join(data_path, file))
        z = f.readline().split()
        name, sfreq, length = str(z[0]), int(z[2]), int(z[3])
        name_list.append(name)
        sfreq_list.append(sfreq)
        length_list.append(length)

    result_df = pd.DataFrame({'ID': name_list, 'sfreq': sfreq_list, 'length': length_list})
    result_df.to_csv(stage_type + '_meta.csv', index=False)

    # Show different length case num
    results_length_list = result_df['length'].tolist()
    result = Counter(results_length_list)
    x_list = sorted(result.keys())
    y_list = [result[y] for y in x_list]
    x_list = list(range(len(y_list)))

    plt.figure()
    sns.lineplot(x=x_list, y=y_list, ci=None)
    plt.xlabel = []
    plt.title('{}set Case Num (Length)'.format(stage_type))
    plt.savefig('{}set_case_num_length.png'.format(stage_type), dpi=300)
    plt.close()


def vis_data_num(data_df, stage_type):
    # Show different length case num
    class_names = ['N', 'A', 'O', '~']
    class_counter = [data_df[data_df['Label'].isin([x])].shape[0] for x in class_names]
    plt.figure()
    sns.barplot(x=class_names, y=class_counter)
    plt.title('{}set Case Num (Class)'.format(stage_type))
    plt.savefig('{}set_case_num_class.png'.format(stage_type), dpi=300)
    plt.close()


def main():
    # Get overlap
    train_data_list = sorted(glob.glob(train_data_root + "*.hea"))
    train_data_list = [x.replace(train_data_root, '') for x in train_data_list]
    valid_data_list = sorted(glob.glob(valid_data_root + "*.hea"))
    valid_data_list = [x.replace(valid_data_root, '') for x in valid_data_list]
    train_list_no_overlap = list(set(train_data_list).difference(set(valid_data_list)))
    print('train data num: {}, valid data num: {}, train data no overlap num: {}'.format(len(train_data_list),
                                                                                         len(valid_data_list),
                                                                                         len(train_list_no_overlap)))

    # Visualization training data
    vis_data_distribution(train_data_root, train_list_no_overlap, 'Train')
    vis_data_distribution(valid_data_root, valid_data_list, 'Valid')

    # Show each case num
    train_df = pd.read_csv(train_csv_root)
    train_df.columns = ['ID', 'Label']
    train_list_no_overlap = [x.replace('.hea', '') for x in train_list_no_overlap]
    train_df = train_df[train_df['ID'].isin(train_list_no_overlap)]
    train_df.to_csv('../data/REFERENCE-v3-training-nodup.csv', index=False)
    valid_df = pd.read_csv(valid_csv_root)
    valid_df.columns = ['ID', 'Label']
    vis_data_num(train_df, 'Train')
    vis_data_num(valid_df, 'Valid')
    valid_df.to_csv('../data/REFERENCE-v3-validation.csv', index=False)


if __name__ == "__main__":
    random.seed(random_seed)
    np.random.seed(random_seed)
    main()

