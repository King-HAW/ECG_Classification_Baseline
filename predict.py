# -*- coding: utf-8 -*-

from __future__ import print_function  # do not delete this line if you want to save your log file.
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.optim import lr_scheduler
from torch.utils import model_zoo
from model.resnet1D import resnet18, resnet34, resnet50
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn as nn
import shutil
import torch
import os
import math
import random
import pandas as pd
import numpy as np
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

random_seed = 2333
model_save_path = './checkpoints/'
train_data_root = './preprocess/trainset.npy'
infer_data_root = './preprocess/validset.npy'
train_csv_path = './preprocess/training-nodup.csv'
infer_csv_path = './preprocess/infer.csv'


n_folds = 5
batchsize = 128
num_workers = 7
model_name = 'resnet18'
lr = 2e-4
lr_reduce_epoch = 5
epochs = 12
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def seed_reproducer(seed=2333):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def open_log(model_name, outputs_path, name='train'):
    # open the log file
    log_savepath = os.path.join(outputs_path, 'logs', name)
    if not os.path.exists(log_savepath):
        os.makedirs(log_savepath)
    log_name = model_name
    if os.path.isfile(os.path.join(log_savepath, '{}.log'.format(log_name))):
        os.remove(os.path.join(log_savepath, '{}.log'.format(log_name)))
    initLogging(os.path.join(log_savepath, '{}.log'.format(log_name)))


def initLogging(logFilename):
    """Init for logging
    """
    logger = logging.getLogger('')

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s-%(levelname)s] %(message)s',
            datefmt='%y-%m-%d %H:%M:%S',
            filename=logFilename,
            filemode='w')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(asctime)s-%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1, sum_flag=True):
        if sum_flag:
            self.val = val
            self.sum += val * n
        else:
            self.val = val / n
            self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, name, data_file, fold_file, folds):
        self.name = name
        self.data_file = data_file

        # Load DataFrame
        self.path_df = fold_file

        if folds:
            self.path_df = self.path_df[self.path_df['Fold'].isin(folds)]
            self.path_df = self.path_df.reset_index(drop=True)

        # self.path_df = self.path_df.sample(100)

    def __len__(self):
        return len(self.path_df)

    def __getitem__(self, index):
        row = self.path_df.iloc[index]
        data = self.data_file[row['Idex'], :]
        data = np.expand_dims(data, axis=0)

        assert len(data.shape) == 2, 'Data Shape ERROR'

        data = torch.from_numpy(data).float()

        if self.name != 'infer':
            label = row['Label']
            label = np.array(label)
            label = torch.from_numpy(label).long()
            return data, label, row['ID']

        else:
            return data, row['ID']


def load_data_infer(data_file, infer_fold_file, modelbatchsize):
    infer_dataset = CustomDataset('infer', data_file, infer_fold_file, None)
    infer_dataloader = torch.utils.data.DataLoader(infer_dataset, batch_size=modelbatchsize, shuffle=False,
                                                    num_workers=0, drop_last=False)
    return infer_dataloader


def model_fn(model_name, num_classes):
    if model_name.startswith('resnet'):
        model = eval(model_name)(num_classes=num_classes)
        return model

    else:
        raise Exception('Model Not Define!!!')


def infer_net(infer_dataloader, model):
    model.eval()

    ids_all = []
    outputs_all = []

    with torch.no_grad():
        for i, (data, name) in enumerate(infer_dataloader):
            data = data.to(device)
            preds = model(data)
            ids_all.extend(name)
            outputs_all.extend(torch.softmax(preds, dim=1).cpu().numpy())

    result = {
        'ids': ids_all,
        'outputs': np.array(outputs_all),
    }
    return result


def save_model(model, model_name, fold, epoch, val_loss, _best=None, best=0.0):
    savepath = os.path.join(model_save_path, model_name, 'fold' + str(fold))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    file_name = os.path.join(savepath, "{}_epoch_{:0>4}".format(model_name, epoch) + '.pth')
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
    }, file_name)
    remove_flag = False
    if _best:
        best_name = os.path.join(savepath, "{}_best_{}".format(model_name, _best) + '.pth')
        shutil.copy(file_name, best_name)
        remove_flag = True
        file = open(os.path.join(savepath, "{}_best_{}".format(model_name, _best) + '.txt'), 'w')
        file.write('arch: {}'.format(model_name) + '\n')
        file.write('epoch: {}'.format(epoch) + '\n')
        file.write('best {}: {}'.format(_best, best) + '\n')
        file.close()
    if remove_flag:
        os.remove(file_name)


def load_model(path, model):
    # remap everthing onto CPU
    state = torch.load(str(path), map_location=lambda storage, location: storage)
    model.load_state_dict(state['model'])
    model.to(device)
    return model


def avg_predictions(results):
    outputs_all = np.array([result['outputs'] for result in results])
    outputs = outputs_all.mean(axis=0)
    return {
        'ids': results[0]['ids'],
        'outputs': outputs,
    }


def main():
    # check_data_exist_and_get_data()
    infer_df = pd.read_csv(infer_csv_path)
    infer_data = np.load(infer_data_root)
    idex_fold = dict()
    all_names = infer_df['ID'].tolist()
    for idx, name in enumerate(all_names):
        idex_fold[name] = idx
    infer_df['Idex'] = infer_df['ID'].map(idex_fold)

    num_classes = len(['N', 'A', 'O', '~'])
    results_fold = []

    for fold in range(n_folds):
        open_log(model_name, os.path.join(model_save_path, model_name), name='infer')
        infer_dataloader = load_data_infer(infer_data, infer_df, batchsize)

        ### Valid valid part
        logging.info('Model: {} Fold: {} Training'.format(model_name, fold))
        model = model_fn(model_name, num_classes=num_classes)
        model = load_model(os.path.join(model_save_path, model_name, 'fold' + str(fold), "{}_best_{}".format(model_name, 'acc.pth')), model)
        result = infer_net(infer_dataloader, model.to(device))
        results_fold.append(result)

    final_result = avg_predictions(results_fold)

    result_df = pd.read_csv(infer_csv_path)
    IDs = {}
    for id, outputs in zip(final_result['ids'], final_result['outputs']):
        IDs[id] = np.argmax(outputs)
    result_df['Preds'] = result_df['ID'].map(IDs)
    y_pred = result_df['Preds'].tolist()
    y_true = result_df['Label'].tolist()

    print(classification_report(y_pred=y_pred, y_true=y_true))


if __name__ == "__main__":
    seed_reproducer(random_seed)
    main()
