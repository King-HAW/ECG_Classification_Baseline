# -*- coding: utf-8 -*-

from __future__ import print_function  # do not delete this line if you want to save your log file.
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
model_name = 'resnet34'
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


def onehot_encoding(label, n_classes):
    return torch.zeros(label.size(0), n_classes).to(label.device).scatter_(1, label.view(-1, 1), 1)


def cross_entropy_loss(input, target, reduction='mean'):
    logp = torch.log_softmax(input, dim=1)
    loss = torch.sum(-logp * target, dim=1)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')


def label_smoothing_criterion(preds, targets, epsilon=0.1, reduction='mean'):
    n_classes = preds.size()[1]
    onehot = onehot_encoding(targets, n_classes).float().to(device)
    targets = onehot * (1 - epsilon) + torch.ones_like(onehot).to(device) * epsilon / n_classes
    loss = cross_entropy_loss(preds, targets, reduction)
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError('`reduction` must be one of \'none\', \'mean\', or \'sum\'.')


def get_fold_file(df_file, n_folds=5):
    # The main df file
    KF = KFold(n_splits=n_folds)
    case_fold = dict()
    idex_fold = dict()
    all_names = df_file['ID'].tolist()
    for idx, name in enumerate(all_names):
        idex_fold[name] = idx
    max_label = max(list(df_file['Label'].unique()))
    for label_idx in range(max_label + 1):
        class_df = df_file[df_file['Label'].isin([label_idx])]
        class_df_name_list = class_df['ID'].tolist()
        all_num = len(class_df_name_list)
        for fold_index, (train_index, test_index) in enumerate(KF.split(range(all_num))):
            for case_index in test_index:
                case_fold[class_df_name_list[case_index]] = int(fold_index)
    df_file['Idex'] = df_file['ID'].map(idex_fold)
    df_file['Fold'] = df_file['ID'].map(case_fold)
    return df_file


def load_data_train_valid(data_file, train_fold_file, valid_fold, modelbatchsize):
    folds = [fold for fold in range(n_folds) if valid_fold != fold]
    train_dataset = CustomDataset('train', data_file, train_fold_file, folds)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=modelbatchsize, shuffle=True,
                                                   num_workers=num_workers, drop_last=True)

    valid_dataset = CustomDataset('valid', data_file, train_fold_file, [valid_fold])
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=modelbatchsize, shuffle=False,
                                                   num_workers=0, drop_last=False)
    return train_dataloader, valid_dataloader


def model_fn(model_name, num_classes):
    if model_name.startswith('resnet'):
        model = eval(model_name)(num_classes=num_classes)
        return model

    else:
        raise Exception('Model Not Define!!!')


def train_net(train_dataloader, valid_dataloader, model, optimizer, epochs, model_name, fold):
    best_acc = 0.0
    best_index = [best_acc]
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_reduce_epoch, gamma=0.5, last_epoch=-1)

    for epoch in range(1, epochs + 1):
        train(train_dataloader, model, optimizer, epoch)
        best_index = valid_net(valid_dataloader, model, best_index, epoch, model_name, fold)
        logging.info('Valid-Cls: Best ACC update to: {:.4f}'.format(float(best_index[0])))
        scheduler.step()
    pass


def valid_net(valid_dataloader, model, best_index, epoch, model_name, fold):
    m_acc, m_loss = valid(valid_dataloader, model)
    best_acc = best_index[0]

    logging.info('Valid-Cls: Mean ACC: {:.4f}'.format(m_acc))
    # save_model(model, model_name, fold, epoch, m_loss)

    if m_acc >= best_acc:
        save_model(model, model_name, fold, epoch, m_loss, _best='acc', best=m_acc)
        best_acc = m_acc

    return [best_acc]


def valid(valid_dataloader, model):
    cls_ACCs_valid = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (data, label, name) in enumerate(valid_dataloader):
            bs = data.shape[0]
            data = data.to(device)
            label = label.to(device)
            preds = model(data)
            valid_loss = label_smoothing_criterion(preds, label)
            if i == 0:
                all_valid_loss = valid_loss
            else:
                all_valid_loss += valid_loss

            preds_raw = torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().detach().numpy()
            label_raw = label.cpu().detach().numpy()

            acc = accuracy_score(label_raw, preds_raw)
            cls_ACCs_valid.update(acc, bs)

    all_valid_loss = all_valid_loss / i

    return cls_ACCs_valid.avg, all_valid_loss


def train(train_dataloader, model, optimizer, epoch):
    cls_losses = AverageMeter()
    cls_ACCs_train = AverageMeter()
    model.train()

    for i, (data, label, name) in enumerate(train_dataloader):
        bs = data.shape[0]

        data = data.to(device)
        label = label.to(device)
        preds = model(data)
        loss = label_smoothing_criterion(preds, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds_raw = torch.argmax(torch.softmax(preds, dim=1), dim=1).cpu().detach().numpy()
        label_raw = label.cpu().detach().numpy()
        acc = accuracy_score(label_raw, preds_raw)

        cls_ACCs_train.update(acc, bs)
        cls_losses.update(loss.item(), bs)
        lr_current = optimizer.param_groups[0]['lr']

        if i % 10 == 0:
            logging.info('Epoch: [{}][{}/{}]\t'
                         'lr: {lr:.5f} '
                         'Loss: {loss.val:.4f} ({loss.avg:.4f}) '
                         'Cls_ACC: {cls_acc.val:.4f} ({cls_acc.avg:.4f}) '.format(
                epoch, i, len(train_dataloader), lr=lr_current, loss=cls_losses, cls_acc=cls_ACCs_train))


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
    train_valid_df = pd.read_csv(train_csv_path)
    train_data = np.load(train_data_root)
    train_valid_df = get_fold_file(train_valid_df, n_folds)

    num_classes = max(list(train_valid_df['Label'].unique())) + 1

    for fold in range(n_folds):
        open_log(model_name, os.path.join(model_save_path, model_name))
        train_dataloader, valid_dataloader = load_data_train_valid(train_data, train_valid_df, fold, batchsize)

        ### Train valid part
        logging.info('Model: {} Fold: {} Training'.format(model_name, fold))
        model = model_fn(model_name, num_classes=num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr)
        train_net(train_dataloader, valid_dataloader, model.to(device), optimizer, epochs, model_name, fold)


if __name__ == "__main__":
    seed_reproducer(random_seed)
    main()

