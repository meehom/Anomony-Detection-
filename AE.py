# -*- coding: utf-8 -*-
"""
 @Time    : 2022/4/20 14:33
 @Author  : meehom
 @Email   : meehomliao@163.com
 @File    : AE.py
 @Software: PyCharm
"""
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score
from torch.utils.data import Dataset, DataLoader
# --------train-------
import os
import torch
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


class autoencoder(nn.Module):
    def __init__(self ,input_size):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 12),
            nn.ReLU(True),
            nn.Linear(12, 3))
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 32),
            nn.ReLU(True),
            nn.Linear(32, 128),
            nn.ReLU(True),
            nn.Linear(128, input_size))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class kddcup99_Dataset(Dataset):
    def __init__(self, data , mode='train', ratio=0.8):
        self.mode = mode
        data = torch.from_numpy(data)

        normal_data = data[data[:, -1] == 0]
        abnormal_data = data[data[:, -1] == 1]

        train_normal_mark = int(normal_data.shape[0] * ratio)
        train_abnormal_mark = int(abnormal_data.shape[0] * ratio)

        train_normal_data = normal_data[:train_normal_mark, :]
        train_abnormal_data = abnormal_data[:train_abnormal_mark, :]
        self.train_data = np.concatenate((train_normal_data, train_abnormal_data), axis=0)
        np.random.shuffle(self.train_data)

        test_normal_data = normal_data[train_normal_mark:, :]
        test_abnormal_data = abnormal_data[train_abnormal_mark:, :]
        self.test_data = np.concatenate((test_normal_data, test_abnormal_data), axis=0)
        np.random.shuffle(self.test_data)

    def __len__(self):
        if self.mode == 'train':
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]

    def __getitem__(self, index):
        if self.mode == 'train':
            return self.train_data[index, :-1], self.train_data[index, -1]
        else:
            return self.test_data[index, :-1], self.test_data[index, -1]


def get_loader(data, mode='train'):
    """Build and return data loader."""

    dataset = kddcup99_Dataset(data, mode)

    shuffle = True if mode == 'train' else False

    data_loader = DataLoader(dataset=dataset,
                             batch_size=128,
                             shuffle=shuffle)

    return data_loader, len(dataset)


data = np.load('kdd_data_10_percent_preprocessed.npy', allow_pickle=True)
print(data.shape)
data = data.astype(np.float64)
data = data[:128000, :]




def reconstruct_error(x, x_hat):  # 重构误差
    e = torch.tensor(0.0)
    for i in range(x.shape[0]):
        e += torch.dist(x[i], x_hat[i])
    return e

def train():
    train_loader, _ = get_loader(data=data, mode='train')
    model = autoencoder(data.shape[1] - 1)
    model = model.double()

    optim = torch.optim.Adam(model.parameters(), 0.001, amsgrad=True)
    scheduler = MultiStepLR(optim, [5, 8], 0.1)
    #     iter_wrapper = lambda x: tqdm(x, total=len(train_loader))

    loss_total = 0

    for epoch in range(30):
        for i, (input_data, labels) in enumerate(train_loader):
            model.train()
            optim.zero_grad()

            output = model(input_data)


            loss = reconstruct_error(input_data, output)

            loss_total += loss.item()

            #             model.zero_grad()

            loss.backward()
            #             torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()

            if (i + 1) % 100 == 0:
                print("batch:{}, loss: {}".format(i+1, loss))
            if epoch == 29:
                torch.save(model.state_dict(),
                          '{}_dagmm.pth'.format(epoch))

        scheduler.step()
        print("epoch:{}, loss_toral: {}".format(epoch, loss_total))
        loss_total = 0


def load_model():
    model = autoencoder(data.shape[1] - 1)
    try:
        model.load_state_dict(torch.load('./29_dagmm.pth'))
        print('success to load model')
    except Exception as e:
        print('Failed to load model: %s' % (e))
        exit(1)

    return model


def compute_threshold(model, train_loader, len_data):

    loss = []
    with torch.no_grad():
        for x, y in train_loader:
            model = model.double()
            output = model(x)

            e = torch.tensor(0.0)
            for i in range(output.shape[0]):
                e = torch.dist(x[i], output[i])
                loss += [e]


    threshold = np.percentile(loss, 98)
    print('threshold: %.4f' % (threshold))

    return threshold

def test():
    '''test
    :return:
    '''
    model = load_model()
    train_loader, len_data = get_loader(data=data, mode='train')
    test_loader, data_test = get_loader(data=data, mode='test')
    threshold = compute_threshold(model, train_loader, len_data)

    step = 0
    scores = np.zeros(shape=(data_test,2))
    with torch.no_grad():
        for x, y in test_loader:

            outputs = model(x)
            for i in range(outputs.shape[0]):
                e = torch.dist(x[i], outputs[i])
                if e > threshold:
                    scores[step] = [0, y[i]]
                    step = step + 1
                else:
                    scores[step] = [1, y[i]]
                    step = step + 1
    accuracy = accuracy_score(scores[:, 0], scores[:, 1])
    precision, recall, fscore, support = precision_recall_fscore_support(scores[:, 0], scores[:, 1], average='binary')
    print('Accuracy: %.4f  Precision: %.4f  Recall: %.4f  F-score: %.4f' % (accuracy, precision, recall, fscore))

if __name__ == '__main__':
    # train()
    test()








