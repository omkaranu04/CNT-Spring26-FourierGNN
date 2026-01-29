import os
import datetime
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Dataloader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Traffic Dataset Class
class Dataset_DHFM(Dataset):
    def __init__(self, root_path, flag, seq_len, pre_len, type, train_ratio, val_ratio):
        assert flag in ['train', 'val', 'test']
        self.path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        load_data = np.load(root_path)
        data = load_data.transpose()
        if type == '1':
            mms = MinMaxScaler(feature_range=(0, 1))
            training_end = int(len(data) * self.train_ratio)
            mms.fit(data[:training_end])
            data = mms.transform(data)
        if self.flag == 'train':
            begin = 0
            end = int(len(data) * self.train_ratio)
            self.trainData = data[begin:end]
        if self.flag == 'val':
            begin = int(len(data) * self.train_ratio)
            end = int(len(data) * (self.val_ratio + self.train_ratio))
            self.valData = data[begin:end]
        if self.flag == 'test':
            begin = int(len(data) * (self.val_ratio + self.train_ratio))
            end = len(data)
            self.testData = data[begin:end]
    def __getitem__(self, idx):
        begin = idx
        end = idx + self.seq_len
        next_end = end + self.pre_len
        if self.flag == 'train':
            data = self.trainData[begin:end]
            next_data = self.trainData[end:next_end]
        if self.flag == 'val':
            data = self.valData[begin:end]
            next_data = self.valData[end:next_end]
        if self.flag == 'test':
            data = self.testData[begin:end]
            next_data = self.testData[end:next_end]
        return data, next_data
    def __len__(self):
        if self.flag == 'train':
            return len(self.trainData) - self.seq_len - self.pre_len
        if self.flag == 'val':
            return len(self.valData) - self.seq_len - self.pre_len
        if self.flag == 'test':
            return len(self.testData) - self.seq_len - self.pre_len