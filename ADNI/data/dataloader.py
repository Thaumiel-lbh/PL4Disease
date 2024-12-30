# coding=utf-8
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import random
import xlrd
from torchvision import transforms


########################### Dataset Class ###########################
class ExtractorData(Dataset):
    def __init__(self, data_index, data_root, mode='train'):
        self.data_index = data_index
        self.root = data_root
        self.mode = mode
        # All Data
        workbook = xlrd.open_workbook(self.data_index)
        if self.mode == 'train':
            worksheet = workbook.sheet_by_name('train')
        else:
            worksheet = workbook.sheet_by_name('test')
        self.samples = []
        self.labels = []
        for i in range(worksheet.nrows):
            self.samples.append(worksheet.cell_value(i, 5).split('/')[-1])
            self.labels.append(float(worksheet.cell_value(i, 4)))

    def onehot(self, label):
        assert label == 0.5 or label == 1 or label == 2, 'label error!' + str(label)
        tmp = torch.zeros(3)
        if label == 0.5:
            tmp[0] = 1
        elif label == 1:
            tmp[1] = 1
        elif label == 2:
            tmp[2] = 1
        return tmp

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root, self.samples[idx]))
        data = np.expand_dims(data, 0)
        label = self.onehot(self.labels[idx])
        return data, label

    def __len__(self):
        return len(self.samples)


class Order1Data(Dataset):
    def __init__(self, data_index, data_root, mode='train', delta_t=None):
        self.data_index = data_index
        self.root = data_root
        self.mode = mode
        # All Data
        workbook = xlrd.open_workbook(self.data_index)
        if self.mode == 'train':
            worksheet = workbook.sheet_by_name('train')
        else:
            worksheet = workbook.sheet_by_name('test')
        self.samples_t = []
        self.time_t = []
        self.samples_T = []
        self.time_T = []
        self.labels = []
        for i in range(worksheet.nrows):
            if delta_t:
                if (int(worksheet.cell_value(i, 10)) - int(worksheet.cell_value(i, 3)))%100 // 6 == delta_t:
                    self.samples_t.append(worksheet.cell_value(i, 5).split('/')[-1])
                    self.time_t.append(int(worksheet.cell_value(i, 3)))

                    self.samples_T.append(worksheet.cell_value(i, 12).split('/')[-1])
                    self.time_T.append(int(worksheet.cell_value(i, 10)))
                    self.labels.append(float(worksheet.cell_value(i, 11)))
            else:
                self.samples_t.append(worksheet.cell_value(i, 5).split('/')[-1])
                self.time_t.append(int(worksheet.cell_value(i, 3)))

                self.samples_T.append(worksheet.cell_value(i, 12).split('/')[-1])
                self.time_T.append(int(worksheet.cell_value(i, 10)))
                self.labels.append(float(worksheet.cell_value(i, 11)))

    def onehot(self, label):
        tmp = torch.zeros(3)
        if label == 0.5:
            tmp[0] = 1
        elif label == 1:
            tmp[1] = 1
        elif label >= 2:
            tmp[2] = 1
        return tmp

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        data_t = np.load(os.path.join(self.root, self.samples_t[idx]))
        data_t = np.expand_dims(data_t, 0)
        time_t = self.time_t[idx] % 100 // 6

        data_T = np.load(os.path.join(self.root, self.samples_T[idx]))
        data_T = np.expand_dims(data_T, 0)
        time_T = self.time_T[idx] % 100 // 6
        label = self.onehot(self.labels[idx])

        return data_t, data_T, label, time_t, time_T

    def __len__(self):
        return len(self.samples_t)


class Order2Data(Dataset):
    def __init__(self, data_index, data_root, mode='train'):
        self.data_index = data_index
        self.root = data_root
        self.mode = mode
        # All Data
        workbook = xlrd.open_workbook(self.data_index)
        if self.mode == 'train':
            worksheet = workbook.sheet_by_name('train')
        else:
            worksheet = workbook.sheet_by_name('test')

        self.time1 = []
        self.labels_t1 = []
        self.samples1 = []

        self.time2 = []
        self.labels_t2 = []
        self.samples2 = []

        self.timeT = []
        self.labels_T = []
        self.samples_T = []

        for i in range(worksheet.nrows):
            self.time1.append(int(worksheet.cell_value(i, 3)))
            self.labels_t1.append(float(worksheet.cell_value(i, 4)))
            self.samples1.append(worksheet.cell_value(i, 5).split('/')[-1])

            self.time2.append(int(worksheet.cell_value(i, 10)))
            self.labels_t2.append(float(worksheet.cell_value(i, 11)))
            self.samples2.append(worksheet.cell_value(i, 12).split('/')[-1])

            self.timeT.append(int(worksheet.cell_value(i, 17)))
            self.labels_T.append(float(worksheet.cell_value(i, 18)))
            self.samples_T.append(worksheet.cell_value(i, 19).split('/')[-1])

    def onehot(self, label):
        tmp = torch.zeros(3)
        if label == 0.5:
            tmp[0] = 1
        elif label == 1:
            tmp[1] = 1
        elif label >= 2:
            tmp[2] = 1
        return tmp

    def get_labels(self):
        return self.labels_T

    def __getitem__(self, idx):
        data_t1 = np.load(os.path.join(self.root, self.samples1[idx]))
        data_t1 = np.expand_dims(data_t1, 0)
        time_t1 = self.time1[idx] % 100 // 6

        data_t2 = np.load(os.path.join(self.root, self.samples2[idx]))
        data_t2 = np.expand_dims(data_t2, 0)
        time_t2 = self.time2[idx] % 100 // 6

        data_T = np.load(os.path.join(self.root, self.samples_T[idx]))
        data_T = np.expand_dims(data_T, 0)
        time_T = self.timeT[idx] % 100 // 6

        label_t1 = self.onehot(self.labels_t1[idx])
        label_t2 = self.onehot(self.labels_t2[idx])
        label_T = self.onehot(self.labels_T[idx])

        return data_t1, data_t2, data_T, label_t1, label_t2, label_T, time_t1, time_t2, time_T

    def __len__(self):
        return len(self.samples1)


if __name__ == '__main__':
    test_dataset = Order1Data(data_index='./dataset_order1.xls',
                              data_root='/root/autodl-tmp/ImageAll_V1_part1/npy_processed_data/',
                              delta_t=2,
                              mode='test')
    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=1)
    for data in test_loader:
        print(data[3])



