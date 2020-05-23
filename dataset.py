import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import os
from glob import glob


class AxisDataSet(Dataset):
    def __init__(self, path, target_path='./target'):
        self.target = []
        self.csv_list = []
        self.stroke_type = []

        for idx, dir_name in enumerate(sorted(os.listdir(path))):
            for csv_name in glob(os.path.join(path, dir_name, '*.csv')):
                self.csv_list.append(csv_name)
                self.stroke_type.append(idx)

        for csv_name in sorted(glob(os.path.join(target_path, '*.csv'))):
            tmp = pd.read_csv(csv_name, header=None)
            data = tmp.iloc[:, :-1].to_numpy()
            data = torch.from_numpy(data).unsqueeze(0).float()
            self.target.append(data)

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        """
        return:
            input data
            target data
        """
        # csv to tensor
        csv_file = pd.read_csv(self.csv_list[idx], header=None)

        data = csv_file.iloc[:, :-1].to_numpy()
        data = torch.from_numpy(data).unsqueeze(0).float()
        index = self.stroke_type[idx]

        return data, self.target[index]


def cross_validation(train_set, p=0.8):
    """
    hold out cross validation
    """
    train_len = len(train_set)

    # get shuffled indices
    indices = np.random.permutation(range(train_len))
    split_idx = int(train_len * p)

    train_idx, valid_idx = indices[:split_idx], indices[split_idx:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler
