import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from glob import glob
from sklearn.preprocessing import MinMaxScaler


#TODO: determine test path or add axis test set
#TODO: out2csv with same data
class AxisDataSet(Dataset):
    """
    file location: PATH/task/word directory/stroke_num/*.csv
        ex: PATH/train/0042/05/0042_01.csv

    file format: (word dir)_(stroke num).csv
        ex: 0042_01.csv
    """
    def __init__(self, path: str, target_path: str):
        """
        specific the where the directory is

        using list to save all csv file path, word index, and stroke num in 'path'
            csv_list = [
                (0001_01.csv, 0001, 1),
                (0001_02.csv, 0001, 1),
                 ...]
        
        Save target data as tensor and save in dictionary
            target = {
                '0001': [tensor_1 , tensor_2,...]
                '0002': [...],
            }

        Args:
            path (str): PATH/train/ or PATH/test/
            target_path (str): PATH/target/
        """
        self.csv_list = []

        self.target = {}

        # list all word directory name
        for word_dir in sorted(os.listdir(path)):

            # store stroke path
            stroke_path = os.path.join(path, word_dir)

            # list all stroke number and index
            for stroke_idx, stroke_num in enumerate(sorted(os.listdir(stroke_path))):

                # store num of stroke path
                file_path = os.path.join(stroke_path, stroke_num, '*.csv')

                # list all csv file in stroke num
                for csv_path in glob(file_path):

                    # store tuple (csv_path, word directory name, idx of stroke)
                    self.csv_list.append(
                        (csv_path, word_dir, stroke_idx)
                    )
        
        # store target word directory name
        for word_dir in sorted(os.listdir(target_path)):
            
            # store list to control stroke_num
            self.target[word_dir] = []

            # list all csv file in target data
            for csv_path in sorted(glob(os.path.join(target_path, word_dir, '**/*.csv'))):
                
                # get csv file
                tmp = pd.read_csv(csv_path, header=None)

                # last column is no longer required
                data = tmp.iloc[:, :-1].to_numpy()

                # transform data from numpy.ndarray to torch.FloatTensor
                data = torch.from_numpy(data).unsqueeze(0).float()
                
                self.target[word_dir].append(data)

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        """
        csv list = (csv_path, directory name, stroke_num)

        return:
            input data
            target data
        """
        # csv to tensor
        csv_file = pd.read_csv(self.csv_list[idx][0], header=None)

        data = csv_file.iloc[:, :-1].to_numpy()
        
        # data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        
        # regard 2d array to gray scale image format (*, 6) -> (1, *, 6)
        data = torch.from_numpy(data).unsqueeze(0).float()

        # store other infomation just for readability
        word_dir = self.csv_list[idx][1]
        index = self.csv_list[idx][2]

        return data, self.target[word_dir][index], index+1


def cross_validation(train_set, mode='hold', **kwargs):
    """split dataset into train and valid dataset

    hold out:
        A key word argument 'p' to control hold probability

    Args:
        train_set (torch.utils.data.Dataset): dataset to be split
        mode (str, optional): control the valid method. Defaults to 'hold'.

    Returns:
        train, valid: return train and valid sampler
    """

    train_len = len(train_set)

    # get shuffled indices
    indices = np.random.permutation(range(train_len))

    # hold out validation
    if mode == 'hold':
        p = kwargs['p']
        split_idx = int(train_len * p)

    # k fold
    elif mode == 'fold':
        k = kwargs['k']
        return NotImplemented

    train_idx, valid_idx = indices[:split_idx], indices[split_idx:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    return train_sampler, valid_sampler
