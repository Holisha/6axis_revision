import pandas as pd
import torch
from torch.utils.data import Dataset
import os
from glob import glob
from utils import argument_setting
from sklearn import preprocessing


class AxisDataSet(Dataset):
    def __init__(self, path, target_path=argument_setting().target_path):
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
        # csv to tensor
        csv_file = pd.read_csv(self.csv_list[idx], header=None)

        data = csv_file.iloc[:, :-1].to_numpy()
        
        data = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        
        data = torch.from_numpy(data).unsqueeze(0).float()
        index = self.stroke_type[idx]

        return data, self.target[index]

if __name__ == '__main__':
    AxisDataSet('./train')