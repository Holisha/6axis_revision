import os
import csv
import json
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from argparse import ArgumentParser
from glob import glob
from typing import Union


def stroke_statistics(path='6d/', mode='max'):
    r""" 
    parameter:
    path: path of the 6d axis csv file
    mode: output statstic. i.e. mode:max means return the maximum stroke number
    count the mean, maximum, minimum of the stroke statistics
    output: use parameter 
    """
    max_cnt = np.int64(0)
    mean_cnt = np.int64(0)
    min_cnt = np.int64(100)

    for dir_name in os.listdir(path):
        df = pd.read_csv(
            os.path.join(path, dir_name, dir_name + '.csv'),
            header=None
        )

        tmp = df.groupby(6)[0].count()

        if tmp.max() > max_cnt:
            max_cnt = tmp.max()
        
        if tmp.mean() > mean_cnt:
            mean_cnt = tmp.mean()

        if tmp.min() < min_cnt:
            min_cnt = tmp.min()
    
    print(f'max stroke number:{max_cnt}\nmean stroke number:{mean_cnt}\nmin stroke number:{min_cnt}')

    return {
        'max' : max_cnt,
        'mean': mean_cnt,
        'min' : min_cnt
    }.get(mode, 'error')


def writer_builder(log_root, load=False):
    """Build writer acording to exist or new logs

    Args:
        log_root (str): logs root
        load (bool, optional): load existed Tensorboard. Defaults to False.
        
    Returns:
        SummaryWriter: tensorboard
    """

    from torch.utils.tensorboard import SummaryWriter

    version = os.listdir(log_root)

    # make sure logs directories exist
    if not os.path.exists('./logs'):
        os.mkdir('logs')

    if not os.path.exists(log_root):
        os.mkdir(log_root)

    # load exist logs
    if version and load == True:
        log_path = os.path.join(log_root, version[-1])

    # create new log directory indexed by exist directories
    else:
        log_path = os.path.join(log_root, f'version_{len(version)}')
        os.mkdir(
            log_path
        )
    
    return SummaryWriter(log_path)


def model_builder(model_name, *args, **kwargs):
    """choose which model would be training

    Args:
        model_name (str): FSRCNN, DDBPN 

    Returns:
        model(torch.nn.module): instantiate model
    """
    from model import FSRCNN, DDBPN

    # class object, yet instantiate
    model = {
        'fsrcnn': FSRCNN,    # scale_factor, num_channels=1, d=56, s=12, m=4
        'ddbpn': DDBPN,      # scale_factor, num_channels=1, stages=7, n0=256, nr=64
    }.get(model_name.lower())

    return model(*args, **kwargs)


def model_config(args, save: Union[str, bool]=False):
    """record model configuration

    if save is path, save to the path
    if save is True, save in current directory
    Args:
        args (Argparse object): Model setting
        save (Union[str, bool], optional): save as json file or just print to stdout. Defaults to False.
    """
    print('\n####### model arguments #######\n')
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    print('\n####### model arguments #######\n')

    # save config as .json file        
    if save is True:
        config = open('config.json', 'w')
        json.dump(vars(args), config, indent=4)
        config.close()
    # save config as .json file to path
    elif type(save) is str:
        config = open(
            os.path.join(save,'config.json'), 'w')
        json.dump(vars(args), config, indent=4)
        config.close()


def optimizer_builder(optim_name: str):
    """build optimizer

    Args:
        optim_name (str): choose which optimizer for training
            'adam': optim.Adam
            'sgd': optim.SGD
            'ranger': Ranger
            'rangerva': RangerVA

    Returns:
        optimizer class, yet instantiate
    """
    from model import Ranger, RangerVA
    
    return {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'ranger': Ranger,
        'rangerva': RangerVA,
    }.get(optim_name, optim.Adam)

##### training #####


def inverse_scaler_transform(pred, target):
    """Inverse pred from range (0, 1) to target range.
    
    pred_inverse = (pred * (max - min)) + min
    
    ---
    Arguments:
        pred {torch.tensor} -- Tensor which is inversed from range (0, 1) to target range.
        target {torch.tensor} -- Inversion reference range.
    ---
    Returns:
        torch.tensor -- pred after inversed.
    """

    # max and min shape is [batch_size, 1, 1, 6]
    max = torch.max(target, 2, keepdim = True)[0]
    min = torch.min(target, 2, keepdim = True)[0]
    
    # pred_inverse = (pred * (max - min)) + min
    pred_inverse = torch.add(torch.mul(pred, torch.sub(max, min)), min)

    return pred_inverse


def out2csv(inputs, file_string, stroke_length):
    """
    store input to csv file.

    input: tensor data, with cuda device and size = [batch 1 STROKE_LENGTH 6]
    file_string: string, filename

    no output
    """
    output = np.squeeze(inputs.cpu().detach().numpy())
    table = output[0]

    # with open('output/' + file_string + '.csv', 'w', newline='') as csvfile:
    with open(f'output/{file_string}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(stroke_length):
            row = [] * 7
            row[1:6] = table[i][:]
            row.append('stroke' + str(1))
            writer.writerow(row)


def csv2txt(path='./output'):
    r"""
    Convert all CSV files to txt files.

    path: store the csv files (default: './output')
    """
    for csv_name in sorted(glob(os.path.join(path, '*.csv'))):

        # read csv file content
        with open(csv_name, newline='') as csv_file:
            rows = csv.reader(csv_file)
            txt_name = f'{csv_name[:-4]}.txt'

            # store in txt file
            with open(txt_name, "w") as txt_file:
                for row in rows:
                    txt_file.write("movl 0 ")

                    for j in range(len(row) - 1):
                        txt_file.write(f'{float(row[j]):0.4f} ')

                    txt_file.write("100.0000 ")
                    txt_file.write(f'{row[6]}\n')


def save_final_predict_and_new_dataset(inputs,stroke_num, file_string, args,store_data_cnt):
    output = np.squeeze(inputs.cpu().detach().numpy())
    
    for index in range(args.batch_size):
        try:
            table = output[index]
        except:
            break
        num = stroke_num[index]
        if not os.path.isdir(f'final_output/{num}'):
            # os.mkdir(f'new_train/{num}')
            os.mkdir(f'final_output/{num}')

        with open(f'{file_string}/{num}/{num}_{store_data_cnt+index}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(args.stroke_length):
                row = [] * 7
                row[1:6] = table[i][:]
                row.append(f'stroke{num}')
                writer.writerow(row)
