import numpy as np
import pandas as pd
import csv
import os
from argparse import ArgumentParser
from glob import glob
import torch

def argument_setting():
    r"""
    return arguments
    """
    parser = ArgumentParser()

    # data pre-processing
    parser.add_argument('--stroke-length', type=int, default=150,
                        help='control the stroke length (default: 150)')
    parser.add_argument('--check_interval', type=int, default=100,
                        help='setting output a csv file every epoch of interval (default: 100)')

    # model setting
    parser.add_argument('--light', action='store_true', default=False,
                        help='train by pytorch-lightning model (default: False)')
    parser.add_argument('--train-path', type=str, default='./dataset_436/train',
                        help='training dataset path (default: ./dataset_436/train)')
    parser.add_argument('--test-path', type=str, default='./dataset_436/test',
                        help='test dataset path (default: ./dataset_436/test)')
    parser.add_argument('--target-path', type=str, default='./dataset_436/target',
                        help='target dataset path (default: ./dataset_436/target)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='set the batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='set the number of processes to run (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='set the learning rate (default: 1e-3)')
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='set the epochs (default: 50)')
    parser.add_argument('--holdout-p', type=float, default=0.8,
                        help='set hold out CV probability (default: 0.8)')

    # logger setting
    parser.add_argument('--log-path', type=str, default='./logs/FSRCNN',
                        help='set the logger path of pytorch model (default: ./logs/FSRCNN)')
    parser.add_argument('--light-path', type=str, default='./logs/FSRCNN_light',
                        help='set the logger path of pytorch-lightning model (default: ./logs/FSRCNN_light)')
    
    # save setting
    parser.add_argument('--save-path', type=str, default='./output',
                        help='set the output file (csv or txt) path (default: ./output)')
    # output become input                    
    parser.add_argument('--retrain-epochs', type=int, default=5,
                        help='retrain the output file')

    return parser.parse_args()


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


def out2csv(inputs, file_string, stroke_length, index=0):
    """Store input to csv file

    Arguments:
        inputs {tensor} -- with cuda device and size = [batch, 1, STROKE_LENGTH, 6]
        file_string {string} -- filename
        stroke_length {int} -- length of each stroke

    Keyword Arguments:
        index {int} -- index of stroke (default: {0})
    """
    output = np.squeeze(inputs.cpu().detach().numpy())
    table = output[index]

    with open(f'output/{file_string}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(stroke_length):
            row = [] * 7
            row[1:6] = table[i][:]
            row.append('stroke' + str(1))
            writer.writerow(row)

def save_final_predict_and_new_dataset(inputs, file_string, args,store_data_cnt):
    output = np.squeeze(inputs.cpu().detach().numpy())
    
    for index in range(args.batch_size):
        table = output[index]
        with open(f'{file_string}_{store_data_cnt+index}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(args.stroke_length):
                row = [] * 7
                row[1:6] = table[i][:]
                row.append('stroke' + str(1))
                writer.writerow(row)

def csv2txt(path='./output'):
    """Convert all CSV files to TXT files.

    Keyword Arguments:
        path {str} -- Path to output directory (default: {'./output'})
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
