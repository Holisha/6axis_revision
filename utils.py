import os
import csv
import numpy as np
import pandas as pd
from glob import glob
from argparse import ArgumentParser


def argument_setting():
    r"""
    return arguments
    """
    parser = ArgumentParser()

    # data pre-processing
    parser.add_argument('--stroke-length', type=int, default=59,
                        help='control the stroke length (default: 59)')
    parser.add_argument('--check_interval', type=int, default=100,
                        help='setting output a csv file every epoch of interval (default: 100)')

    # model setting
    parser.add_argument('--light', action='store_true', default=False,
                        help='train by pytorch-lightning model (default: False)')
    parser.add_argument('--train-path', type=str, default='./train',
                        help='training dataset path (default: ./train)')
    parser.add_argument('--test-path', type=str, default='./test',
                        help='test dataset path (default: ./test)')
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

    # logger setting
    parser.add_argument('--log-path', type=str, default='./logs/FSRCNN',
                        help='set the logger path of pytorch model (default: ./logs/FSRCNN)')
    parser.add_argument('--light-path', type=str, default='./logs/FSRCNN_light',
                        help='set the logger path of pytorch-lightning model (default: ./logs/FSRCNN_light)')
    
    # save setting
    parser.add_argument('--save-path', type=str, default='./output',
                        help='set the output file (csv or txt) path (default: ./output)')

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


def out2csv(inputs, file_string, stroke_length):
    """
    store input to csv file.

    input: tensor data, with cuda device and size = [batch 1 STROKE_LENGTH 6]
    file_string: string, filename

    no output
    """
    output = np.squeeze(inputs.cpu().detach().numpy())
    table = output[0]

    # with open('output/' +  file_string + '.csv', 'w', newline='') as csvfile:
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
