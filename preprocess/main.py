import os
import numpy as np
import pandas as pd
from glob import glob
from utils import argument_setting

def addnoise(target_data, noise):
    """add noise in range [noise[0], noise[1]]

    Args:
        target_data (pandas.Dataframe): the data to be added noise
        noise ([float, float]): the range of noise to add

    Returns:
        pandas.Dataframe: the data after added noise
    """
    random_data = pd.DataFrame(
        np.random.uniform(noise[0], noise[1], size=(target_data.shape[0], target_data.shape[1] - 1))
    )
    random_data[6] = [""] * random_data.shape[0]
    data = target_data.reset_index(drop=True).add(random_data)
    
    return data

def add_last_line(data, stroke_len):
    """add last line to len of stroke_len.

    Args:
        data (pd.dataframe): data to be added last line
        stroke_len (int): the len of each stroke

    Returns:
        pd.dataframe: the data after added last line
    """
    add_last_line_num = stroke_len - data.shape[0]
    data = data.append(data.iloc[[-1] * add_last_line_num], ignore_index=True)
    
    return data


def csv_parser(char_num, txt_name, target_path, train_path, args):
    """split txt file to csv file by stroke

    Args:
        char_num (string): the number of the character
        txt_name (string): the txt file name of the character
        args : argument
    """

    stroke_len = args.stroke_len

    if not os.path.exists(f'{target_path}/{char_num}'):
        os.mkdir(f'{target_path}/{char_num}')
    if not os.path.exists(f'{train_path}/{char_num}'):
        os.mkdir(f'{train_path}/{char_num}')

    data = pd.read_table(txt_name, header=None, sep=' ')    # read txt file to pandas dataframe
    data.drop(columns=[0, 1, 8], inplace=True)              # 把不用用到的column砍掉
    data.columns = range(data.shape[1])                     # 重新排列 column
    stroke_total=len(data.groupby(data.iloc[:, -1]).groups) # 總筆畫
    
    for stroke_num in range(stroke_total):
        stroke_num = stroke_num + 1         # int
        stroke_idx = f'{stroke_num:0{args.stroke_idx}d}'    # string
        
        # make directory
        if not os.path.exists(f'{target_path}/{char_num}/{stroke_idx}'):
            os.mkdir(f'{target_path}/{char_num}/{stroke_idx}')
        if not os.path.exists(f'{train_path}/{char_num}/{stroke_idx}'):
            os.mkdir(f'{train_path}/{char_num}/{stroke_idx}')
        
        # index of each stroke
        each_stroke = data.groupby(data.iloc[:, -1]).groups.get(f'stroke{stroke_num}')
        target_data = data.iloc[each_stroke, :]

        # build training data
        for train_num in range(args.train_num):
            filename = f'{char_num}_{stroke_idx}_{train_num + 1 + args.train_start_num:0{args.num_idx}d}.csv'
            file_path = f'{train_path}/{char_num}/{stroke_idx}'
            train_data = addnoise(target_data, args.noise)

            # add train data last line
            train_data = add_last_line(train_data, stroke_len)
            
            # store training data
            train_data.to_csv(f'{file_path}/{filename}', header=False ,index=False)

        # add target last line
        target_data = add_last_line(target_data, stroke_len)

        # store target data
        target_data.to_csv(
            f'{target_path}/{char_num}/{stroke_idx}/{char_num}_{stroke_idx}.csv',
            header=False, index=False
        )

if __name__ == '__main__':
    args = argument_setting()

    # build training data
    if args.test_char == None:
        print('Build training data ...\n')
        print(f'input path = {args.input_path}')
        print(f'root path = {args.root_path}')
        target_path = f'{args.root_path}/{args.target_path}/'
        train_path = f'{args.root_path}/{args.train_path}/'
        print(f'target path = {target_path}')
        print(f'train path = {train_path}')
        if not os.path.exists(args.root_path):
            os.mkdir(args.root_path)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
        if not os.path.exists(train_path):
            os.mkdir(train_path)

        for txt_name in sorted(glob(os.path.join(args.input_path, '*.txt'))):
            char_num = txt_name.split('_')[0][-1 * args.char_idx:]
            csv_parser(char_num, txt_name, target_path, train_path, args)
            print(f'{char_num} finished ...')

    # build testing data
    else:
        char_num = f'{args.test_char:0{args.char_idx}d}'
        test_path = f'{args.root_path}/{args.test_path}'
        if not os.path.exists(args.root_path):
            os.mkdir(args.root_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)
        test_char_path = f'{test_path}{char_num}/'

        print(f'Building {char_num} testing data ...\n')
        print(f'input path = {args.input_path}')
        print(f'root path = {args.root_path}')
        print(f'test path = {test_path}\n')

        if not os.path.exists(test_char_path):
            os.mkdir(test_char_path)
        print(f'Build the director {test_char_path} success ...\n')
        
        txt_name = f'{args.input_path}/char0{char_num}_stroke.txt'
        data = pd.read_table(txt_name, header=None, sep=' ')        # read txt file to pandas dataframe
        data.drop(columns=[0, 1, 8], inplace=True)                  # 把不用用到的column砍掉
        data.columns = range(data.shape[1])                         # 重新排列 column
        stroke_total = len(data.groupby(data.iloc[:, -1]).groups)   # 總筆畫
        
        for stroke_num in range(stroke_total):
            stroke_num = stroke_num + 1         # int
            stroke_idx = f'{stroke_num:0{args.stroke_idx}d}'    # string

            # make testing directory
            if not os.path.exists(f'{test_char_path}/{stroke_idx}'):
                os.mkdir(f'{test_char_path}/{stroke_idx}')
            
            # index of each stroke
            each_stroke = data.groupby(data.iloc[:, -1]).groups.get(f'stroke{stroke_num}')
            target_data = data.iloc[each_stroke, :]

            # build training data
            for test_num in range(args.test_num):
                filename = f'{char_num}_{stroke_idx}_{test_num + 1:0{args.num_idx}d}.csv'

                test_data = addnoise(target_data, args.noise)
                
                # add test data last line
                test_data = add_last_line(test_data, args.stroke_len)
                
                # store training data
                test_data.to_csv(f'{test_char_path}{stroke_idx}/{filename}', header=False ,index=False)
        print(f'Build {char_num} testing data finished ...\n')

    print('All Done!!!')