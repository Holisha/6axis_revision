import os
from glob import glob
import re
import pandas as pd

from stroke2char import stroke2char, get_len
from axis2img import axis2img
from csv2txt import csv2txt
from utils import argument_setting
# from tqdm import tqdm

def postprocessor():

    path = args.path

    # check the path exists or not
    if not os.path.exists(args.path):
        print(f'{args.path} is not exist!!!')
        return

    for root, dirs, files in os.walk(path):
        for dir in dirs:

            # # initialize for test files
            test_target = pd.DataFrame(None)
            test_input = pd.DataFrame(None)
            test_output = pd.DataFrame(None)

            dir_path = os.path.join(root, dir)

            csv_list = sorted(glob(os.path.join(dir_path, '*.csv')))

            if len(csv_list) % 3 != 0:
                print(f"Error!!! csv file numbers in {dir_path} can't be divided by 3 !!!")
                continue

            # the total number of different files
            file_len = int(len(csv_list) / 3)

            for file_idx in range(file_len):

                # get feature of file name
                file_feature = re.split(r'[\\/]', csv_list[file_idx * 3])[-1]
                file_feature = re.split(r'_input.csv', file_feature)[0]

                target = pd.read_csv(os.path.join(dir_path, f'{file_feature}_target.csv'), header=None)

                # get original length
                if file_feature != 'test_all':
                    stroke_len = get_len(target)
                else:
                    stroke_len = target.shape[0]
                
                # drop the extra rows and read input and output data
                target.drop(target.index[stroke_len:], inplace=True)
                input = pd.read_csv(os.path.join(dir_path, f'{file_feature}_input.csv'), nrows=stroke_len, header=None)
                output = pd.read_csv(os.path.join(dir_path, f'{file_feature}_output.csv'), nrows=stroke_len, header=None)

                if file_feature[:5] == 'test_' and file_feature != 'test_all':
                    test_flag, test_target, test_input, test_output = stroke2char(
                                                                            target.iloc[:, :-1],
                                                                            input.iloc[:, :-1],
                                                                            output.iloc[:, :-1],
                                                                            test_target, test_input, test_output,
                                                                            dir_path, stroke_len, file_feature[5:]
                                                                        )

                axis2img(target, input, output, file_feature, dir_path)
                csv2txt(target, os.path.join(dir_path, f'{file_feature}_target.txt'))
                csv2txt(input, os.path.join(dir_path, f'{file_feature}_input.txt'))
                csv2txt(output, os.path.join(dir_path, f'{file_feature}_output.txt'))

            print(f'{dir_path}\tfinished...')

    print('All Done!!!')


if __name__ == '__main__':
    args = argument_setting()
    postprocessor()