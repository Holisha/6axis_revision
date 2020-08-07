import os
from glob import glob
import re
import pandas as pd

from stroke2char import stroke2char
from axis2img import axis2img
from csv2txt import csv2txt
from utils import argument_setting, get_len

def postprocessor_dir(dir_path, csv_list):
    """Do postprocessor to the directory

    Args:
        dir_path (string): the directory path
        csv_list (list of string): the list of csv files
    """

    # initialize for test files
    test_target = pd.DataFrame(None)
    test_input = pd.DataFrame(None)
    test_output = pd.DataFrame(None)

    if len(csv_list) % 3 != 0:
        print(f"Error!!! csv file numbers in {dir_path} can't be divided by 3 !!!")
        return

    # the total number of different files
    file_len = int(len(csv_list) / 3)

    for file_idx in range(file_len):

        # get feature of file name
        file_feature = re.split(r'_input.csv', csv_list[file_idx * 3])[0]

        target = pd.read_csv(os.path.join(dir_path, f'{file_feature}_target.csv'), header=None)

        # get original length
        stroke_len = get_len(target)

        # drop the extra rows and read input and output data
        target.drop(target.index[stroke_len:], inplace=True)
        input = pd.read_csv(
                    os.path.join(dir_path, f'{file_feature}_input.csv'),
                    nrows=stroke_len,
                    header=None
                )
        output = pd.read_csv(
                    os.path.join(dir_path, f'{file_feature}_output.csv'),
                    nrows=stroke_len,
                    header=None
                )

        # test stroke2char
        if file_feature[:5] == 'test_' and file_feature != 'test_all':
            (
                test_flag,
                test_target, test_input, test_output
            ) = stroke2char(
                    target.iloc[:, :-1], input.iloc[:, :-1], output.iloc[:, :-1],
                    test_target, test_input, test_output,
                    dir_path, stroke_len, int(file_feature[5:])
                )

        # axis2img
        axis2img(target, input, output, file_feature, dir_path)

        # csv2txt
        csv2txt(target, os.path.join(dir_path, f'{file_feature}_target.txt'))
        csv2txt(input, os.path.join(dir_path, f'{file_feature}_input.txt'))
        csv2txt(output, os.path.join(dir_path, f'{file_feature}_output.txt'))

    # save test char file
    if test_target.shape[0] != 0:
        test_target.to_csv(os.path.join(dir_path, 'test_all_target.csv'), header=False, index=False)
        test_input.to_csv(os.path.join(dir_path, 'test_all_input.csv'), header=False, index=False)
        test_output.to_csv(os.path.join(dir_path, 'test_all_output.csv'), header=False, index=False)


def postprocessor():

    # check the path exists or not
    if not os.path.exists(args.path):
        print(f'{args.path} is not exist!!!')
        return

    for root, _ , files in os.walk(args.path):

        # get the list of the csv file name
        csv_files = sorted(list(filter(lambda x: re.match(r'(.*).csv', x), files)))

        # postprocess
        postprocessor_dir(root, csv_files)

        print(f'{root}\tfinished...')

    print('All Done!!!')


if __name__ == '__main__':
    args = argument_setting()
    postprocessor()