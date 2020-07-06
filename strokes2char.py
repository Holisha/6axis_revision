import os, sys, re
import pandas as pd
from glob import glob
from argparse import ArgumentParser

def get_len(data):
    """get the original length of the stroke

    Args:
        data (pandas.Dataframe): the stroke data to count the length

    Returns:
        int: the original length of the stroke
    """
    match = data.iloc[:,:-1].eq(data.iloc[:,:-1].shift())
    stroke_len = match.groupby(match.iloc[:,0]).groups.get(True)[0]
    return stroke_len


def main():
    """Merge all strokes of one character to one csv file.
    Execution Argument Type: python merge_stroke2one.py --input-path TEST_DIR_PATH"
    Test File Name Type Example: test_15_input.csv
    """
    path = args.input_path
    
    # get the list of all test csv file
    csv_list = sorted(glob(os.path.join(path, 'test_*.csv')))
    
    # get the numbers of stroke
    stroke_num = int(len(csv_list) / 3)

    print(f'test input path = {path}')
    print(f'stroke numbers = {stroke_num}\n')

    # initialize
    target_data = pd.DataFrame()
    input_data = pd.DataFrame()
    output_data = pd.DataFrame()

    # merge by stroke
    for stroke_tmp_idx in range(stroke_num):

        stroke_idx = stroke_tmp_idx + 1

        # read target original data
        target_org = pd.read_csv(csv_list[stroke_idx * 3 - 1], header=None, delimiter=',')

        # get the original length of each stroke
        stroke_len = get_len(target_org)
        
        input_org = pd.read_csv(csv_list[stroke_idx * 3 - 3], nrows=stroke_len, header=None, delimiter=',')
        output_org = pd.read_csv(csv_list[stroke_idx * 3 - 2], nrows=stroke_len, header=None, delimiter=',')
        
        target_org.drop(target_org.index[stroke_len:], inplace=True)

        # Update stroke number
        target_org[6] = [f'stroke{stroke_idx}'] * stroke_len
        input_org[6] = [f'stroke{stroke_idx}'] * stroke_len
        output_org[6] = [f'stroke{stroke_idx}'] * stroke_len
        
        # append data
        target_data = target_data.append(target_org, ignore_index=True)
        input_data = input_data.append(input_org, ignore_index=True)
        output_data = output_data.append(output_org, ignore_index=True)
        
        print(f'stroke {stroke_idx} finished ...')

    # save as csv file
    target_data.to_csv(f'{path}/test_all_target.csv', header=False, index=False)
    input_data.to_csv(f'{path}/test_all_input.csv', header=False, index=False)
    output_data.to_csv(f'{path}/test_all_output.csv', header=False ,index=False)
    
    print(f'\nAll Done, and save all in {path} !!!')

def argument_setting():
    r"""
    return arguments
    """
    parser = ArgumentParser()

    parser.add_argument('--input-path', type=str, default='./output/test/0436/',
                        help='set the input data path (default: ./output/test/0436/)')

    return parser.parse_args()

if __name__ == "__main__":
    args = argument_setting()
    main()