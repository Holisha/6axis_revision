import os
from glob import glob
import re
import pandas as pd

from stroke2char import stroke2char
from axis2img import axis2img
from csv2txt import csv2txt
from post_utils import get_len, argument_setting

def postprocessor_dir(dir_path, csv_list,path):
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
                    header=None,
                )
        output = pd.read_csv(
                    os.path.join(dir_path, f'{file_feature}_output.csv'),
                    nrows=stroke_len,
                    header=None
                )

        # test stroke2char
        if file_feature[:5] == 'test_' and file_feature != 'test_all':
            (
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
        org_list=compare(test_target,path)
        test_target,test_input,test_output=inverse_len(test_target,test_input,test_output,org_list)
        print(test_target)
        print(test_input)

        # TODO: delete the uselee line judging from the target files in the original txt file
        test_target.to_csv(os.path.join(dir_path, 'test_all_target.csv'), header=False, index=False)
        test_input.to_csv(os.path.join(dir_path, 'test_all_input.csv'), header=False, index=False)
        test_output.to_csv(os.path.join(dir_path, 'test_all_output.csv'), header=False, index=False)
        # axis2img
        axis2img(test_target, test_input, test_output, 'test_all', dir_path)

        # csv2txt
        csv2txt(test_target, os.path.join(dir_path, f'test_all_target.txt'))
        csv2txt(test_input, os.path.join(dir_path, f'test_all_input.txt'))
        csv2txt(test_output, os.path.join(dir_path, f'test_all_output.txt'))
def compare(test_target,path):
    # pd.options.display.float_format='${:,.4f}'.format
    # test_target=test_target.iloc[:,:].map('${:,.4f}'.format)
    ### path 須為output/char00042
    
    path=path.split('/')
    try:
        filename="/home/jefflin/6axis/"+path[1]+"_stroke.txt"
        data_txt = pd.read_csv(filename, sep=" ", header=None)
    except:
        print("Output file path is not found! Please check the format" )
        print("Expecting the format such like output/char00042")

    print(data_txt.iloc[1,2:8].values)
    print(test_target.iloc[1,0:6].values)
    org_list=[]
    for i in range(test_target.shape[0]):
        for j in range(data_txt.shape[0]):
            ans=(test_target.iloc[i,0:6].values==data_txt.iloc[j,2:8].values)
            if ans.all():
                org_list.append(i)
                continue
    print(org_list)
    return org_list

def inverse_len(test_target,test_input,test_output,org_list):
    
    new_target,new_input,new_output=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    for idx in org_list:
        new_target.append(test_target.iloc[idx],ignore_index=True)
        new_input.append(test_input.iloc[idx],ignore_index=True)
        new_output.append(test_output.iloc[idx],ignore_index=True)
        new_target=pd.concat([new_target,test_target.iloc[idx]],axis=1)
        new_input=pd.concat([new_input,test_input.iloc[idx]],axis=1)
        new_output=pd.concat([new_output,test_output.iloc[idx]],axis=1)
    
    return new_target.T,new_input.T,new_output.T
def postprocessor(path):
    """postprocess output files

    Args:
        path (string, optional): the path of the output directory.
    """

    # check the path exists or not
    if not os.path.exists(path):
        print(f'{path} is not exist!!!')
        return

    for root, _ , files in os.walk(path):

        # get the list of the csv file name
        csv_files = sorted(list(filter(lambda x: re.match(r'(.*).csv', x), files)))

        # postprocess
        postprocessor_dir(root, csv_files, path)

        print(f'{root}\tfinished...')

    print('All Done!!!')


if __name__ == '__main__':
    args = argument_setting()
    postprocessor(args.path)