import os, re
from argparse import ArgumentParser

def argument_setting():
    r"""
    return arguments
    """
    parser = ArgumentParser()

    parser.add_argument('--path', type=str, default='./output',
                        help='set the data path (default: ./output)')

    return parser.parse_args()

def ver(args):
    input_path = '../preprocessing/6axis/'
    flag = True
    for root, _ , files in os.walk(args.path):
            # get the list of the csv file name
            csv_files = list(filter(lambda x: re.match(r'test_all_target.txt', x), files))
            if len(csv_files) == 0:
                print(f'{root}: No files found!!!')
                continue
            print(f'{root}\t{csv_files[0]}')
            if root[-4:] == '_all':
                print('all skip...')
                continue
            with open(f'{input_path}/char0{root[-4:]}_stroke.txt', mode='r') as correct_file:
                correct_content = correct_file.read()
                # print(correct_content)
                with open(f'{root}/{csv_files[0]}') as test_file:
                    test_content = test_file.read()
                    if correct_content != test_content:
                        print(f'Error: {root[-4:]} is NOT Correct!!!')
                        flag = False
                        continue
    if flag is True:
        print('All correct!!!')

if __name__ == '__main__':
    args = argument_setting()
    ver(args)