import os, re
from argparse import ArgumentParser

def argument_setting():
    r"""
    return arguments
    """
    parser = ArgumentParser()

    parser.add_argument('--path', type=str, default='./output',
                        help='set the data path (default: ./output)')
    parser.add_argument('--dataset-path', type=str, default='/home/jefflin/6axis/',
                        help='set the original 6axis data path (default: /home/jefflin/6axis/)')

    return parser.parse_args()

def ver(args):

    flag = True
    for root, _ , files in os.walk(args.path):
            # get the list of the csv file name
            csv_files = list(filter(lambda x: re.match(r'test_all_target.txt', x), files))
            # no test_all_target.txt in root
            if len(csv_files) == 0:
                print(f'{root}:\tNo files found!!!')
                continue
            print(f'{root}\t{csv_files[0]}')

            # in test_all/
            if root[-4:] == '_all':
                print('\tskip test_all/...')
                continue
            with open(f'{args.dataset_path}/char0{root[-4:]}_stroke.txt', mode='r') as correct_file:
                correct_content = correct_file.read()
                with open(f'{root}/{csv_files[0]}') as test_file:
                    test_content = test_file.read()
                    if correct_content != test_content:
                        print(f'\nError: {root[-4:]} is NOT Correct!!!\n')
                        flag = False
                        continue
    if flag is True:
        print('\nAll correct!!!')

if __name__ == '__main__':
    args = argument_setting()
    ver(args)