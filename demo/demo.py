import os, sys, shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
sys.path.append(r'..')
sys.path.append(r'../postprocessing')
sys.path.append(r'../preprocessing')
from demo_utils import argument_setting
from preprocessing import preprocessor
from postprocessing import (postprocessor, verification)
from eval import test

# test defined
from model import FeatureExtractor
from utils import  (model_builder, out2csv, model_config, config_loader, NormScaler,
    criterion_builder, writer_builder)
from dataset import AxisDataSet

def demo_test(args):

    if args.doc:
        args = config_loader(args.doc, args)
    # config
    model_config(args, save=False)     # print model configuration of evaluation

    # set cuda
    torch.cuda.set_device(args.gpu_id)

    # model
    model = model_builder(args.model_name, args.scale, **args.model_args).cuda()

    # criteriohn
    criterion = criterion_builder(args.criterion)

    # dataset
    test_set = AxisDataSet(args.test_path, args.target_path)

    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                            #  pin_memory=True,
                             pin_memory=False,
                             )

    # test
    test(model, test_loader, criterion, args)

def demo(args):

    # Input the number of the character
    test_char = 0
    while test_char < 1 or test_char > args.char_max:
        test_char = int(input(f"Input the number of character (1-{args.char_max}):"))
        if test_char < 1 or test_char > args.char_max:
            print("Please input the number in the correct range!")
        else:
            args.test_char = test_char
            break
    
    # Input the upper limit of the noise range
    noise = -1
    while noise < 0 or noise > args.noise_max:
        noise = float(input(f"Input the upper limit of the noise range (0-{args.noise_max}):"))
        if noise < 0 or noise > args.noise_max:
            print("Please input the number in the correct range!")
        else:
            args.noise = [-1 * noise, noise]
            break
    
    if os.path.exists(f'{args.root_path}/{args.test_path}'):
        shutil.rmtree(f'{args.root_path}/{args.test_path}')
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)

    preprocessor(args)

    print('\n===================================================')
    demo_test(args)

    print('\n===================================================')
    postprocessor(args)

    print('\n===================================================')
    verification(args)

    print('\n===================================================')
    print(f'Testing {args.test_char} with {args.noise}, Done!!!')

if __name__ == '__main__':
	
    # argument setting
    args = argument_setting()
    # config
    # model_config(args, save=False)   # print model configuration of evaluation
    demo(args)