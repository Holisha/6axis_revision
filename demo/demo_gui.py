import os, sys, shutil
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

# self defined
sys.path.append(r'..')
sys.path.append(r'../postprocessing')
sys.path.append(r'../preprocessing')
from demo_utils import argument_setting, timer
from preprocessing import preprocessor
from postprocessing import (postprocessor, verification)
from eval import test

# test defined
from utils import  (model_builder, model_config, config_loader, criterion_builder)
from dataset import AxisDataSet

# execution statistics
exe_stat = []
def demo_test(args):

    if args.doc:
        args = config_loader(args.doc, args)
    # config
    # model_config(args, save=False)     # print model configuration of evaluation

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
@timer
def demo(args,noise,test_char):

    args.test_char = test_char
    args.noise = [-1 * noise, noise]

    if os.path.exists(args.test_path):
        shutil.rmtree(args.test_path)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    exe_stat.append(
            preprocessor(args)
        )

    print('\n===================================================')
    exe_stat.append(
        demo_test(args)
    )

    print('\n===================================================')
    exe_stat.append(
        postprocessor(args)
    )

    print('\n===================================================')
    exe_stat.append(
        verification(args)
    )

    print('\n===================================================')
    print(f'Testing number {args.test_char} with noise {args.noise}, Done!!!')

def main(noise,word_idx):
	
    # argument setting
    args = argument_setting()
    # config
    # model_config(args, save=False)   # print model configuration of evaluation
    if args.timer:
        preprocessor = timer(preprocessor)
        demo_test = timer(demo_test)
        postprocessor = timer(postprocessor)
        verification = timer(verification)


    # execution main function
    demo(args,noise,word_idx)

    # timer statistics
    if args.timer:
        import pandas as pd
        stat = pd.DataFrame(exe_stat, index=['preprocessor', 'demo_test', 'postprocessor', 'verification'], columns=['time'])

        stat['percent'] = stat / stat.sum() * 100

        stat = stat.round(
            {'time': 2, 'percent': 2,}
        )
        print(f'\nperformance statistics:\n{stat}')
        print(f'total execution time: {stat["time"].sum()}')