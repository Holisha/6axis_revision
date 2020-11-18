import os, sys, shutil
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser

# system path
sys.path.append(r'..')
sys.path.append(r'../postprocessing')
sys.path.append(r'../preprocessing')
sys.path.append(r'../model')
from demo_utils import argument_setting, timer
from preprocessing import preprocessor
from postprocessing import (postprocessor, verification)
from model import FeatureExtractor
from eval import test, demo_eval

# test defined
from utils import  (model_builder, model_config, config_loader, criterion_builder)
from dataset import AxisDataSet

# execution statistics
exe_stat = []
def model_env(args):
    """building model environment avoiding to instantiate model.

    Args:
        args : model arguments which is control by demo_utils.argument_setting

    Returns:
        model (torch.nn): build model in cuda device
        criterion(torch.nn): build criterion. Default to mse loss
        extractor(torch.nn): build vgg content loss in cuda device
    """

    if args.doc:
        args = config_loader(args.doc, args)

    # set cuda device
    torch.cuda.set_device(args.gpu_id)

    # model version control
    version = args.load if type(args.load) is int else 0

    # model path and parameter
    model_path = os.path.join(
        args.log_path, args.model_name, f'version_{version}',f'{args.model_name}_{args.scale}x.pt')

    checkpoint = torch.load(model_path, map_location=f'cuda:{args.gpu_id}')

    # loading model
    model = model_builder(args.model_name, args.scale, **args.model_args).cuda()
    model.load_state_dict(checkpoint['state_dict'])

    # build criterion
    criterion = criterion_builder(args.criterion)

    # loading feature extractor
    extractor = FeatureExtractor().cuda() if args.content_loss else None

    return model, criterion, extractor


def data_env(args):
    """build data loader

    Args:
        args : model arguments which is control by demo_utils.argument_setting

    Returns:
        data_loader [torch.utils.data.DataLoader]: dataloader for only one character
    """
    dataset = AxisDataSet(args.test_path, args.target_path)

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                            #  pin_memory=True,
                             pin_memory=False,
                            )
    
    return data_loader

@timer
def efficient_demo(args,noise,test_char):
    

    args.test_char = test_char
    args.noise = [-1 * noise, noise]
    
    if os.path.exists(args.test_path):
        shutil.rmtree(args.test_path)
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)

    exe_stat.append(
        preprocessor(args)
    )

    # construction env first
    model, critetion, extractor = model_env(args)
    data_loader = data_env(args)

    print('\n===================================================')
    exe_stat.append(
        demo_eval(model, data_loader, critetion, args, extractor)
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

def main(noise,word_idx,eff=False):
	
    # argument setting
    args = argument_setting()
    # config
    # model_config(args, save=False)   # print model configuration of evaluation
    if args.timer:
        preprocessor = timer(preprocessor)
        demo_test = timer(demo_test)
        demo_eval = timer(demo_eval)

        postprocessor = timer(postprocessor)
        verification = timer(verification)

    args.efficient=eff
    # execution main function
    demo_func = efficient_demo if args.efficient else demo
    demo_func(args,noise,word_idx)

    # timer statistics
    if args.timer:
        import pandas as pd
        stat = pd.DataFrame(exe_stat, index=['preprocessor', 'demo_efficient', 'postprocessor', 'verification'], columns=['time'])

        stat['percent'] = stat / stat.sum() * 100

        stat = stat.round(
            {'time': 2, 'percent': 2,}
        )
        print(f'\nperformance statistics:\n{stat}')
        print(f'total execution time: {stat["time"].sum()}')