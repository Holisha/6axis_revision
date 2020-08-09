import re
import os
import csv
import sys
import json
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from argparse import Action, ArgumentParser, Namespace
from collections import OrderedDict
from glob import glob
from typing import Union, Optional


def stroke_statistics(path='6d/', mode='max'):
    r""" 
    parameter:
    path: path of the 6d axis csv file
    mode: output statstic. i.e. mode:max means return the maximum stroke number
    count the mean, maximum, minimum of the stroke statistics
    output: use parameter 
    """
    max_cnt = np.int64(0)
    mean_cnt = np.int64(0)
    min_cnt = np.int64(100)

    for dir_name in os.listdir(path):
        df = pd.read_csv(
            os.path.join(path, dir_name, dir_name + '.csv'),
            header=None
        )

        tmp = df.groupby(6)[0].count()

        if tmp.max() > max_cnt:
            max_cnt = tmp.max()
        
        if tmp.mean() > mean_cnt:
            mean_cnt = tmp.mean()

        if tmp.min() < min_cnt:
            min_cnt = tmp.min()
    
    print(f'max stroke number:{max_cnt}\nmean stroke number:{mean_cnt}\nmin stroke number:{min_cnt}')

    return {
        'max' : max_cnt,
        'mean': mean_cnt,
        'min' : min_cnt
    }.get(mode, 'error')

################################################################
########################## model info ##########################
################################################################


class StorePair(Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(StorePair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
    
    def __call__(self, parser, namespace, values, option_string=None):
        target = {}

        for pair in values:
            k, v = pair.split('=')

            if re.match(r'^-?\d+\.\d+$', v):
                v = float(v)
            elif v.isdigit():
                v = int(v)

            target[k] = v
        
        # assign value to namespace
        setattr(namespace, self.dest, target)


def writer_builder(log_root, model_name, load: Union[bool, int]=False):
    """Build writer acording to exist or new logs

    save summary writer in: ./log_root/model_name/version_*

    Args:
        log_root (str): logs root
        model_name (str): model's name
        load (Union[bool, int], optional): load existed Tensorboard. Defaults to False.

    Returns:
        SummaryWriter: tensorboard
    """

    from torch.utils.tensorboard import SummaryWriter

    log_root = os.path.join(log_root, model_name.upper())
    print('\n####### logger info #######\n')

    # make sure logs directories exist
    if not os.path.exists('./logs'):
        os.mkdir('logs')

    if not os.path.exists(log_root):
        os.mkdir(log_root)

    # list version of model
    version = os.listdir(log_root)

    # load exist logs
    if version and type(load) is int:
        # check log path is exist or not
        if f'version_{load}' not in version:
            print(f'Logger Error: load non existent writer: {log_path}')
            print('\n####### logger info #######\n')
            os._exit(0)

        log_path = os.path.join(log_root, f'version_{load}')
        # print(f'load exist logger:{log_path}')

    # load specific version
    elif version and load is True:
        log_path = os.path.join(log_root, version[-1])
        # print(f'load exist logger:{log_path}')

    # create new log directory indexed by exist directories
    else:
        log_path = os.path.join(log_root, f'version_{len(version)}')
        os.mkdir(
            log_path
        )
        print(f'create new logger in:{log_path}')
    
    print(f'Tensorboard logger save in:{log_path}')
    print('\n####### logger info #######\n')

    return SummaryWriter(log_path)


def model_builder(model_name, *args, **kwargs):
    """choose which model would be training

    Args:
        model_name (str): FSRCNN, DDBPN 

    Returns:
        model(torch.nn.module): instantiate model
    """
    from model import FSRCNN, DDBPN

    # class object, yet instantiate
    model = {
        'fsrcnn': FSRCNN,    # scale_factor, num_channels=1, d=56, s=12, m=4
        'ddbpn': DDBPN,      # scale_factor, num_channels=1, stages=7, n0=256, nr=64
    }.get(model_name.lower())

    return model(*args, **kwargs)


def model_config(args, save: Union[str, bool]=False):
    """record model configuration

    save model config as config.json
        if save is path, save to the path
        if save is True, save in current directory

    Args:
        args (Argparse object): Model setting
        save (Union[str, bool], optional): save as json file or just print to stdout. Defaults to False.
    """
    print('\n####### model arguments #######\n')
    for key, value in vars(args).items():
        
        # format modified
        value = {
            'model_name': f'{value}'.upper(),
        }.get(key, value)

        print(f'{key}: {value}')
    print('\n####### model arguments #######\n')

    if save:
        # save config as .json file 
        # if user has determined path
        config_path = os.path.join(save, 'config.json') if type(save) is str else 'config.json'

        with open(config_path, 'w') as config:
            json.dump(vars(args), config, indent=4)


def config_loader(doc_path, args):
    """load config instead of argparser

    Noticed that keys must be the same as original arguments
    support config type:
        .json
        .yaml (load with save loader)

    Args:
        doc_path (str): document path
        args : To be replaced arguments 

    Returns:
        argparse's object: for the compatiable 
    """
    with open(doc_path, 'r') as doc:
        format = doc_path.split('.')[-1]
        # determine the doc format
        load_func={
            'yaml': yaml.safe_load,
            'json': json.load,
        }[format]
        doc_args = load_func(doc)

    try:
        del args.doc
        del doc_args['doc']
    except:
        print('There is no "doc" in args parser\n')
        
    try:
        # train path exist
        if args.train_path:
            args.test_path = None
    except:
        print(f'No "train_path" founded in {sys.argv[0]}\n')

    # check which key value is missing
    if vars(args).keys() != doc_args.keys():
        for key in vars(args).keys():
            if not key in doc_args.keys():
                print(f'"{key}" not found in document file!')
        
        print('\nWarning: missing above key in document file, which would raising error')
        # os._exit(0)

    print(f'config loaded: {doc_path}')
    return Namespace(**doc_args)


def optimizer_builder(optim_name: str):
    """build optimizer

    Args:
        optim_name (str): choose which optimizer for training
            'adam': optim.Adam
            'sgd': optim.SGD
            'ranger': Ranger
            'rangerva': RangerVA

    Returns:
        optimizer class, yet instantiate
    """
    from model import Ranger, RangerVA
    
    return {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'ranger': Ranger,   # Bug in Ranger
        'rangerva': RangerVA,
    }.get(optim_name.lower(), 'Error optimizer')


def summary(model, input_size, batch_size=-1, device="cuda", model_name: Optional[str]=None):
    """reference: https://github.com/sksq96/pytorch-summary
    
    modified to desired format

    Args:
        model (nn.module): torch model
        input_size (tuple, list): compute info
        batch_size (int, optional): Defaults to -1.
        device (str, optional): Control tensor dtype. Defaults to "cuda".
        model_name (Optional[str], optional): set model name or use class name. Defaults to None.
    """

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    # print model name
    name = model_name if model_name else model.__class__.__name__
    print(f'{name} summary')
    
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")

################################################################
########################### training ###########################
################################################################


class NormScaler:
    """
    Normalize tensor's value into range 1~0
    And interse tensor back to original rage
    """
    def __init__(self):
        self.min = None
        self.interval = None
    
    def fit(self, tensor):
        """transform tensor into range 1~0

        Args:
            tensor (torch.Tensor): unnormalized value

        Returns:
            shape as origin: inverse value
        """
        shape = tensor.shape
        tensor = tensor.view(shape[0], -1)

        self.min = tensor.min(1, keepdim=True)[0]
        self.interval = tensor.max(1, keepdim=True)[0] - self.min
        tensor = (tensor - self.min) / self.interval
        
        return tensor.view(shape)
        
    def inverse_transform(self, tensor):
        """inverse tensor's value back

        Args:
            tensor (torch.Tensor): normalized value

        Returns:
            shape as origin: inverse value 
        """
        assert self.min is not None, r'ValueError: scaler must fit data before inverse transform'
        assert self.interval is not None, r'ValueError: scaler must fit data before inverse transform'

        shape = tensor.shape
        tensor = tensor.view(shape[0], -1)

        tensor = tensor * self.interval + self.min

        return tensor.view(shape)


def inverse_scaler_transform(pred, target):
    """Inverse pred from range (0, 1) to target range.
    
    pred_inverse = (pred * (max - min)) + min
    
    ---
    Arguments:
        pred {torch.tensor} -- Tensor which is inversed from range (0, 1) to target range.
        target {torch.tensor} -- Inversion reference range.
    ---
    Returns:
        torch.tensor -- pred after inversed.
    """

    # max and min shape is [batch_size, 1, 1, 6]
    max = torch.max(target, 2, keepdim = True)[0]
    min = torch.min(target, 2, keepdim = True)[0]
    
    # pred_inverse = (pred * (max - min)) + min
    pred_inverse = torch.add(torch.mul(pred, torch.sub(max, min)), min)

    return pred_inverse


def out2csv(inputs, file_string, save_path, stroke_length):
    """
    store input to csv file.

    input: tensor data, with cuda device and size = [batch 1 STROKE_LENGTH 6]
    file_string: string, filename

    no output
    """
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    output = np.squeeze(inputs.cpu().detach().numpy())
    table = output[0]

    with open(f'{save_path}/{file_string}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(stroke_length):
            row = [] * 7
            row[1:6] = table[i][:]
            row.append('stroke' + str(1))
            writer.writerow(row)


def save_final_predict_and_new_dataset(inputs,stroke_num, file_string, args,store_data_cnt):
    output = np.squeeze(inputs.cpu().detach().numpy())
    
    for index in range(args.batch_size):
        try:
            table = output[index]
        except:
            break
        num = stroke_num[index]
        if not os.path.isdir(f'final_output/{num}'):
            # os.mkdir(f'new_train/{num}')
            os.mkdir(f'final_output/{num}')

        with open(f'{file_string}/{num}/{num}_{store_data_cnt+index}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in range(args.stroke_length):
                row = [] * 7
                row[1:6] = table[i][:]
                row.append(f'stroke{num}')
                writer.writerow(row)


#early stopping from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, threshold=0.1, path='./checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 5
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            threshold (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.1
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.threshold = threshold
        self.path = path

    def __call__(self, val_loss, model, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score * (1. + self.threshold):
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...\n')

        # save current epoch and model parameters
        torch.save(
            {
                'state_dict': model.state_dict(),
                'epoch': epoch,
            }
            , self.path)
        self.val_loss_min = val_loss


'''def csv2txt(path='./output'):
    r"""
    Convert all CSV files to txt files.

    path: store the csv files (default: './output')
    """
    for csv_name in sorted(glob(os.path.join(path, '*.csv'))):

        # read csv file content
        with open(csv_name, newline='') as csv_file:
            rows = csv.reader(csv_file)
            txt_name = f'{csv_name[:-4]}.txt'

            # store in txt file
            with open(txt_name, "w") as txt_file:
                for row in rows:
                    txt_file.write("movl 0 ")

                    for j in range(len(row) - 1):
                        txt_file.write(f'{float(row[j]):0.4f} ')

                    txt_file.write("100.0000 ")
                    txt_file.write(f'{row[6]}\n')
'''