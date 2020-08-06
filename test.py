import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
from model import FeatureExtractor
from utils import writer_builder, model_builder, out2csv, inverse_scaler_transform, model_config
from dataset import AxisDataSet, cross_validation


def test_argument(inhert=False):
    """return test arguments

    Args:
        inhert (bool, optional): return parser for compatiable. Defaults to False.

    Returns:
        parser_args(): if inhert is false, return parser's arguments
        parser(): if inhert is true, then return parser
    """

    if inhert is True:
        parser = ArgumentParser(add_help=False)
    else:
        parser = ArgumentParser(add_help=True)

    # dataset setting
    parser.add_argument('--test-path', type=str, default='../dataset/test',
                        help='test dataset path (default: ../datasettest)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='set the number of processes to run (default: 8)')

    # model setting
    parser.add_argument('--model-name', type=str, default='FSRCNN',
                        metavar='FSRCNN, DDBPN' ,help="set model name (default: 'FSRCNN')")
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')
    parser.add_argument('--model-args', nargs='*', type=int, default=[],
                        help="set other args (default: [])")
    parser.add_argument('--load', action='store_false', default=True,
                        help='load model parameter from exist .pt file (default: True)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')

    # logger setting
    parser.add_argument('--log-path', type=str, default='../logs/FSRCNN',
                        help='set the logger path of pytorch model (default: ../logs/FSRCNN)')
    
    # save setting
    parser.add_argument('--save-path', type=str, default='../output',
                        help='set the output file (csv or txt) path (default: ../output)')

    # for the compatiable
    if inhert is True:
        return parser

    return parser.parse_args()

# same as with torch.no_grad()
@torch.no_grad()
def test(model, test_loader, criterion, args):
    # load model parameters
    checkpoint = torch.load(f'fsrcnn_{args.scale}x.pt', map_location=f'cuda:{args.gpu_id}')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # declare content loss
    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.eval()

    err = 0.0

    # out2csv
    i = 0   # count the number of loops
    j = 0   # count the number of data

    for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
        inputs, target, _ = data
        inputs, target = inputs.cuda(), target.cuda()

        # get inputs min and max
        scale_min = inputs.min(2, keepdim=True)[0].detach()
        scale_interval = (inputs.max(2, keepdim=True)[0].detach() - scale_min)

        # normalize to 0~1
        inputs = (inputs - scale_min) / scale_interval

        pred = model(inputs)

        # inverse transform
        # pred = inverse_scaler_transform(pred, target)
        pred = pred * scale_interval + scale_min

        # inverse transform inputs
        # inputs_inverse = inverse_scaler_transform(inputs, target)
        inputs_inverse = inputs * scale_interval + scale_min

        # out2csv
        while j - (i * 64) < pred.size(0):
            out2csv(inputs_inverse, f'test_{int(j/30)+1}_input', args.stroke_length, args.save_path, j - (i * 64))
            out2csv(pred, f'test_{int(j/30)+1}_output', args.stroke_length, args.save_path, j - (i * 64))
            out2csv(target, f'test_{int(j/30)+1}_target', args.stroke_length, args.save_path, j - (i * 64))
            j += 30
        i += 1

        # MSE loss
        mse_loss = criterion(pred, target)

        # content loss
        gen_feature = feature_extractor(pred)
        real_feature = feature_extractor(target)
        content_loss = criterion(gen_feature, real_feature)

        # for compatible
        loss = content_loss + mse_loss
        err += loss.sum().item() * inputs.size(0)

    err /= len(test_loader.dataset)
    print(f'test error:{err:.4f}')


if __name__ == '__main__':
    # argument setting
    test_args = test_argument()

    # config
    model_config(test_args, save=False)     # print model configuration of evaluation

    # set cuda
    torch.cuda.set_device(test_args.gpu_id)

    # model
    model = model_builder(test_args.model_name, test_args.scale, *test_args.model_args).cuda()

    # optimizer and criteriohn
    optimizer = optim.Adam(model.parameters(), lr=test_args.lr)
    criterion = nn.MSELoss()

    # dataset
    test_set = AxisDataSet(test_args.test_path)


    test_loader = DataLoader(test_set,
                             batch_size=test_args.batch_size,
                             shuffle=False,
                             num_workers=test_args.num_workers,
                             pin_memory=True,)

    # test
    test(model, test_loader, criterion, test_args)
