import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
# from model import FeatureExtractor
from utils import out2csv, csv2txt, writer_builder, model_builder
from dataset import AxisDataSet, cross_validation

def test_argument():
    r"""
    return test arguments
    """
    parser = ArgumentParser()

    # dataset setting
    parser.add_argument('--test-path', type=str, default='../dataset/test',
                        help='test dataset path (default: ../datasettest)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='set the batch size (default: 128)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='set the number of processes to run (default: 8)')

    # model setting
    parser.add_argument('--load', action='store_false', default=True,
                        help='load model parameter from exist .pt file (default: True)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')

    # logger setting
    parser.add_argument('--log-path', type=str, default='../logs/FSRCNN',
                        help='set the logger path of pytorch model (default: ../logs/FSRCNN)')
    
    # save setting
    parser.add_argument('--save-path', type=str, default='../output',
                        help='set the output file (csv or txt) path (default: ../output)')

    return parser.parse_args()

# same as with torch.no_grad()
@torch.no_grad()
def test(model, test_loader, criterion, args):
    # load model parameters
    checkpoint = torch.load(f'fsrcnn_{args.scale}x.pt', map_location=f'cuda:{args.gpu_id}')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # decalre content loss
    # feature_extractor = FeatureExtractor().cuda()
    # feature_extractor.eval()

    err = 0.0

    for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
        inputs, target, _ = data
        inputs, target = inputs.cuda(), target.cuda()

        pred = model(inputs)

        # MSE loss
        mse_loss = criterion(pred - inputs, target - inputs)

        # content loss
        # gen_feature = feature_extractor(pred)
        # real_feature = feature_extractor(target)
        # content_loss = criterion(gen_feature, real_feature)

        # for compatible
        loss = mse_loss
        err += loss.sum().item() * inputs.size(0)

    err /= len(test_loader.dataset)
    print(f'test error:{err:.4f}')

if __name__ == '__main__':
    # argument setting
    test_args = test_argument()

    # set cuda
    torch.cuda.set_device(test_args.gpu_id)

    # model
    model = model_builder('FSRCNN', test_args.scale).cuda()

    # optimizer and criteriohn
    optimizer = optim.Adam(model.parameters(), lr=test_args.lr)
    criterion = nn.MSELoss()

    # dataset
    test_set = AxisDataSet(test_args.test_path)

    # dataloader
    test_set = AxisDataSet(test_args.test_path)
    test_loader = DataLoader(test_set,
                             batch_size=test_args.batch_size,
                             shuffle=False,
                             num_workers=test_args.num_workers,
                             pin_memory=True,)

    # test
    test(model, test_loader, criterion, test_args)
