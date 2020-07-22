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
from test import test_argument
from utils import out2csv, csv2txt, writer_builder, model_builder, model_config
from dataset import AxisDataSet, cross_validation


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

    # dataloader
    test_set = AxisDataSet(test_args.test_path)
    test_loader = DataLoader(test_set,
                             batch_size=test_args.batch_size,
                             shuffle=False,
                             num_workers=test_args.num_workers,
                             pin_memory=True,)

    # test
    test(model, test_loader, criterion, test_args)
