import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
from torch.utils.data import DataLoader

from models import FSRCNN, LightFSRCNN
from dataset import AxisDataSet
from train import train
from test import test
from utils import argument_setting


def main():

    # can call the lightning model in linux
    if sys.platform.startswith('linux') and args.light is True:
        light()
    else:
        normal()


def normal():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = AxisDataSet(args.train_path)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    test_set = AxisDataSet(args.test_path)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    model = FSRCNN(args.scale).to(device)
    if os.path.exists(f'fsrcnn_{args.scale}x.pt'):
        model.load_state_dict(torch.load(f'fsrcnn_{args.scale}x.pt'))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train(model, device, train_loader, optimizer, criterion, args)
    test(model, device, test_loader, criterion, args)


def light():
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger

    logger = TensorBoardLogger('./logs', name='FSRCNN_light')
    model = LightFSRCNN(args)
    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        gpus=1
    )

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    args = argument_setting()
    main()
