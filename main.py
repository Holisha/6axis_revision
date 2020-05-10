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

def out2csv(input, file_string, stroke_length):
    '''
    store input to csv file.
    
    input: tensor data, with cuda device and size = [batch 1 stroke_length 6]
    file_string: string, filename
    stroke_length: length of each stroke

    no output
    '''
    output_dir = './output/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output = np.squeeze(input.cpu().detach().numpy())
    table = output[0]
    with open(output_dir + 'output_' + file_string + '.csv', 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(stroke_length):
            row=[]*7
            row[1:6] = table[i][:]
            row.append('stroke' + str(1))
            writer.writerow(row)

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
