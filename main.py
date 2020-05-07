import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import FSRCNN
from dataset import AxisDataSet

NUM_EPOCHS = 10     # number of epochs
STROKE_LENGTH = 59  # length of each stroke

def train(model, device, train_loader, optimizer, criterion, args):
    best_err = None

    for epoch in range(args.num_epochs):
        model.train()
        err = 0.0

        for data in tqdm(train_loader, desc=f'epoch: {epoch+1}/{args.num_epochs}'):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            pred = model(inputs)
            loss = criterion(pred, target)
            err += loss.sum().item()

            if ((epoch + 1) % (NUM_EPOCHS / 10) == 0):
                # output 6axis of inputs, pred and target to csv file
                out2csv(inputs, str(epoch + 1) + '_input')
                out2csv(pred, str(epoch + 1) + '_output')
                out2csv(target, str(epoch + 1) + '_target')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        err /= len(train_loader)
        print(f'loss: {err:.4f}')
        # update every epoch
        if best_err is None or err < best_err:
            best_err = err
            torch.save(model.state_dict(), f'fsrcnn_{args.scale}x.pt')


def test(model, device, test_loader, criterion, args):

    model.load_state_dict(torch.load(f'fsrcnn_{args.scale}x.pt'))
    model.eval()
    err = 0.0

    with torch.no_grad():

        for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)
            loss = criterion(pred, target)
            err += loss.sum().item()
            # out2csv(pred, "test_final")
    
    print(f'test error:{err:.4f}')

def out2csv(input, file_string):
    '''
    store input to csv file.
    
    input: tensor data, with cuda device and size = [batch 1 STROKE_LENGTH 6]
    file_string: string, filename
    
    no output
    '''
    output = np.squeeze(input.cpu().detach().numpy())
    table = output[0]
    with open('output/output_' + file_string + '.csv', 'w', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(STROKE_LENGTH):
            row=[]*7
            row[1:6] = table[i][:]
            row.append('stroke' + str(1))
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./train')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--scale', type=int, default=1)
    parser.add_argument('--num-epochs', type=int, default=NUM_EPOCHS)

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('device='+str(device))
    train_set = AxisDataSet(args.path)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=True)

    test_set = AxisDataSet('test')
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


if __name__ == '__main__':
    main()
