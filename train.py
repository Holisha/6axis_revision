# TODO: train_argument load  store_true -> str or bool both

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
from model import FeatureExtractor
from utils import writer_builder, model_builder, optimizer_builder, out2csv, inverse_scaler_transform, model_config
from dataset import AxisDataSet, cross_validation


# TODO: change path name, add other args
def train_argument(inhert=False):
    """return train arguments

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
    parser.add_argument('--stroke-length', type=int, default=150,
                        help='control the stroke length (default: 150)')
    parser.add_argument('--train-path', type=str, default='/home/jefflin/dataset/train',
                        help='training dataset path (default: /home/jefflin/dataset/train)')
    parser.add_argument('--target-path', type=str, default='/home/jefflin/dataset/target',
                        help='target dataset path (default: /home/jefflin/dataset/target)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='set the batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='set the number of processes to run (default: 8)')
    parser.add_argument('--holdout-p', type=float, default=0.8,
                        help='set hold out CV probability (default: 0.8)')

    # model setting
    parser.add_argument('--model-name', type=str, default='FSRCNN',
                        metavar='FSRCNN, DDBPN' ,help="set model name (default: 'FSRCNN')")
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')
    parser.add_argument('--model-args', nargs='*', type=int, default=[],
                        help="set other args (default: [])")
    parser.add_argument('--optim', type=str, default='adam',
                        help='set optimizer')
    parser.add_argument('--load', action='store_true', default=False,
                        help='load model parameter from exist .pt file (default: False)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='set the learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', '--wd', type=float, default=1e-2,
                        help="set weight decay (default: 1e-2)")

    # training setting
    parser.add_argument('--alpha', type=float, default=1e-3,
                        help="set loss 1's weight (default: 1e-3)")
    parser.add_argument('--beta', type=float, default=1,
                        help="set loss 2's weight (default: 1)")
    parser.add_argument('--epochs', type=int, default=50,
                        help='set the epochs (default: 50)')
    parser.add_argument('--check-interval', type=int, default=5,
                        help='setting output a csv file every epoch of interval (default: 5)')

    # logger setting
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='set the logger path of pytorch model (default: ./logs)')
    
    # save setting
    parser.add_argument('--save-path', type=str, default='./output',
                        help='set the output file (csv or txt) path (default: ./output)')

    # for the compatiable
    if inhert is True:
        return parser
    
    return parser.parse_args()


def train(model, train_loader, valid_loader, optimizer, criterion, args):
    # call content_loss
    best_err = None
    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.eval()

    device = 'cuda:0'
    # load data
    model_path = f'{args.model_name}_{args.scale}x.pt'
    checkpoint = {'epoch': 1}   # start from 1

    # load model from exist .pt file
    if args.load is True and os.path.isfile(model_path):
        r"""
        load a pickle file from exist parameter

        state_dict: model's state dict
        epoch: parameters were updated in which epoch
        """
        checkpoint = torch.load(model_path, map_location=f'cuda:{args.gpu_id}')
        checkpoint['epoch'] += 1    # start from next epoch
        model.load_state_dict(checkpoint['state_dict'])

    # store the training time
    writer = writer_builder(args.log_path, args.model_name, args.load)

    for epoch in range(checkpoint['epoch'], args.epochs+1):
        model.train()
        err = 0.0
        valid_err = 0.0

        for data in tqdm(train_loader, desc=f'train epoch: {epoch}/{args.epochs}'):
            inputs, target, _ = data
            inputs, target = inputs.cuda(), target.cuda()

            pred = model(inputs)

            # inverse transform pred
            pred = inverse_scaler_transform(pred, target)

            # MSE loss
            mse_loss = args.alpha * criterion(pred, target)

            # content loss
            gen_features = feature_extractor(pred)
            real_features = feature_extractor(target)
            content_loss = args.beta * criterion(gen_features, real_features)

            # for compatible but bad for memory usage
            loss = mse_loss + content_loss

            # print('train loss:', mse_loss, content_loss, loss)
            err += loss.sum().item() * inputs.size(0)

            # out2csv every check interval epochs (default: 5)
            if epoch % args.check_interval == 0:
                out2csv(inputs, f'{epoch}_input', args.stroke_length)
                out2csv(pred, f'{epoch}_output', args.stroke_length)
                out2csv(target, f'{epoch}_target', args.stroke_length)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # cross validation
        model.eval()
        with torch.no_grad():
            for data in tqdm(valid_loader, desc=f'valid epoch: {epoch}/{args.epochs}'):
            # for data in valid_loader:
                inputs, target, _ = data
                inputs, target = inputs.cuda(), target.cuda()

                pred = model(inputs)

                # inverse transform pred
                pred = inverse_scaler_transform(pred, target)
                
                # MSE loss
                mse_loss = criterion(pred, target)

                # content loss
                gen_features = feature_extractor(pred)
                real_features = feature_extractor(target)
                content_loss = criterion(gen_features, real_features)

                # for compatible
                loss = mse_loss + content_loss
                valid_err += loss.sum().item() * inputs.size(0)

        # compute loss
        err /= len(train_loader.dataset)
        valid_err /= len(valid_loader.dataset)
        print(f'train loss: {err:.4f}, valid loss: {valid_err:.4f}')

        # update every epoch
        # save model as pickle file
        if best_err is None or err < best_err:
            best_err = err

            # save current epoch and model parameters
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                }
                , model_path)

        # update loggers
        writer.add_scalars('Loss/',
                           {'train loss': err, 'valid loss': valid_err},
                           epoch,)

    writer.close()


if __name__ == '__main__':
    # argument setting
    train_args = train_argument()

    # config
    model_config(train_args, save=True)     # save model configuration before training

    # set cuda
    torch.cuda.set_device(train_args.gpu_id)

    # model
    model = model_builder(train_args.model_name, train_args.scale, *train_args.model_args).cuda()
    
    # optimizer and critera
    optimizer = optimizer_builder(train_args.optim) # optimizer class
    optimizer = optimizer(                          # optmizer instance
        model.parameters(),
        lr=train_args.lr,
        weight_decay=train_args.weight_decay
    )
    criterion = nn.MSELoss()

    # dataset
    train_set = AxisDataSet(train_args.train_path, train_args.target_path)

    # build hold out CV
    train_sampler, valid_sampler = cross_validation(
        train_set,
        mode='hold',
        p=train_args.holdout_p,)

    # dataloader
    train_loader = DataLoader(train_set,
                              batch_size=train_args.batch_size,
                              # shuffle=True,
                              num_workers=train_args.num_workers,
                              sampler=train_sampler,
                              pin_memory=True,)
    valid_loader = DataLoader(train_set,
                              batch_size=train_args.batch_size,
                              # shuffle=True,
                              num_workers=train_args.num_workers,
                              sampler=valid_sampler,
                              pin_memory=True,)

    # training
    train(model, train_loader, valid_loader, optimizer, criterion, train_args)

    # config
    model_config(train_args, save=False)     # print model configuration after training
