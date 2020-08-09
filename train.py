import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
from model import FeatureExtractor
from utils import (writer_builder, model_builder, optimizer_builder, StorePair,
                out2csv, NormScaler, model_config, summary, config_loader, EarlyStopping)
from dataset import AxisDataSet, cross_validation
from postprocessing import postprocessor

def train_argument(inhert=False):
    """return train arguments

    Args:
        inhert (bool, optional): return parser for compatiable. Defaults to False.

    Returns:
        parser_args(): if inhert is false, return parser's arguments
        parser(): if inhert is true, then return parser
    """

    # for compatible
    parser = ArgumentParser(add_help=not inhert)

    # doc setting
    parser.add_argument('--doc', type=str, metavar='./doc/sample.yaml',
                        help='load document file by position(default: None)')

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
    parser.add_argument('--model-args', action=StorePair, nargs='+', default={},
                        metavar='key=value', help="set other args (default: {})")
    parser.add_argument('--optim', type=str, default='adam',
                        help='set optimizer')
    parser.add_argument('--load', action='store_true', default=False,
                        help='load model parameter from exist .pt file (default: False)')
    parser.add_argument('--version', type=int, dest='load',
                        help='load specific version (default: False)')
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

    # Early-Stop setting
    parser.add_argument('--early-stop', action='store_false', default=True,
                        help='Early stops the training if validation loss does not improve (default: True)')
    parser.add_argument('--patience', type=int, default=5,
                        help='How long to wait after last time validation loss improved. (default: 5)')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Minimum change in the monitored quantity to qualify as an improvement. (default: 0.1)')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If True, prints a message for each validation loss improvement. (default: False)')

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
    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.eval()

    # normalize scaler
    input_scaler = NormScaler()
    # target_scaler = NormScaler()

    # load data
    model_path = f'{args.model_name}_{args.scale}x.pt'
    checkpoint = {'epoch': 1}   # start from 1

    # load model from exist .pt file
    if args.load and os.path.isfile(model_path):
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

    # initialize the early_stopping object
    if args.early_stop:
        early_stopping = EarlyStopping(patience=args.patience, verbose=args.verbose, threshold=args.threshold, path=model_path)

    progress_bar = tqdm(total=len(train_loader)+len(valid_loader))

    for epoch in range(checkpoint['epoch'], args.epochs+1):
        model.train()
        err = 0.0
        valid_err = 0.0

        progress_bar.reset(total=len(train_loader)+len(valid_loader))        
        progress_bar.set_description(f'Train epoch: {epoch}/{args.epochs}')
        for data in train_loader:
            inputs, target, _ = data
            inputs, target = inputs.cuda(), target.cuda()

            # normalize inputs and target
            inputs = input_scaler.fit(inputs)
            # target = target_scaler.fit(target)

            pred = model(inputs)

            # denormalize
            pred = input_scaler.inverse_transform(pred)

            # MSE loss
            mse_loss = args.alpha * criterion(pred, target)

            # content loss
            gen_features = feature_extractor(pred)
            real_features = feature_extractor(target)
            content_loss = args.beta * criterion(gen_features, real_features)

            # for compatible but bad for memory usage
            loss = mse_loss + content_loss

            # update progress bar
            progress_bar.set_postfix({'MSE loss': mse_loss.item(), 'Content loss': content_loss.item()})
            progress_bar.update()

            err += loss.sum().item() * inputs.size(0)

            # update model parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            


        # cross validation
        progress_bar.set_description(f'Valid epoch:{epoch}/{args.epochs}')
        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                inputs, target, _ = data
                inputs, target = inputs.cuda(), target.cuda()

                # normalize inputs and target
                inputs = input_scaler.fit(inputs)
                # target = target_scaler.fit(target)

                pred = model(inputs)

                # unnormalize
                pred = input_scaler.inverse_transform(pred)

                # MSE loss
                mse_loss = criterion(pred, target)

                # content loss
                gen_features = feature_extractor(pred)
                real_features = feature_extractor(target)
                content_loss = criterion(gen_features, real_features)

                # for compatible
                loss = mse_loss + content_loss

                # update tqdm info
                progress_bar.set_postfix({'MSE loss': mse_loss.sum().item(), 'Content loss': content_loss.sum().item()})
                progress_bar.update()

                valid_err += loss.sum().item() * inputs.size(0)

                # out2csv every check interval epochs (default: 5)
                if epoch % args.check_interval == 0:

                    # denormalize value for visualize
                    inputs = input_scaler.inverse_transform(inputs)
                    # pred = input_scaler.inverse_transform(pred)
                    # target = target_scaler.inverse_transform(target)

                    # tensor to csv file
                    out2csv(inputs, f'{epoch}_input', args.save_path, args.stroke_length)
                    out2csv(pred, f'{epoch}_output', args.save_path, args.stroke_length)
                    out2csv(target, f'{epoch}_target', args.save_path, args.stroke_length)

        # compute loss
        err /= len(train_loader.dataset)
        valid_err /= len(valid_loader.dataset)
        print(f'\ntrain loss: {err:.4f}, valid loss: {valid_err:.4f}')

        # update every epoch
        # save model as pickle file
        """if epoch == checkpoint['epoch'] or err < best_err:
            best_err = err  # save err in first epoch

            # save current epoch and model parameters
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                }
                , model_path)
        """

        # update loggers
        writer.add_scalars('Loss/',
                           {'train loss': err, 'valid loss': valid_err},
                           epoch,)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if args.early_stop:
            early_stopping(valid_err, model, epoch)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    writer.close()
    progress_bar.close()


if __name__ == '__main__':
    # argument setting
    train_args = train_argument()

    # replace args by document file
    if train_args.doc:
        train_args = config_loader(train_args.doc, train_args)

    # config
    model_config(train_args, save=True)     # save model configuration before training

    # set cuda
    torch.cuda.set_device(train_args.gpu_id)

    # model
    model = model_builder(train_args.model_name, train_args.scale, **train_args.model_args).cuda()

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
                              pin_memory=False,)
    valid_loader = DataLoader(train_set,
                              batch_size=train_args.batch_size,
                              # shuffle=True,
                              num_workers=train_args.num_workers,
                              sampler=valid_sampler,
                              pin_memory=False,)

    # model summary
    data, _, _ = train_set[0]
    summary(model,
        tuple(data.shape),
        batch_size=train_args.batch_size,
        device='cuda',
        model_name=train_args.model_name.upper(),
        )
    
    # training
    train(model, train_loader, valid_loader, optimizer, criterion, train_args)

    # config
    model_config(train_args, save=False)     # print model configuration after training

    postprocessor(train_args.save_path)