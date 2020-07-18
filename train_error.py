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
from utils import out2csv, csv2txt, writer_builder, model_builder, model_config
from dataset import AxisDataSet, cross_validation

# TODO: change path name, add other args
def train_argument():
    r"""
    return training arguments
    """
    parser = ArgumentParser()

    # dataset setting
    parser.add_argument('--stroke-length', type=int, default=150,
                        help='control the stroke length (default: 150)')
    parser.add_argument('--train-path', type=str, default='../dataset/train',
                        help='training dataset path (default: ../dataset/train)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='set the batch size (default: 64)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='set the number of processes to run (default: 8)')
    parser.add_argument('--holdout-p', type=float, default=0.8,
                        help='set hold out CV probability (default: 0.8)')

    # model setting
    parser.add_argument('--model-args', nargs='*', default=['FSRCNN', 1],
                        help="set model name and args (default: ['FSRCNN', 1])")
    parser.add_argument('--load', action='store_true', default=False,
                        help='load model parameter from exist .pt file (default: False)')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='set the model to run on which gpu (default: 0)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='set the learning rate (default: 1e-3)')
    parser.add_argument('--scale', type=int, default=1,
                        help='set the scale factor for the SR model (default: 1)')

    # training setting
    parser.add_argument('--epochs', type=int, default=50,
                        help='set the epochs (default: 50)')
    parser.add_argument('--check-interval', type=int, default=5,
                        help='setting output a csv file every epoch of interval (default: 5)')

    # logger setting
    parser.add_argument('--log-path', type=str, default='../logs/FSRCNN',
                        help='set the logger path of pytorch model (default: ../logs/FSRCNN)')
    
    # save setting
    parser.add_argument('--save-path', type=str, default='../output',
                        help='set the output file (csv or txt) path (default: ../output)')

    return parser.parse_args()

def train(model, train_loader, valid_loader, optimizer, criterion, args):
    # content_loss
    best_err = None
    # feature_extractor = FeatureExtractor().cuda()
    # feature_extractor.eval()

    # load data
    model_path = f'fsrcnn_{args.scale}x.pt'
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
    writer = writer_builder(args.log_path)

    for epoch in range(checkpoint['epoch'], args.epochs+1):
        model.train()
        err = 0.0
        valid_err = 0.0

        for data in tqdm(train_loader, desc=f'train epoch: {epoch}/{args.epochs}'):
            # load data from data loader
            inputs, target, _ = data
            inputs, target = inputs.cuda(), target.cuda()

            # predicted fixed 6 axis data
            pred = model(inputs)

            # MSE loss
            mse_loss = criterion(pred - inputs, target - inputs)

            # content loss
            # gen_features = feature_extractor(pred)
            # real_features = feature_extractor(target)
            # content_loss = criterion(gen_features, real_features)

            # for compatible but bad for memory usage
            loss = mse_loss

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

                inputs, target = data
                inputs, target = inputs.cuda(), target.cuda()

                pred = model(inputs)

                # MSE loss
                mse_loss = criterion(pred - inputs, target - inputs)

                # content loss
                # gen_features = feature_extractor(pred)
                # real_features = feature_extractor(target)
                # content_loss = criterion(gen_features, real_features)

                # for compatible
                loss = mse_loss

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
        writer.add_scalars('Loss/', {'train loss': err,
                                          'valid loss': valid_err}, epoch)

    writer.close()

if __name__ == '__main__':
    # argument setting
    train_args = train_argument()

    # config
    model_config(train_args, save=True)     # save model configuration before training

    # set cuda
    torch.cuda.set_device(train_args.gpu_id)

    # model
    model = model_builder(*train_args.model_args).cuda()


    # optimizer and criteriohn
    optimizer = optim.Adam(model.parameters(), lr=train_args.lr)
    criterion = nn.MSELoss()

    # dataset
    train_set = AxisDataSet(train_args.train_path)

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

