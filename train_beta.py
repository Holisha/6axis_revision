#TODO: add amp training
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from argparse import ArgumentParser

# self defined
from train import train_argument
from model import FeatureExtractor
from utils import (writer_builder, model_builder, optimizer_builder, StorePair,
                out2csv, NormScaler, model_config, summary, config_loader, EarlyStopping)
from dataset import AxisDataSet, cross_validation
from postprocessing import postprocessor


def train(model, train_loader, valid_loader, optimizer, criterion, args):
    # call content_loss
    feature_extractor = FeatureExtractor().cuda()
    feature_extractor.eval()

    # normalize scaler
    input_scaler = NormScaler(mean=args.mean, std=args.std)
    target_scaler = NormScaler(mean=args.mean, std=args.std)

    # amp
    grad_scaler = GradScaler()

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

    # progress_bar = tqdm(total=len(train_loader)+len(valid_loader))

    for epoch in range(checkpoint['epoch'], args.epochs+1):
        err = 0.0
        valid_err = 0.0

        # progress_bar.reset(total=len(train_loader)+len(valid_loader))        
        # progress_bar.set_description(f'Train epoch: {epoch}/{args.epochs}')
        train_bar = tqdm(train_loader, desc=f'Train epoch: {epoch}/{args.epochs}')
        model.train()
        for data in train_bar:
            inputs, target, _ = data
            inputs, target = inputs.cuda(), target.cuda()

            # normalize inputs and target
            inputs = input_scaler.fit(inputs)
            target = target_scaler.fit(target)

            with autocast():
                pred = model(inputs)

                # denormalize
                # pred = input_scaler.inverse_transform(pred)

                # MSE loss
                mse_loss = args.alpha * criterion(pred, target)

                # content loss
                gen_features = feature_extractor(pred)
                real_features = feature_extractor(target)
                content_loss = args.beta * criterion(gen_features, real_features)

                # for compatible but bad for memory usage
                loss = mse_loss + content_loss

                # update progress bar
                train_bar.set_postfix({'MSE loss': mse_loss.item(), 'Content loss': content_loss.item()})
                # progress_bar.set_postfix({'MSE loss': mse_loss.item(), 'Content loss': content_loss.item()})
                # progress_bar.update()

                err += loss.sum().item() * inputs.size(0)


            # update model parameters
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            # scale loss
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        # cross validation
        # progress_bar.set_description(f'Valid epoch:{epoch}/{args.epochs}')
        valid_bar = tqdm(valid_loader, desc=f'Valid epoch:{epoch}/{args.epochs}', leave=False)
        model.eval()
        with torch.no_grad():
            for data in valid_bar:
            # for data in valid_loader:
                inputs, target, _ = data
                inputs, target = inputs.cuda(), target.cuda()

                # normalize inputs and target
                inputs = input_scaler.fit(inputs)
                target = target_scaler.fit(target)

                pred = model(inputs)

                # denormalize
                # pred = input_scaler.inverse_transform(pred)

                # MSE loss
                mse_loss = criterion(pred, target)

                # content loss
                gen_features = feature_extractor(pred)
                real_features = feature_extractor(target)
                content_loss = criterion(gen_features, real_features)

                # for compatible
                loss = mse_loss + content_loss

                # update tqdm info
                valid_bar.set_postfix({'MSE loss': mse_loss.sum().item(), 'Content loss': content_loss.sum().item()})
                # progress_bar.set_postfix({'MSE loss': mse_loss.sum().item(), 'Content loss': content_loss.sum().item()})
                # progress_bar.update()

                valid_err += loss.sum().item() * inputs.size(0)

                # out2csv every check interval epochs (default: 5)
                if epoch % args.check_interval == 0:

                    # denormalize value for visualize
                    inputs = input_scaler.inverse_transform(inputs)
                    pred = input_scaler.inverse_transform(pred)
                    target = target_scaler.inverse_transform(target)

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
    # progress_bar.close()

def profile(args, model):
    import torch.autograd.profiler as profiler

    inputs = torch.rand(args.batch_size, 1, args.stroke_length, 6).cuda()

    with profiler.profile(profile_memory=True, use_cuda=True, record_shapes=False) as prof:
        model(inputs)
    
    # print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_memory_usage"))
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="self_cuda_memory_usage"))

if __name__ == '__main__':
    assert torch.__version__ >= '1.6', 'torch version must greater than 1.6.0'
    # argument setting
    train_args = train_argument()

    assert train_args.amp == True, 'Must contain --amp when training beta version'

    # replace args by document file
    if train_args.doc:
        train_args = config_loader(train_args.doc, train_args)

    # config
    model_config(train_args, save=True)     # save model configuration before training

    # set cuda
    torch.cuda.set_device(train_args.gpu_id)

    # model
    model = model_builder(train_args.model_name, train_args.scale, **train_args.model_args).cuda()

    profile(train_args, model)
    os._exit(0)

    # optimizer and critera
    optimizer = optimizer_builder(train_args.optim) # optimizer class
    optimizer = optimizer(                          # optmizer instance
        model.parameters(),
        lr=train_args.lr,
        weight_decay=train_args.weight_decay
    )
    criterion = nn.MSELoss()

    # dataset
    full_set = AxisDataSet(train_args.train_path, train_args.target_path)

    # build hold out CV
    train_set, valid_set = cross_validation(
        full_set,
        mode='hold',
        p=train_args.holdout_p,)

    # dataloader
    train_loader = DataLoader(train_set,
                              batch_size=train_args.batch_size,
                              shuffle=True,
                              num_workers=train_args.num_workers,
                            #   sampler=train_sampler,
                              pin_memory=False,)
    valid_loader = DataLoader(valid_set,
                              batch_size=train_args.batch_size,
                              shuffle=False,
                              num_workers=train_args.num_workers,
                            #   sampler=valid_sampler,
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