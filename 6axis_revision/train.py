import torch
from datetime import datetime
from tqdm import tqdm
<<<<<<< HEAD
from utils import out2csv
=======
from utils import out2csv, inverse_scaler_transform
>>>>>>> 6d81f74abb067d17271f3e609abaa55aa01d64be
from loss.models import FeatureExtractor
from torch.utils.tensorboard import SummaryWriter


def train(model, device, train_loader, valid_loader, optimizer, criterion, args):
    best_err = None
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    # store the training time
    writer = SummaryWriter(args.log_path + '/' + datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))

    for epoch in range(1, args.epochs+1):
        model.train()
        err = 0.0
        valid_err = 0.0
        train_cnt, valid_cnt = 0, 0         # to compute the loss

        for data in tqdm(train_loader, desc=f'train epoch: {epoch}/{args.epochs}'):
            train_cnt += 1

            # read data from data loader
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            # predicted fixed 6 axis data
            pred = model(inputs)

<<<<<<< HEAD
=======
            # inverse transform pred
            pred = inverse_scaler_transform(pred, target)

>>>>>>> 6d81f74abb067d17271f3e609abaa55aa01d64be
            # MSE loss
            mse_loss = criterion(pred, target)

            # content loss
            gen_features = feature_extractor(pred)
            real_features = feature_extractor(target)
            content_loss = criterion(gen_features, real_features)

            # for compatible
            loss = mse_loss + content_loss

            err += loss.sum().item()

<<<<<<< HEAD
            # out2csv every 10 epochs
            if epoch % args.check_interval == 0:
                out2csv(inputs, f'{epoch}_input', args.stroke_length)
=======
            # out2csv each args.check_interval epochs
            if epoch % args.check_interval == 0:
                # inverse transform inputs
                inputs_inverse = inverse_scaler_transform(inputs, target)

                # out2csv
                out2csv(inputs_inverse, f'{epoch}_input', args.stroke_length)
>>>>>>> 6d81f74abb067d17271f3e609abaa55aa01d64be
                out2csv(pred, f'{epoch}_output', args.stroke_length)
                out2csv(target, f'{epoch}_target', args.stroke_length)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # cross validation
        model.eval()
        for data in tqdm(valid_loader, desc=f'valid epoch: {epoch}/{args.epochs}'):
            valid_cnt += 1

            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)
<<<<<<< HEAD
=======
            
            # inverse transform pred
            pred = inverse_scaler_transform(pred, target)
>>>>>>> 6d81f74abb067d17271f3e609abaa55aa01d64be

            # MSE loss
            mse_loss = criterion(pred, target)

            # content loss
            gen_features = feature_extractor(pred)
            real_features = feature_extractor(target)
            content_loss = criterion(gen_features, real_features)

            # for compatible
            loss = mse_loss + content_loss

            valid_err += loss.sum().item()

        err /= train_cnt
        valid_err /= valid_cnt
        print(f'train loss: {err:.4f}, valid loss: {valid_err:.4f}')

        # update every epoch
        if best_err is None or err < best_err:
            best_err = err
            torch.save(model.state_dict(), f'fsrcnn_{args.scale}x.pt')

        writer.add_scalars('Loss/', {'train loss': err,
                                          'valid loss': valid_err}, epoch)

    writer.close()