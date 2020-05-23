import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from loss.models import FeatureExtractor
from utils import out2csv, inverse_scaler_transform

def test(model, device, test_loader, criterion, args):

    # call model
    model.load_state_dict(torch.load(f'fsrcnn_{args.scale}x.pt'))
    model.eval()

    # call content loss
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    err = 0.0

    with torch.no_grad():
        i = 1
        for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)
            pred = inverse_scaler_transform(pred, target)

            # inverse transform inputs
            inputs_inverse = inverse_scaler_transform(inputs, target)

            # out2csv
            out2csv(inputs_inverse, f'test_{str(i)}_input', args.stroke_length)
            out2csv(pred, f'test_{str(i)}_output', args.stroke_length)
            out2csv(target, f'test_{str(i)}_target', args.stroke_length)
            i += 1

            # MSE loss
            mse_loss = criterion(pred, target)

            # content loss
            gen_feature = feature_extractor(pred)
            real_feature = feature_extractor(target)
            content_loss = criterion(gen_feature, real_feature)

            # for compatible
            loss = content_loss + mse_loss
            err += loss.sum().item()

    err /= len(test_loader)
    print(f'test error:{err:.4f}')


def test_gan(model, device, test_loader, criterion, args):

    writer = SummaryWriter(args.log_path)

    # call model
    model.load_state_dict(torch.load(f'fsrcnn_{args.scale}x.pt'))
    model.eval()

    # call content loss
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    err = 0.0

    with torch.no_grad():

        for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)

            # MSE loss
            mse_loss = criterion(pred, target)

            # content loss
            gen_feature = feature_extractor(pred)
            real_feature = feature_extractor(target)
            content_loss = criterion(gen_feature, real_feature)

            # for compatible
            loss = content_loss + mse_loss
            err += loss.sum().item()

    err /= len(test_loader)
    print(f'test error:{err:.4f}')
    writer.add_scalar('loss/test', err)

    writer.close()
