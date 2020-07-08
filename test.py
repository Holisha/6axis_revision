import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from loss.models import FeatureExtractor
from utils import out2csv, inverse_scaler_transform,save_final_predict_and_new_dataset

def test(model, device, test_loader, criterion, args):

    # call model
    model.load_state_dict(torch.load(f'fsrcnn_{args.scale}x.pt'))
    model.eval()

    # call content loss
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()

    err = 0.0

    with torch.no_grad():
        store_data_cnt=0
        for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
            inputs, target, stroke_num= data
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)
            pred = inverse_scaler_transform(pred, target)



            # save_final_predict_and_new_dataset(pred, f'output_test',args,store_data_cnt)
            store_data_cnt+=args.batch_size

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
            inputs, target, stroke_num= data
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
