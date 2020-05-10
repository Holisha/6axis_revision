import torch
from tqdm import tqdm
from utils import out2csv
from loss.models import FeatureExtractor
from torch.utils.tensorboard import SummaryWriter


def train(model, device, train_loader, optimizer, criterion, args):
    best_err = None
    feature_extractor = FeatureExtractor().to(device)
    feature_extractor.eval()
    writer = SummaryWriter(args.log_path)

    for epoch in range(1, args.epochs+1):
        model.train()
        err = 0.0

        for data in tqdm(train_loader, desc=f'epoch: {epoch}/{args.epochs}'):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)

            # MSE loss
            mse_loss = criterion(pred, target)

            # content loss
            gen_features = feature_extractor(pred)
            real_features = feature_extractor(target)
            content_loss = criterion(gen_features, real_features)

            # for compatible
            loss = mse_loss + content_loss

            err += loss.sum().item()

            # out2csv every 10 epochs
            if epoch % args.check_interval == 0:
                out2csv(inputs, f'{epoch}_input', args.stroke_length)
                out2csv(pred, f'{epoch}_output', args.stroke_length)
                out2csv(target, f'{epoch}_target', args.stroke_length)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        err /= len(train_loader)
        print(f'loss: {err:.4f}')

        # update every epoch
        if best_err is None or err < best_err:
            best_err = err
            torch.save(model.state_dict(), f'fsrcnn_{args.scale}x.pt')

        writer.add_scalar('Loss/train', err)

    writer.close()