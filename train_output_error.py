import torch
from tqdm import tqdm
from utils import out2csv
from loss.models import FeatureExtractor
from torch.utils.tensorboard import SummaryWriter
import numpy as np

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
            diff=torch.sub(pred, inputs)

            # print("preds",pred.size()) ## tensor
            # model_output=np.squeeze(pred.cpu().detach().numpy())
            # table_out = model_output[0]
            # model_input = np.squeeze(inputs.cpu().detach().numpy())      
            # table_in = model_input[0]
            # diff=abs(table_out-table_in)
            # diff_t=torch.from_numpy(diff)
            # pred[0]=diff_t
            # print(pred.size())
            # print(pred) ## pred為誤差 type:tensor

            # MSE loss
            mse_loss = criterion(diff, target-inputs)

            # for compatible
            loss = mse_loss 
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