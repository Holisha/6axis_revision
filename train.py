import torch
from tqdm import tqdm


def train(model, device, train_loader, optimizer, criterion, args):
    best_err = None

    for epoch in range(1, args.epochs+1):
        model.train()
        err = 0.0

        for data in tqdm(train_loader, desc=f'epoch: {epoch+1}/{args.epochs}'):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)
            loss = criterion(pred, target)
            err += loss.sum().item()

            if epoch % (args.epochs+1) == 0:
                out2csv(inputs, pred, target, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        err /= len(train_loader)
        print(f'loss: {err:.4f}')
        # update every epoch
        if best_err is None or err < best_err:
            best_err = err
            torch.save(model.state_dict(), f'fsrcnn_{args.scale}x.pt')
