import torch
from tqdm import tqdm


def test(model, device, test_loader, criterion, args):

    model.load_state_dict(torch.load(f'fsrcnn_{args.scale}x.pt'))
    model.eval()
    err = 0.0

    with torch.no_grad():

        for data in tqdm(test_loader, desc=f'scale: {args.scale}'):
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)

            pred = model(inputs)
            loss = criterion(pred, target)
            err += loss.sum().item()

    err /= len(test_loader)
    print(f'test error:{err:.4f}')
