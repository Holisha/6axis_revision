# FSRCNN
import math
import torch.nn as nn

# Light
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# self defined
from dataset import AxisDataSet


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        r"""
        input shape: batch, 1, 59, 6
        output shape: batch, 1, 59 * scale, 6 * scale
        """
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3//2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x


class LightFSRCNN(pl.LightningModule):
    def __init__(self, args, scale_factor=1, num_channels=1, d=56, s=12, m=4, criterion=nn.MSELoss()):
        super(LightFSRCNN, self).__init__()

        self.args = args
        self.criterion = criterion
        self.model = FSRCNN(scale_factor, num_channels, d, s, m)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        self.train_set = AxisDataSet(self.args.train_path)
        self.test_set = AxisDataSet(self.args.test_path)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(),
                                lr=self.args.lr)
    # training
    def train_dataloader(self):
        return DataLoader(self.train_set,
                          batch_size=self.args.batch_size,
                          shuffle=True,
                          num_workers=self.args.num_workers,
                          pin_memory=True)

    def training_step(self, data, idx):
        inputs, target = data
        outputs = self(inputs)

        loss = self.criterion(outputs, target)

        if self.logger is not None:
            self.logger.experiment.add_scalar('train/loss', loss)

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        loss_mean = torch.stack([
            x['loss'] for x in outputs
        ]).mean()

        logs = {'loss': loss_mean}
        return {
            'progress_bar': logs,
            'loss': loss_mean
        }

    # test
    def test_dataloader(self):
        return DataLoader(self.test_set,
                          batch_size=self.args.batch_size,
                          shuffle=False,
                          num_workers=self.args.num_workers,
                          pin_memory=True)

    def test_step(self, data, idx):
        inputs, target = data
        outputs = self(inputs)

        loss = self.criterion(outputs, target)
        if self.logger is not None:
            self.logger.experiment.add_scalar('test/loss', loss)

        return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        loss_mean = torch.stack([
            x['test_loss'] for x in outputs
        ]).mean()

        logs = {'test_loss': loss_mean}
        return {
            'progress_bar': logs,
            'test_loss': loss_mean
        }
