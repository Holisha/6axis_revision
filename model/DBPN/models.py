import torch
import torch.nn as nn


def unit_discriminator(in_channels, out_channels, kernel_size, stride=1, padding=0, up=True):
    # upsampling
    if up:
        conv = nn.ConvTranspose2d
    else:
        conv = nn.Conv2d

    return nn.Sequential(
        conv(in_channels, out_channels, kernel_size, stride, padding),
        nn.PReLU(out_channels)
    )


class Projection(nn.Module):
    # build project unit
    def __init__(self, nr, kernel_size, stride, padding, up=True):
        super(Projection, self).__init__()

        self.block1 = unit_discriminator(nr, nr, kernel_size, stride, padding, up)
        self.block2 = unit_discriminator(nr, nr, kernel_size, stride, padding, not up)
        self.block3 = unit_discriminator(nr, nr, kernel_size, stride, padding, up)

    def forward(self, x):
        # resampling
        scale_0 = self.block1(x)
        # error
        scale_1 = self.block2(scale_0)
        scale_2 = self.block3(scale_1 - x)

        # resampling + error
        return scale_0 + scale_2


class DenseProjection(nn.Module):
    # in order to solve gradient vanishing
    def __init__(self, cur_stage, nr, kernel_size, stride, padding, up=True):
        super(DenseProjection, self).__init__()

        # conv(1 * 1)
        self.block1 = unit_discriminator(cur_stage * nr, nr, 1, up=False)
        # same as projection
        self.block2 = unit_discriminator(nr, nr, kernel_size, stride, padding, up)
        self.block3 = unit_discriminator(nr, nr, kernel_size, stride, padding, not up)
        self.block4 = unit_discriminator(nr, nr, kernel_size, stride, padding, up)

    def forward(self, x):
        # bottle neck
        x = self.block1(x)

        # resampling
        scale_0 = self.block2(x)

        # errror
        scale_1 = self.block3(scale_0)
        scale_2 = self.block4(scale_1 - x)
        
        # resampling + error
        return scale_0 + scale_2


class DDBPN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, stages=7, n0=256, nr=64):
        super(DDBPN, self).__init__()
        self.num_stages = stages

        # convolution setting
        kernel_size, stride, padding = {
            1: ((4, 3), 1, (2, 1)),   # version 1
            # 1: (4, 1, 2),     # version 2
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2),
        }[scale_factor]

        # reconstruction layer
        if scale_factor == 1:
            recon_kernel = (4, 3)
            recon_pad = (2, 1)
        else:
            recon_kernel = 3
            recon_pad = 1

        # Feature Extraction
        self.feature = nn.Sequential(
            nn.Conv2d(num_channels, n0, 3, padding=1),
            nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1),
            nn.PReLU(nr)
        )
        # Back Projection Stages
        # projection unit

        # in order to assign parameters
        self.up_projection = nn.ModuleList([
            Projection(nr, kernel_size, stride, padding),
            Projection(nr, kernel_size, stride, padding),
        ])

        self.down_projection = nn.ModuleList([
            Projection(nr, kernel_size, stride, padding, False),
        ])

        # Dense projection
        for i in range(2, stages):
            self.up_projection.append(
                DenseProjection(i, nr, kernel_size, stride, padding)
            )
            self.down_projection.append(
                DenseProjection(i, nr, kernel_size, stride, padding, False)
            )
        
        self.up_projection = nn.Sequential(*self.up_projection)
        self.down_projection = nn.Sequential(*self.down_projection)

        # Reconstruction
        self.reconstruction = nn.Conv2d(stages * nr, num_channels, recon_kernel, padding=recon_pad)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Bottle Neck
        x = self.feature(x)

        # Dense Connection
        h_list = []     # HR image
        l_list = []     # LR image
        for i in range(self.num_stages - 1):

            h_list.append(
                self.up_projection[i](x)
            )

            l_list.append(
                self.down_projection[i](torch.cat(h_list, dim=1))
            )

            x = torch.cat(l_list, dim=1)

        h_list.append(
            self.up_projection[-1](torch.cat(l_list, dim=1))
        )

        x = self.reconstruction(torch.cat(h_list, dim=1))

        return x


class DBPN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, stages=7, n0=256, nr=64):
        super(DBPN, self).__init__()

        self.stages = stages

        # convolution setting
        kernel_size, stride, padding = {
            1: ((4, 3), 1, (2, 1)),   # version 1
            # 1: (4, 1, 2),     # version 2
            2: (6, 2, 2),
            4: (8, 4, 2),
            8: (12, 8, 2),
        }[scale_factor]

        # reconstruction layer
        if scale_factor == 1:
            recon_kernel = (4, 3)
            recon_pad = (2, 1)
        else:
            recon_kernel = 3
            recon_pad = 1
    
        # Feature Extraction
        self.feature = nn.Sequential(
            nn.Conv2d(num_channels, n0, 3, padding=1),
            nn.PReLU(n0),
            nn.Conv2d(n0, nr, 1),
            nn.PReLU(nr)
        )

        # projection unit
        # in order to assign parameters
        self.up_projection = nn.ModuleList([
            Projection(nr, kernel_size, stride, padding),
        ])

        self.down_projection = nn.ModuleList()

        for _ in range(1, stages):
            # up sampling
            self.up_projection.append(
                Projection(nr, kernel_size, stride, padding),
            )

            # down sampling
            self.down_projection.append(
                Projection(nr, kernel_size, stride, padding, False),
            )
        
        self.up_projection = nn.Sequential(*self.up_projection)
        self.down_projection = nn.Sequential(*self.down_projection)

        # Reconstruction
        self.reconstruction = nn.Conv2d(stages * nr, num_channels, recon_kernel, padding=recon_pad)

        self.init_weight()        

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # feature extraction
        x = self.feature(x)

        # DBPN
        hr = []
        for i in range(self.stages-1):

            x = self.up_projection[i](x)
            hr.append(x)
            x = self.down_projection[i](x)
        
        x = self.up_projection[i+1](x)
        hr.append(x)

        # reconstruction
        x = self.reconstruction(
            torch.cat(hr, dim=1)
        )

        return x


class ADBPN(DBPN):
    def __init__(self, *args, col_slice=3, stroke_len=150, **kwargs):
        """
        Attention DBPN
            pool -> fc -> tanh

        Args:
            col_slice (int, optional): slice the column to reduce repetitively value impact on avg pool. Defaults to 3.
            stroke_len (int, optional): stroke length. Defaults to 150 (maximum stroke in dataset).
        """
        super().__init__(*args, **kwargs)
        self.stroke_len = stroke_len
        
        self.glob_avgpool = nn.AdaptiveAvgPool1d(6 * col_slice)

        self.attention = nn.Sequential(
            nn.Linear(6 * col_slice, 6),
            nn.Tanh(),
        )

        self.init_weight()
    
    def forward(self, x):
        weight = x.view(-1, 1, self.stroke_len*6)
        weight = self.glob_avgpool(weight)
        weight = self.attention(weight).unsqueeze(2)
        
        return DBPN.forward(self, weight * x)


if __name__ == '__main__':
    # model = DDBPN(1, 1, 7)
    # model = DBPN(1, 1, 2) 
    model = ADBPN(1, 1, 2, col_slice=3, stroke_len=150)

    # print(model)
    # for idx, param in enumerate(model.modules()):
        # if isinstance(param, nn.PReLU):
            # print(f'{idx}:{param}')

    torch.manual_seed(1)

    with torch.no_grad():
        x = torch.rand(150, 6).view(1, 150, 6)
        print(f'input: {x.shape}')

        out = model(x.unsqueeze(0))
        print(f'output: {out.squeeze(0).shape}')
