import torch
import torch.nn as nn
import torch.nn.functional as F


def trilinear(img, lut):
    img = (img - .5) * 2.
    lut = lut[None]
    if img.shape[0] != 1:
        lut = lut.expand(img.shape[0], -1, -1, -1, -1)
    img = img.permute(0, 2, 3, 1)[:, None]
    result = F.grid_sample(lut, img, mode='bilinear', padding_mode='border', align_corners=True)
    return result.squeeze(2)


class LUT3D(nn.Module):
    def __init__(self, dim=33, mode='zero'):
        super(LUT3D, self).__init__()
        if mode == 'zero':
            self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
            self.LUT = nn.Parameter(self.LUT.clone().detach())
        elif mode == 'identity':
            if dim == 33:
                file = open("./IdentityLUT33.txt", 'r')
            elif dim == 64:
                file = open("./IdentityLUT64.txt", 'r')
            lut = file.readlines()
            self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
            for i in range(0, dim):
                for j in range(0, dim):
                    for k in range(0, dim):
                        n = i * dim * dim + j * dim + k
                        x = lut[n].split()
                        self.LUT[0, i, j, k] = float(x[0])
                        self.LUT[1, i, j, k] = float(x[1])
                        self.LUT[2, i, j, k] = float(x[2])
            self.LUT = nn.Parameter(self.LUT.clone().detach())
        else:
            raise NotImplementedError

    def forward(self, img):
        return trilinear(img, self.LUT)


def discriminator_block(in_filters, out_filters, normalization=False):
    """Returns downsampling layers of each discriminator block"""
    layers = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_filters, affine=True))
        # layers.append(nn.BatchNorm2d(out_filters))
    return layers


class Classifier(nn.Module):
    def __init__(self, in_channels=3, num_class=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            # *discriminator_block(128, 128, normalization=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, num_class, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class LUTNet(nn.Module):
    def __init__(self):
        super(LUTNet, self).__init__()
        self.lut0 = LUT3D(mode='identity')
        self.lut1 = LUT3D(mode='zero')
        self.lut2 = LUT3D(mode='zero')
        self.classifier = Classifier()

    def forward(self, inp):
        if self.training:
            pred = self.classifier(inp).squeeze()
            if len(pred.shape) == 1:
                pred = pred.unsqueeze(0)
            gen_0 = self.lut0(inp)
            gen_1 = self.lut1(inp)
            gen_2 = self.lut2(inp)
            res = inp.new(inp.size())
            for b in range(inp.size(0)):
                res[b, :, :, :] = pred[b, 0] * gen_0[b, :, :, :] + pred[b, 1] * gen_1[b, :, :, :] + pred[
                    b, 2] * gen_2[b, :, :, :]
            return res
        else:
            res = []
            for i in range(inp.shape[0]):
                pred = self.classifier(inp[i].unsqueeze(0)).squeeze()
                lut = pred[0] * self.lut0.LUT + pred[1] * self.lut1.LUT + pred[2] * self.lut2.LUT
                res.append(trilinear(inp[i].unsqueeze(0), lut))
            res = torch.cat(res)
            return res
