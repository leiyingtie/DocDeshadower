import math

import torch
from timm.models.layers import trunc_normal_
from torch import nn

from models.blocks import CBlock_ln, Global_pred, DHAN
import torch.nn.functional as F


# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]

        self.branch1 = nn.Sequential(*blocks1)
        self.branch2 = DHAN()

        self.branch1_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())

        self.coefficient_1_0 = nn.Parameter(torch.ones((2, int(int(in_dim)))), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, img):
        img1 = self.relu(self.conv1(img))

        # short cut connection
        branch1 = self.branch1(img1)

        branch2 = self.branch2(img)

        branch1 = self.branch1_end(branch1)

        out = self.coefficient_1_0[0, :][None, :, None, None] * branch1 + self.coefficient_1_0[1, :][None, :, None,
                                                                          None] * branch2

        return out


def apply_color(image, ccm):
    shape = image.shape
    image = image.view(-1, 3)
    image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
    image = image.view(shape)
    return torch.clamp(image, 1e-8, 1.0)


def laplacian_pyramid_decomposition(img, depth):
    current = img
    pyramid = []
    for i in range(depth):
        blurred = F.interpolate(current, scale_factor=0.5, mode='bilinear', align_corners=False)
        expanded = F.interpolate(blurred, current.shape[2:], mode='bilinear', align_corners=False)
        residual = current - expanded
        pyramid.append(residual)
        current = blurred
    pyramid.append(current)
    return pyramid


def laplacian_pyramid_reconstruction(pyramid):
    current = pyramid[-1]
    for i in reversed(range(len(pyramid) - 1)):
        expanded = F.interpolate(current, pyramid[i].shape[2:], mode='bilinear', align_corners=False)
        current = expanded + pyramid[i]
    return current


class Model(nn.Module):
    def __init__(self, in_dim=3):
        super(Model, self).__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)

    def forward(self, inp):
        inps = laplacian_pyramid_decomposition(inp, 2)
        inps[-1] = self.local_net(inps[-1])
        res = laplacian_pyramid_reconstruction(inps)
        return res


if __name__ == '__main__':
    t = torch.randn(1, 3, 512, 512).cuda()
    model = Model().cuda()
    out = model(t)
    print(out.shape)
