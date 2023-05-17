import math

import torch
from timm.models.layers import trunc_normal_
from torch import nn

from models.blocks import CBlock_ln, Global_pred, DHAN, LAM_Module_v2, OverlapPatchEmbed, TransformerBlock
from models.lut import LUTNet
import torch.nn.functional as F


# Short Cut Connection on Final Layer
class Local_pred_S(nn.Module):
    def __init__(self, in_dim=3, dim=16):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1),
                   nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU()]

        self.branch1 = nn.Sequential(*blocks1)
        self.branch2 = DHAN()

        self.conv_fuss1 = nn.Conv2d(int(in_dim * 2), in_dim, kernel_size=1, bias=False)

        self.patch_embed = OverlapPatchEmbed(in_dim, dim)

        self.encoder_1 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for _ in range(1)])

        self.encoder_2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for _ in range(1)])

        self.encoder_3 = nn.Sequential(*[
            TransformerBlock(dim=int(dim), num_heads=1, ffn_expansion_factor=2.66, bias=False,
                             LayerNorm_type='WithBias') for _ in range(1)])

        self.layer_fussion = LAM_Module_v2(in_dim=int(dim * 3))
        self.conv_fuss2 = nn.Conv2d(int(dim * 3), in_dim, kernel_size=1, bias=False)

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

    def forward(self, inp):
        # short cut connection
        branch1 = self.branch1(self.relu(self.conv1(inp)))

        branch2 = self.branch2(inp)

        res = self.conv_fuss1(torch.cat((branch1, branch2), dim=1))

        enc_out = self.patch_embed(res)

        enc_out_1 = self.encoder_1(enc_out)
        enc_out_2 = self.encoder_2(enc_out_1)
        enc_out_3 = self.encoder_3(enc_out_2)

        res = self.layer_fussion(torch.cat(
            [enc_out_1.unsqueeze(1), enc_out_2.unsqueeze(1), enc_out_3.unsqueeze(1)], dim=1))

        res = self.conv_fuss2(res)

        return res


class Model(nn.Module):
    def __init__(self, in_dim=3, depth=2):
        super(Model, self).__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)
        self.depth = depth
        self.lut_arr = nn.ModuleList([LUTNet() for _ in range(depth)])

    def laplacian_pyramid_decomposition(self, img, depth):
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

    def laplacian_pyramid_reconstruction(self, pyramid):
        current = pyramid[-1]
        for i in reversed(range(len(pyramid) - 1)):
            expanded = F.interpolate(current, pyramid[i].shape[2:], mode='bilinear', align_corners=False)
            current = expanded + pyramid[i]
            current = self.lut_arr[i](current)
        return current

    def forward(self, inp):
        inps = self.laplacian_pyramid_decomposition(inp, self.depth)
        inps[-1] = self.local_net(inps[-1])
        res = self.laplacian_pyramid_reconstruction(inps)
        return res


if __name__ == '__main__':
    t = torch.randn(1, 3, 512, 512).cuda()
    model = Model().cuda()
    out = model(t)
    print(out.shape)
