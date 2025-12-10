# code/decoder_heads.py

"""
Decoder head modules for multi-temporal segmentation models.
Includes TempSkip, ConvModule, PPM, and UPerHead_MultiOut.
"""

import torch
import torch.nn as nn
from abc import ABCMeta
from utils import resize

class TempSkip(nn.Module):
    """
    Skip connections module that combines x_t and x_sp by weighting x_sp with x_t element-wise.
    """

    def __init__(self, in_channels: int, out_channels: int, inplace: bool = False, activation: str = 'relu'):
        super().__init__()
        # 1x1 convolution for channel adjustment
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'relu': self.activate = nn.ReLU(inplace=inplace)
        elif activation == 'leakyrelu': self.activate = nn.LeakyReLU(inplace=inplace)
        elif activation == 'gelu': self.activate = nn.GELU()

    def forward(self, x_sp, x_t):
        # Element-wise multiplication and conv/bn/activation
        x_out = torch.mul(x_sp, x_t)
        x_out = self.conv1x1(x_out)
        x_out = self.bn(x_out)
        x_out = self.activate(x_out)
        return x_out

class ConvModule(nn.Module):
    """
    A conv block that bundles conv/norm/activation layers.
    Simplified for usage with Swin Transformer.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, inplace: bool = False, activation: str = 'relu'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation,groups=groups
                              )
        self.bn = nn.BatchNorm2d(out_channels)
        if activation == 'relu': self.activate = nn.ReLU(inplace=inplace)
        elif activation == 'leakyrelu': self.activate = nn.LeakyReLU(inplace=inplace)
        elif activation == 'gelu': self.activate = nn.GELU()

        # Use msra init by default

    def forward(self, x: torch.Tensor):
        x_conv = self.conv(x)
        x_normed = self.bn(x_conv)
        x_out = self.activate(x_normed)
        return x_out

class PPM(nn.ModuleList):
    """
        Pooling Pyramid Module used in PSPNet.
        Simplified for usage with Swin Transformer.
    """

    def __init__(self, pool_scales: tuple, in_channels: int, channels: int, align_corners: bool, activation: str = 'relu'):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.activation = activation

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                        activation=self.activation
                    )
                )
            )

    def forward(self, x: torch.Tensor):
        # Apply each PPM branch and upsample to input size
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class UPerHead_MultiOut(nn.Module, metaclass=ABCMeta):
    """
        Unified Perceptual Parsing head for multi-temporal segmentation.
        Combines PPM and FPN modules.
    """
    def __init__(self,in_channels: list, channels: int, num_classes: int, pool_scales: tuple = (1, 2, 3, 6),
                 dropout_ratio: float = 0.1, align_corners: bool = False, ST_light: bool = True, temp_skip: bool = True,
                 activation: str = 'relu', pe_skip: bool = False, embed_dim_enc: int = 0, ptsz: int = 0, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.temp_skip=temp_skip
        self.ST_light = ST_light
        self.activation = activation
        self.pe_skip = pe_skip
        self.embed_dim_enc = embed_dim_enc
        self.ptsz = ptsz

        self.psp_modules = PPM(
            pool_scales, self.in_channels[-1], self.channels,
            align_corners=self.align_corners, activation=self.activation
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, 3,
            padding=1, activation=self.activation
        )

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:
            if self.temp_skip:
                l_conv = TempSkip(in_channels, self.channels, inplace=False, activation=self.activation)
            else:
                l_conv = ConvModule(in_channels, self.channels, 1, inplace=False, activation=self.activation)
            self.lateral_convs.append(l_conv)
            fpn_conv = ConvModule(self.channels, self.channels, 3, padding=1, inplace=False, activation=self.activation)
            self.fpn_convs.append(fpn_conv)

        # Add bottleneck lateral conv and fpn conv
        if self.temp_skip:
            self.lateral_convs.append(TempSkip(self.in_channels[-1], self.channels,
                                               inplace=False, activation=self.activation))
        else:
            self.lateral_convs.append(ConvModule(self.in_channels[-1], self.channels, 1,
                                                 inplace=False, activation=self.activation))
        self.fpn_convs.append(ConvModule(self.channels, self.channels, 3, padding=1,
                                         inplace=False, activation=self.activation))

        inp_dim_fpn = (len(self.in_channels) + 1) * self.channels
        self.fpn_bottleneck = ConvModule(
            inp_dim_fpn, self.channels,3,padding=1, activation=self.activation
        )

        if self.pe_skip:
            self.add_conv = ConvModule(self.channels + embed_dim_enc, int(self.channels/2),3,
                                       padding=1, activation=self.activation)
            self.conv_seg = nn.Conv2d(int(channels/2), num_classes, kernel_size=1)
        else:
            self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def psp_forward(self, inputs: list):
        """
            Forward function of PSP module.
        """
        x = inputs[-1]
        T = x.shape[1]
        C = x.shape[2]
        H, W = x.shape[-2:]
        x_in = x.view(-1, C, H, W)
        psp_outs = [x_in]
        psp_outs.extend(self.psp_modules(x_in))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        output = output.view(-1, T, self.channels, H, W)
        return output

    def cls_seg(self, feat: torch.Tensor):
        """
            Classify each pixel.
        """
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs: list, x_t: list = None, x_sp: list = None, list_skip_pe: list = None):
        """
            Forward function.
        """

        n_latconvs = len(self.lateral_convs)
        laterals = []
        for i in range(n_latconvs):
            lateral_conv = self.lateral_convs[i]
            T = inputs[i].shape[1]
            H, W = inputs[i].shape[-2:]
            if self.temp_skip:
                x_sp_in = x_sp[i].view(-1, self.in_channels[i], H, W)
                x_t_in = x_t[i].view(-1, self.in_channels[i], H, W)
                lat = lateral_conv(x_sp_in, x_t_in)
                lat = lat.view(-1, T, self.channels, H, W)
            else:
                x_in = inputs[i].view(-1, self.in_channels[i], H, W)
                lat = lateral_conv(x_in)
                lat = lat.view(-1, T, self.channels, H, W)
            laterals.append(lat)
        laterals.append(self.psp_forward(inputs))

        used_backbone_levels = len(laterals)
        dec_outs = laterals.copy()

        for i in range(used_backbone_levels-1, 0,-1):
            prev_shape = laterals[i - 1].shape[-2:]
            T = laterals[i - 1].shape[1]
            if i == used_backbone_levels-1:
                dec_out = [laterals[i-1][:,t,:,:,:] + resize(laterals[i][:,t,:,:,:], size=prev_shape, mode='bilinear',
                                                             align_corners=self.align_corners) for t in range(T)]
            else:
                dec_out = [laterals[i - 1][:, t, :, :, :] + resize(dec_outs[i][:, t, :, :, :], size=prev_shape,
                                            mode='bilinear', align_corners=self.align_corners) for t in range(T)]
            dec_outs[i-1] = torch.stack(dec_out, 1)
        laterals_out = dec_outs

        # FPN outputs
        fpn_outs = []
        for i in range(used_backbone_levels - 1):
            fpn_conv = self.fpn_convs[i]
            T = laterals_out[i].shape[1]
            H, W = laterals_out[i].shape[-2:]
            lats = laterals_out[i].view(-1, self.channels, H, W)
            fpn_out = fpn_conv(lats)
            fpn_out = fpn_out.view(-1, T, self.channels, H, W)
            fpn_outs.append(fpn_out)
        fpn_outs.append(laterals_out[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            H, W = fpn_outs[0].shape[-2:]
            T = fpn_outs[i].shape[1]
            Hh, Ww = fpn_outs[i].shape[-2:]
            fpn_outs[i] = resize(fpn_outs[i].view(-1, self.channels, Hh, Ww),size=(H, W),
                                 mode='bilinear',align_corners=self.align_corners
                                 )
            fpn_outs[i] = fpn_outs[i].view(-1, T, self.channels, H, W)
        T = fpn_outs[0].shape[1]
        fpn_outs = torch.cat(fpn_outs, dim=2)

        segmented = []
        for t in range(T):
            cur = fpn_outs[:, t, :, :, :]
            output = self.fpn_bottleneck(cur)
            segmented.append(output)
        fpn_outs = torch.stack(segmented, 1)

        if self.pe_skip:
            T = fpn_outs.shape[1]
            Hh, Ww = fpn_outs.shape[-2:]
            C_cur = fpn_outs.shape[2]
            fpn_outs = resize(fpn_outs.view(-1, C_cur, Hh, Ww), size=(Hh*2, Ww*2), mode='bilinear', align_corners=False)
            fpn_outs = fpn_outs.view(-1, T, C_cur, Hh*2, Ww*2)
            fpn_outs = torch.cat([fpn_outs, list_skip_pe[-1].permute(0,2,1,3,4)], dim=2)
            segmented = []
            for t in range(T):
                cur = fpn_outs[:, t, :, :, :]
                output = self.add_conv(cur)
                segmented.append(self.cls_seg(output))
            output = torch.stack(segmented, 1)
            return output
        else:
            segmented = []
            for t in range(T):
                cur = fpn_outs[:, t, :, :, :]
                segmented.append(self.cls_seg(cur))
            output = torch.stack(segmented, 1)
            return output