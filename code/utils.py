# code/utils.py

"""
Utility functions for model initialization, resizing, and window partitioning.
"""

import warnings
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from timm.layers import trunc_normal_

############################ Weight initialization functions #############################

def init_weights_const(m: nn.Module):
    """
        Initialize weights with truncated normal and biases with constant 0.
        Supports nn.Linear, nn.Conv2d, nn.LayerNorm.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def init_weights_He(m: nn.Module):
    """
        He initialization for Linear, Conv2d, Conv3d layers.
    """
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
        if isinstance(m, nn.Linear):
            fan_in = m.weight.size(1)
        if isinstance(m, nn.Conv2d):
            fan_in = m.weight.size(1) * m.weight.size(2) * m.weight.size(3)
        if isinstance(m, nn.Conv3d):
            fan_in = m.weight.size(1) * m.weight.size(2) * m.weight.size(3) * m.weight.size(4)
        std = math.sqrt(2.0 / fan_in)
        nn.init.normal_(m.weight, 0.0, std)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

############################# Utils for multi-temporal model #############################

def time_embedding(dates: torch.Tensor, nbts: int, out_dim: int, tau: int =10000):
    """
    Compute temporal positional encoding based on DOY and use position encoding similar to Vaswani et al.
    Args:
        dates: Tensor of date integers (YYYYMMDD).
        nbts: Number of time steps.
        out_dim: Output embedding dimension.
        tau: Scaling parameter.
    Returns:
        te_out: Temporal encoding tensor [B, nbts, out_dim].
    """
    if isinstance(tau, tuple):
        tau = tau[0]

    years = torch.div(dates, 10000, rounding_mode='trunc')
    month_days = dates - (years * 10000)
    month_days[month_days == 229] = 301
    month = torch.div(month_days, 100, rounding_mode='trunc')
    day = month_days - month * 100

    dpm = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(13):
        month[month == i] = sum(dpm[:i])

    doy = month + day
    B = len(dates)
    te_out = torch.zeros(B, nbts, out_dim)

    for i in range(out_dim):
        te_out[:, :, i] = torch.sin(doy / (tau ** (2 * i / out_dim) + math.pi / 2 * (i % 2)))
    return te_out

def resize(input: torch.Tensor, size: tuple = None,scale_factor: float = None, mode: str = 'nearest',
           align_corners: bool = None, warning: bool = True):
    """
        Resize input tensor using torch.nn.functional.interpolate.
        Args:
            input: Input tensor.
            size: Output size.
            scale_factor: Scaling factor.
            mode: Interpolation mode.
            align_corners: Align corners if True.
            warning: Show warning for misaligned corners.
        Returns:
            Resized tensor.
        """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

############################# Window partitioning and reversing functions #############################

def window_partition(x: torch.Tensor, window_size: int):
    """
    Partition input tensor into non-overlapping windows.
    Args:
        x: Tensor [B, H, W, C]
        window_size: int, Size of each window
    Returns:
        windows: Tensor [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
    Reverse window partitioning for single-temporal tensors.
    Args:
        windows: Tensor [num_windows*B, window_size, window_size, C].
        window_size: Size of each window.
        H, W: Height and width of original image.
    Returns:
        x: Tensor [B, H, W, C].
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def window_partition_MultiTime(x: torch.Tensor, window_size: int):
    """
    Partition multi-temporal input tensor into windows.
    Args:
        x: Tensor of shape [B, T, H, W, C]
        window_size: int, Size of each window
    Returns:
        windows: Tensor [num_wind * B, T, window_size, window_size, C]

    """
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, T, window_size, window_size, C)
    return windows

def window_reverse_MultiTime(windows: torch.Tensor, window_size: int, H: int, W: int):
    """
        Reverse window partitioning for multi-temporal tensors.
        Args:
            windows: Tensor [num_windows*B, T, window_size, window_size, C].
            window_size: Size of each window.
            H, W: Height and width of original image.
        Returns:
            x: Tensor [B, T, H, W, C].
        """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    T = windows.shape[1]
    x = windows.view(B, H // window_size, W // window_size, T,
                     window_size, window_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
    return x