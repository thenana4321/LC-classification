# code/backbone.py

"""
Backbone modules for multi-temporal segmentation models.
Includes Patch embedding functions, attentions based layers and spatio-temporal layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath, to_2tuple
import math
from utils import window_partition, window_reverse, window_partition_MultiTime, window_reverse_MultiTime, time_embedding

########################### Patch embedding layer definitions #############################

class PatchEmbed_Conv(nn.Module):
    """
        Image to Patch Embedding with 3D convolution and maxpool.
    """

    def __init__(self, conv_type: str ='3D', red_factor: int = 4, in_chans: int = 3, embed_dim: int = 96,
                 nb_CB_perblock: int = 2, activation: str = 'relu', pe_skip: bool = False):
        super().__init__()
        self.conv_type = conv_type
        self.red_factor = red_factor
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.activation = activation
        self.nb_CB_perblock = nb_CB_perblock
        self.pe_skip = pe_skip

        if self.activation == 'relu': self.activate = nn.ReLU(inplace=True)
        elif self.activation == 'leakyrelu': self.activate = nn.LeakyReLU(inplace=True)
        elif self.activation == 'gelu': self.activate = nn.GELU()

        # Select layers
        if conv_type == '2D':
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            MaxPool = nn.MaxPool2d
            pool_kernel = 2
        elif conv_type == '3D':
            Conv = nn.Conv3d
            BatchNorm = nn.BatchNorm3d
            MaxPool = nn.MaxPool3d
            pool_kernel = (1, 2, 2)
        else:
            raise ValueError("conv_type must be '2D' or '3D'")

        self.conv_blocks = nn.ModuleList()
        in_dim = in_chans
        nb_up = int(red_factor // 2) if red_factor >=2 else 1
        for i in range(nb_up):
            emb_dim_cur = int(embed_dim // (red_factor//2 ** (i + 1))) if red_factor > 1 else embed_dim
            self.conv_blocks.append(
                nn.Sequential(
                    Conv(in_dim, emb_dim_cur, kernel_size=3, padding=1),
                    BatchNorm(emb_dim_cur),
                    self.activate
                )
            )
            for j in range(1, nb_CB_perblock):
                self.conv_blocks.append(
                    nn.Sequential(
                        Conv(emb_dim_cur, emb_dim_cur, kernel_size=3, padding=1),
                        BatchNorm(emb_dim_cur),
                        self.activate
                    )
                )
            in_dim = emb_dim_cur
        self.down = MaxPool(pool_kernel)

    def forward(self, x: torch.Tensor):
        """
            Forward pass for PatchEmbed_Conv3D.
            Args:
                x: Input Tensor of shape (B,C,T,H,W)
            Returns:
                Output tensor, optionally with skip connections.
        """
        list_skip_pe = []
        for i, block in enumerate(self.conv_blocks):
            x = block(x)
            rest = (i+1)%self.nb_CB_perblock if i > 0 else 2
            if rest == 0 and self.red_factor >= 2:
                list_skip_pe.append(x)
                x = self.down(x)
        if self.pe_skip: return x, list_skip_pe
        else: return x

class PatchEmbed_Swin(nn.Module):
    """
        Image to Patch Embedding for Swin Transformer.
    """

    def __init__(self, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        """
            Forward function.
        """
        _, _, H, W = x.size()

        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x

#################### Standard attention-based layer #######################################

class Mlp(nn.Module):
    """ Multilayer perceptron. Per Swin-Transformer Definition.
    Only change wrt. Mlp-Definition in BuildFormer is default for act_layer!"""

    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer: nn.Module = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    """
        Window based multi-head self attention (W-MSA) module with relative position bias.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim: int, window_size: tuple, num_heads: int, qkv_bias: bool = True, qk_scale: float = None,
                 attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor=None):
        """
            Forward function.
            Args:
                x: input features with shape of (num_windows*B, N, C)
                mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
        Swin Transformer Block.
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
            drop (float, optional): Dropout rate. Default: 0.0
            attn_drop (float, optional): Attention dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim: int, num_heads: int, n_parallel: int = 1, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4., qkv_bias: bool = True, qk_scale: float = None, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0., act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.n_parallel = n_parallel
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x: torch.Tensor, mask_matrix: torch.Tensor):
        """
            Forward function.
            Args:
                x: Input feature, tensor size (B, H*W, C).
                mask_matrix: Attention mask for cyclic shift.
        """
        if self.n_parallel == 1:
            B, L, C = x.shape
        else:
            B, T, L, C = x.shape
            assert T == self.n_parallel
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B * self.n_parallel, H, W, C)
        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1,self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        if self.n_parallel == 1:
            x = x.view(B, H * W, C)
        else:
            x = x.view(B, self.n_parallel, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

############################ Spatio-Temporal Layer Definitions ############################

class WindowTimeAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, nbts: int, qkv_bias: bool = True,
                 qk_scale: float = None, attn_drop: float = 0., proj_drop: float = 0., light: bool ="None",
                 conv: str = 'a', kernel_size: int=3):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.nbts = nbts
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.light = light
        self.conv=conv
        self.kernel_size = kernel_size

        if self.conv == 'c':
            pad = 1 if kernel_size == 3 else 2
            self.convblock = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=self.kernel_size, padding=pad))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        if not self.light:
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        else:
            self.proj_temp = nn.Linear(dim, dim)
            self.proj_sp = nn.Linear(dim, dim)
            self.proj_drop_temp = nn.Dropout(proj_drop)
            self.proj_drop_sp = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward function for spatio-temporal attentions.
            Args:
                x: input features with shape of (num_windows*B, T, N, C)
                mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, T, N, C = x.shape
        if not self.light: # full spatio-temporal attention computation
            x = x.view(B_, -1, C)
            qkv = self.qkv(x).reshape(B_, N * T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))
            if mask is not None:
                nW = mask.shape[0]
                attn = attn.view(B_ // nW, nW, self.num_heads, N*T, N*T) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, N*T, N*T)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N * T, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            x = x.view(B_, T, N, C)
            return x
        else: # light version of spatial-temporal attention computation
            # spatial stream
            if self.conv == 'a': # attention in spatial dimension (parallel for all timesteps)
                x_sp = x.view(B_*T, N, C)
                qkv_sp = self.qkv(x_sp).reshape(B_*T, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
                q_sp, k_sp, v_sp = qkv_sp[0], qkv_sp[1], qkv_sp[2]
                q_sp = q_sp * self.scale
                attn_sp = (q_sp @ k_sp.transpose(-2, -1))
                if mask is not None:
                    nW = mask.shape[0]
                    attn_sp = attn_sp.view(B_*T // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                    attn_sp = attn_sp.view(-1, self.num_heads, N, N)
                    attn_sp = self.softmax(attn_sp)
                else:
                    attn_sp = self.softmax(attn_sp)
                attn_sp = self.attn_drop(attn_sp)
                x_sp = (attn_sp @ v_sp).transpose(1, 2).reshape(B_ * T, N, C)
                x_sp = x_sp.view(B_, T, N, C)
            elif self.conv == 'c': #  convolutions in spatial attention (parallel for all timesteps)
                H = int(math.sqrt(N))
                x_sp = x.view(B_, T, H, H, C).permute(0,1,4,2,3) # B, T, C, H, W
                x_sp = [self.convblock(x_sp[:, i, :, :, :]) for i in range(self.nbts)] # input conv: B, C_in, H, W
                x_sp = torch.stack(x_sp, 1)
                x_sp = x_sp.view(B_, T, C, N).permute(0,1,3,2) # B, T, N, C
            x_sp = self.proj_sp(x_sp)
            x_sp = self.proj_drop_sp(x_sp)

            # temporal attention between all timesteps of one patch - in parallel for each patches
            x_t = x.permute(0,2,1,3)
            x_t = x_t.reshape(B_ * N, T, C)
            qkv_t = self.qkv(x_t).reshape(B_ * N, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]
            q_t = q_t * self.scale
            attn_t = (q_t @ k_t.transpose(-2, -1))
            attn_t = self.softmax(attn_t)
            attn_t = self.attn_drop(attn_t)
            x_t = (attn_t @ v_t).transpose(1, 2).reshape(B_*N, T, C)
            x_t = x_t.view(B_, N, T, C).permute(0,2,1,3)
            x_t = self.proj_temp(x_t)
            x_t = self.proj_drop_temp(x_t)
            return x_t, x_sp

class SwinTimeBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, nbts: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4., use_time_embed: bool = True, qkv_bias: bool = True, qk_scale: float = None,
                 drop: float = 0., attn_drop: float = 0., drop_path: float = 0., act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm, ST_light: bool = True, conv: str ='a', kernel_size: int = 3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.nbts = nbts
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_time_embed = use_time_embed
        self.conv = conv
        self.ST_light = ST_light
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.time_attn = WindowTimeAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, nbts=nbts, qkv_bias=qkv_bias,
            qk_scale=qk_scale, attn_drop=attn_drop,proj_drop=drop, light=ST_light, conv=conv, kernel_size=kernel_size
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.ST_light:
            self.mlp_te = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.mlp_sp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.fuse = nn.Linear(2 * dim, dim)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        """
            Forward function.
            Args:
                x: Input feature, tensor size (B,T, H*W, C).
                mask_matrix: Attention mask for cyclic shift
                    None if light == Tru & conv = c, (nb_w, N, N) if light=="None", (nb_w, NT, NT) if light=0 True & conv=c
        """
        B, T, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        if self.conv == 'c' and self.ST_light: # l-STA and conv in spatial stream --> no window partitioning
            x_t, x_sp = self.time_attn(x)
            x_t = shortcut + self.drop_path(x_t)
            x_t = x_t + self.drop_path(self.mlp_te(self.norm2(x_t)))
            x_sp = shortcut + self.drop_path(x_sp)
            x_sp = x_sp + self.drop_path(self.mlp_sp(self.norm2(x_sp)))
            x = torch.cat((x_t, x_sp), dim=3)  # shape: B, T, N, 2C
            x = self.fuse(x)  # shape: B, T, N, C
            return x, x_t, x_sp
        else:
            # window partitioning
            x = x.view(B * self.nbts, H, W, C)
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape
            # cyclic shift
            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                attn_mask = mask_matrix
            else:
                shifted_x = x
                attn_mask = None
            shifted_x = shifted_x.view(B, self.nbts, Hp, Wp, C)
            x_windows = window_partition_MultiTime(shifted_x, self.window_size) # todo: here before x, test if working correctly (light=yes and attention or light=false)
            x_windows = x_windows.view(-1, self.nbts, self.window_size * self.window_size, C)

            if self.ST_light: # --> light STA with att for both streams --> x_t and x_sp as output
                x_t, x_sp = self.time_attn(x_windows, mask=attn_mask)
                x_t = x_t.view(-1, self.nbts, self.window_size, self.window_size, C)
                shifted_x_t = window_reverse_MultiTime(x_t, self.window_size, Hp, Wp)  # B H' W' C
                if self.shift_size > 0:
                    x_t = torch.roll(shifted_x_t, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                else:
                    x_t = shifted_x_t
                if pad_r > 0 or pad_b > 0: x_t = x_t[:, :, :H, :W, :].contiguous()
                x_t = x_t.view(B, self.nbts, H * W, C)
                x_t = shortcut + self.drop_path(x_t)
                x_t = x_t + self.drop_path(self.mlp_te(self.norm2(x_t)))
                x_sp = x_sp.view(-1, self.nbts, self.window_size, self.window_size, C)
                shifted_x_sp = window_reverse_MultiTime(x_sp, self.window_size, Hp, Wp)  # B H' W' C
                if self.shift_size > 0:
                    x_sp = torch.roll(shifted_x_sp, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                else:
                    x_sp = shifted_x_sp
                if pad_r > 0 or pad_b > 0: x_sp = x_sp[:, :, :H, :W, :].contiguous()
                x_sp = x_sp.view(B, self.nbts, H * W, C)
                x_sp = shortcut + self.drop_path(x_sp)
                x_sp = x_sp + self.drop_path(self.mlp_sp(self.norm2(x_sp)))
                # fuse x_t and x_sp
                x = torch.cat((x_t, x_sp), dim=3) # shape: B, T, N, 2C
                x = self.fuse(x) # shape: B, T, N, C
                return x, x_t, x_sp
            else: # no light STA
                attn_windows = self.time_attn(x_windows)
                attn_windows = attn_windows.view(-1, self.nbts,self.window_size, self.window_size, C)
                shifted_x = window_reverse_MultiTime(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
                # reverse cyclic shift
                if self.shift_size > 0:
                    x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                else:
                    x = shifted_x
                if pad_r > 0 or pad_b > 0: x = x[:, :, :H, :W, :].contiguous()
                x = x.view(B, self.nbts, H * W, C)
                x = shortcut + self.drop_path(x)
                x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x

class SpaceTimeLayer(nn.Module):
    """
        Spatio-temporal layer for multi-temporal segmentation models.
        This layer alternates between spatial and spatio-temporal transformer blocks,
        or uses a light spatio-temporal block depending on the ST_light flag.
        Optionally applies patch merging for downsampling.
    """
    def __init__(self, dim: int, depth: int, num_heads: int, nbts: int, window_size: int = 7, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, qk_scale: float = None, drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0., norm_layer: nn.Module = nn.LayerNorm, downsample: callable = None,
                 use_time_embed: bool = True, ST_light: bool = True, conv: str = 'a', kernel_size: int = 3):
        """
            Initialize SpaceTimeLayer
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_time_embed = use_time_embed
        self.nbts = nbts
        self.ST_light = ST_light

        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = 0 if (i % 2 == 0) else self.shift_size
            if ST_light == False:
                layer_2d = SwinTransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, shift_size=shift,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
                layer_time = SwinTimeBlock(dim=dim, num_heads=num_heads, nbts=nbts, window_size=window_size,
                                           shift_size=shift, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           norm_layer=norm_layer, ST_light= ST_light, conv=conv, kernel_size=kernel_size
                                           )
                self.blocks.append(layer_2d)
                self.blocks.append(layer_time)
            else:
                layer_space_time = SwinTimeBlock(dim=dim, num_heads=num_heads, nbts=nbts, window_size=window_size,
                                           shift_size=shift, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           norm_layer=norm_layer, ST_light=ST_light, conv=conv, kernel_size=kernel_size
                                        )
                self.blocks.append(layer_space_time)
        # patch merging layer
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        """
            Forward pass for SpaceTimeLayer.
            Args:
                x: Input feature, tensor size (B, H*W, C).
                H, W: Spatial resolution of the input feature.
        """
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        counter = 0
        for blk in self.blocks:
            blk.H, blk.W = H, W
            counter += 1
            if not self.ST_light: # normal ST-TB, not light
                if counter % 2 == 0:
                    x = blk(x, attn_mask)
                else:
                    blk_outs = []
                    for i in range(self.nbts):
                        blk_outs.append(blk(x[:, i, :, :], attn_mask))
                    x = torch.stack(blk_outs, 1)
            else: # light-STA --> separate x_t, x_sp
                x, x_t, x_sp = blk(x, attn_mask)


        if self.downsample is not None:
            x_down_list = [self.downsample(x[:, i, :, :], H, W) for i in range(self.nbts)]
            x_down = torch.stack(x_down_list, 1)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            if not self.ST_light:
                return x, H, W, x_down, Wh, Ww
            else:
                return x, H, W, x_down, Wh, Ww, x_t, x_sp
        else:
            if not self.ST_light:
                return x, H, W, x,  H, W
            else:
                return x, H, W, x,  H, W, x_t, x_sp

class PatchMerging(nn.Module):
    """
        Patch Merging Layer to reduce spatial resolution and increase feature dimension.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim: int, norm_layer: nn.Module = nn.LayerNorm):
        """
            Initialize PatchMerging layer.
        """
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        """
            Forward function.
            Args:
                x: Input feature, tensor size (B, H*W, C).
                H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)

        # padding if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x

########################### Multi-time encoder definition #############################

class MT_encoder(nn.Module):
    def __init__(self, img_size: int = 512, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96,
                 depths: list = [2, 2, 6, 2], num_heads: list = [3, 6, 12, 24], window_size: int = 7,
                 mlp_ratio: float = 4.,  num_classes: int = 10, nbts: int = 12, use_te: bool = False, tau: int = 10000,
                 qkv_bias: bool = True, qk_scale: float = None, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.2, norm_layer: nn.Module = nn.LayerNorm, patch_norm: bool = True,
                 ST_light: bool = True, patch_embedding: str = 'swin', conv: str = 'a', nb_CB_PE: int = 2,
                 activation: str = 'relu', kernel_size: int = 3, pe_skip: bool = False
                 ):
        """
            Initialize Multi-Temporal Encoder.
        """
        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.use_te = use_te
        self.tau = tau,
        self.patch_norm = patch_norm
        self.nbts = nbts
        self.ST_light = ST_light
        self.out_indices = [i for i in range(0,self.num_layers)]
        self.patch_embedding = patch_embedding
        self.activation = activation
        self.pe_skip = pe_skip

        if self.patch_embedding == 'swin':
            self.patch_embed_fct = PatchEmbed_Swin(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                norm_layer=norm_layer if self.patch_norm else None)
        elif self.patch_embedding == 'conv2d':
            self.patch_embed_fct = PatchEmbed_Conv(conv_type='2D', red_factor=patch_size, in_chans=in_chans,
                                                   embed_dim=embed_dim, activation=self.activation, pe_skip= self.pe_skip)
        elif self.patch_embedding == 'conv3d':
            self.patch_embed_fct = PatchEmbed_Conv(conv_type='3D', red_factor=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                                     nb_CB_perblock=nb_CB_PE, activation=self.activation, pe_skip= self.pe_skip)

        self.pos_drop = nn.Dropout(p=drop_rate)
        # stochastic depth dropout
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SpaceTimeLayer(dim=int(embed_dim * 2 ** i_layer), depth=depths[i_layer],
                                   num_heads=num_heads[i_layer], nbts=nbts, window_size=window_size,
                                   mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                   norm_layer=norm_layer,
                                   downsample=PatchMerging if ( i_layer < self.num_layers - 1) else None,
                                   ST_light= ST_light, conv=conv, kernel_size=kernel_size)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        for i_layer in self.out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

    def forward(self, x):
        """Forward function."""
        if self.use_te:
            x, dates = x
            time_embed = time_embedding(dates, self.nbts, self.embed_dim, tau=self.tau)
        if self.patch_embedding == 'swin' or self.patch_embedding == 'conv2d':
            if self.pe_skip:
                for i in range(self.nbts):
                    embedded_cur, list_skip_pe_cur = self.patch_embed_fct(x[:, i, :, :, :])
                    if i == 0:
                        embedded = [embedded_cur]
                        list_skip_pe = [list_skip_pe_cur]
                    else:
                        embedded.append(embedded_cur)
                        list_skip_pe.append(list_skip_pe_cur)
                    self.patch_embed_fct(x[:, :, i, :, :])
            else: embedded = [self.patch_embed_fct(x[:, :, i, :, :]) for i in range(self.nbts)]
            x = torch.stack(embedded, 1)
        elif self.patch_embedding == 'conv3d':
            if self.pe_skip: embedded, list_skip_pe = self.patch_embed_fct(x)
            else: embedded = self.patch_embed_fct(x)
            embedded = [embedded[:, :, i, :, :] for i in range(self.nbts)]
            x = torch.stack(embedded, 1)

        Wh, Ww = x.size(-2), x.size(-1)
        x = x.flatten(-2).transpose(-2, -1)

        if self.use_te:
            time_embed = torch.stack([time_embed[:, i, :].unsqueeze(1).repeat(1, Wh * Ww, 1) for i in range(self.nbts)],
                                     1)
            time_embed = time_embed.to(x.device)
            x = x + time_embed

        x = self.pos_drop(x)

        outs = []
        outs_t_skip = []
        outs_sp_skip = []
        for i in range(self.num_layers):
            layer = self.layers[i]
            if self.ST_light:
                x_out, H, W, x, Wh, Ww, x_t, x_sp = layer(x, Wh, Ww)
                x_t = x_t.view(-1, self.nbts, H, W, self.num_features[i]).permute(0,1,4,2,3).contiguous() # x_t vorher: BxTxNxC --> BxTxCxHxW
                x_sp = x_sp.view(-1, self.nbts, H, W, self.num_features[i]).permute(0, 1, 4, 2, 3).contiguous()  # x_t vorher: BxTxNxC --> BxTxCxHxW
                outs_t_skip.append(x_t)
                outs_sp_skip.append(x_sp)
            else: x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, self.nbts, H, W,self.num_features[i]).permute(0, 1, 4, 2, 3).contiguous()
                outs.append(out)

        if self.ST_light == False and self.pe_skip==False: return outs
        elif self.ST_light == False and self.pe_skip==True: return outs, list_skip_pe
        elif self.ST_light and self.pe_skip==False: return outs, outs_t_skip, outs_sp_skip
        else: return outs, outs_t_skip, outs_sp_skip, list_skip_pe