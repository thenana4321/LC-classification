# code/model.py

"""
Definition of the Multi-Temporal segmentation model using a backbone based on self-attention and convolution
and a decoder head based on convolutions.
"""

import torch.nn as nn
import torch.nn.functional as F

from decoder_heads import UPerHead_MultiOut
from backbone import MT_encoder

class MultiTemporal_Model(nn.Module):
    """
        Main model for multitemporal pixelwise classification.
        Combines backbone encoder and decoder head.
        """
    def __init__(self,
                 img_size,
                 patch_size,
                 in_chans,
                 embed_dim,
                 dec_head_dim,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 num_classes=10,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.2,
                 dropout_ratio=0.1,
                 decoder="upernet",
                 use_te=False,
                 tau=10000,
                 nbts=1,
                 ST_light=True,
                 patch_embedding='swin',
                 conv='a',
                 temp_skip=True,
                 nb_CB_PE=2,
                 activation='relu',
                 kernel_size=3,
                 pe_skip=False
                 ):
        """
                Args:
                    img_size (int): Input image size.
                    patch_size (int): Patch size for embedding.
                    in_chans (int): Number of input channels.
                    embed_dim (int): Embedding dimension for backbone.
                    dec_head_dim (int): Decoder head dimension.
                    depths (list): Depths of backbone stages.
                    num_heads (list): Number of attention heads per stage.
                    window_size (int): Window size for attention.
                    mlp_ratio (float): MLP ratio in transformer blocks.
                    num_classes (int): Number of output classes.
                    drop_rate (float): Dropout rate.
                    attn_drop_rate (float): Attention dropout rate.
                    drop_path_rate (float): Drop path rate.
                    dropout_ratio (float): Dropout ratio in decoder.
                    decoder (str): Decoder type.
                    use_te (bool): Use time embedding.
                    tau (int): Time embedding parameter.
                    nbts (int): Number of time steps.
                    ST_light (bool): Use lightweight space-time backbone.
                    patch_embedding (str): Patch embedding type.
                    conv (str): Convolution type.
                    temp_skip (bool): Use temporal skip connections.
                    nb_CB_PE (int): Number of conv blocks in patch embedding.
                    activation (str): Activation function.
                    kernel_size (int): Convolution kernel size.
                    pe_skip (bool): Use patch embedding skip connections.
                """
        super().__init__()

        # Backbone output dimensions per stage
        self.backbone_dims = [embed_dim * 2 ** i for i in range(len(depths))]
        self.img_size = img_size
        self.ptsz = patch_size
        self.nbcl = num_classes
        self.nbts = nbts
        self.temp_skip=temp_skip
        self.ST_light = ST_light
        self.activation = activation
        self.pe_skip = pe_skip
        self.use_te = use_te

        # Initialize backbone
        self.backbone = MT_encoder(img_size=img_size,
                                   patch_size=patch_size,
                                   in_chans=in_chans,
                                   embed_dim=embed_dim,
                                   depths=depths,
                                   num_heads=num_heads,
                                   window_size=window_size,
                                   mlp_ratio=mlp_ratio,
                                   num_classes=num_classes,
                                   nbts=nbts,
                                   use_te=use_te,
                                   tau=tau,
                                   drop_rate=drop_rate,
                                   attn_drop_rate=attn_drop_rate,
                                   drop_path_rate=drop_path_rate,
                                   ST_light = ST_light,
                                   patch_embedding=patch_embedding,
                                   conv=conv,
                                   nb_CB_PE=nb_CB_PE,
                                   activation=self.activation,
                                   kernel_size=kernel_size,
                                   pe_skip = pe_skip
                                          )
        if decoder == 'upernet':
            self.decode_head = UPerHead_MultiOut(
                in_channels=self.backbone_dims,
                channels=dec_head_dim,
                num_classes=num_classes,
                dropout_ratio=dropout_ratio,
                ST_light=ST_light,
                temp_skip=temp_skip,
                activation=self.activation,
                pe_skip=self.pe_skip,
                embed_dim_enc=embed_dim,
                ptsz=patch_size
                )
        else:
            print('other decoders can be implemented here')

    def forward(self, x):
        """
            Forward pass through backbone and decoder.
            Args:
                if use_te: x (List): (data, dates) with dates tensor of shape [B, C, T, H, W],
                           dates tensor of shape [T] with acquisition dates.
                else: x (Tensor): Input tensor of shape [B, C, T, H, W].
            Returns:
                Tensor: Output tensor of shape [B, num_classes, nbts, img_size, img_size]
        """
        sz = x[0].size()[-2:] if self.use_te else x.size()[-2:]

        if not self.ST_light and not self.pe_skip:
            x_enc = self.backbone(x)
            x_dec = self.decode_head(x_enc)
        elif not self.ST_light and self.pe_skip:
            x_enc, list_skip_pe = self.backbone(x)
            x_dec = self.decode_head(x_enc, list_skip_pe= list_skip_pe)
        elif self.ST_light and not self.pe_skip:
            x_enc, x_t, x_sp = self.backbone(x)
            x_dec = self.decode_head(x_enc, x_t, x_sp)
        else:
            x_enc, x_t, x_sp, list_skip_pe = self.backbone(x)
            x_dec = self.decode_head(x_enc, x_t, x_sp, list_skip_pe)

        # Upsample output to original image size
        Hh, Ww = x_dec[0].shape[-2:]
        x_out = F.interpolate(
            x_dec.view(-1, self.nbcl, Hh, Ww), sz, mode='bilinear', align_corners=False
        )
        x_out = x_out.view(-1, self.nbts, self.nbcl, self.img_size, self.img_size)
        x_out = x_out.permute(0, 2, 1, 3, 4)
        return x_out