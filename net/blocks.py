import math
from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
from torch.nn.modules.container import Sequential
from einops import rearrange 
from einops.layers.torch import Rearrange, Reduce
from einops import repeat

from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self,seq_length=256, embedding_dim=512):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #8x
        #print(seq_length,embedding_dim)

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings

#Mamba
class Spatial_Mamba_block(nn.Module):
    def __init__(self, input_dim, output_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
                d_model=input_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale= nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]  #batch,channel
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel() #h*w
        img_dims = x.shape[2:]   #h,w
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2) #交换后两个维度的顺序，也就是交换C和n_tokens的顺序
        x_norm = self.norm(x_flat)  #正则化
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat  #经过mamba层处理后，加上经过scale的残差
        x_mamba = self.norm(x_mamba)  #层归一化
        x_mamba = self.proj(x_mamba)   #MLP
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)  #B,output_dim,H,W
        return out







class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#通达注意力
class Channel_Mamba_layer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, 1, padding=0, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_model,
            out_channels=self.d_inner,
            groups=self.d_model,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        # self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, 1, padding=0, bias=bias, **factory_kwargs)

        self.out_act = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None


        self.pool = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 2
        x_hwwh = x.view(B, 1, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 1], dims=[-1]).view(B, 1, -1, L)

        return out_y[:, 0], inv_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        # 池化
        x1 = self.pool(x.permute(0, 3, 1, 2))

        x1 = x1.contiguous()
        x1 = self.conv2d(x1)
        x1 = self.act(x1)
        y1, y2 = self.forward_core(x1)
        assert y1.dtype == torch.float32
        y = y1 + y2
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, 1, 1, -1)
        # y = self.out_norm(y)
        out = self.out_proj(y.permute(0, 3, 1, 2))

        out = self.out_act(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = out*x.permute(0, 3, 1, 2)
        return out.permute(0, 2, 3, 1)

# 通道上的mamba
class Channel_Mamba_block(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.ln_1 = nn.LayerNorm(input_dim)
        # self.mamba  = SCMambaBlock(hidden_dim=dim)
        self.mamba  = Channel_Mamba_layer(d_model=input_dim, d_state=16,expand=2,dropout=0)
        self.ln_2 = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)
        self.ca = CAB(input_dim)
        self.skip_scale1= nn.Parameter(torch.ones(1))
        self.skip_scale2= nn.Parameter(torch.ones(1))

    def forward(self, x):

        x1 = self.ln_1(self.proj(x.permute(0, 2, 3, 1)))
        x1 = self.mamba(x1)+self.skip_scale1*x.permute(0, 2, 3, 1)
        x2 = self.ln_2(x1)
        x2 = self.ca(x2.permute(0, 3, 1, 2)) + self.skip_scale1*x1.permute(0, 3, 1, 2)
        return x2







class CS_Block(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path, H,W):
        """ FWSA and Mamba_Block
        """
        super(CS_Block, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.drop_path = drop_path
        #self.input_resolution = input_resolution
        self.H=H
        self.W=W




        
        self.channel_block = Channel_Mamba_block(input_dim=self.in_channels, output_dim=self.in_channels)
        self.spatial_block = Spatial_Mamba_block(input_dim=self.in_channels, output_dim=self.in_channels)




        self.conv1_1 = nn.Conv2d(self.in_channels, self.in_channels*2, 1, 1, 0, bias=True)
        self.conv1_2 = nn.Conv2d(self.in_channels*2, self.out_channels, 1, 1, 0, bias=True)



    def forward(self, x):

        channel_x, spatial_x = torch.split(self.conv1_1(x), (self.in_channels, self.in_channels), dim=1)# B,C,H,W
        channel_x=self.channel_block(channel_x)+channel_x
        spatial_x=self.spatial_block(spatial_x)+spatial_x
        #trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        #print(trans_x.shape)
        #trans_x = self.trans_block(trans_x)
        #trans_x = Rearrange('b h w c -> b c h w')(trans_x)
        res = self.conv1_2(torch.cat((spatial_x, channel_x), dim=1))
        x = x + res

        return x

































