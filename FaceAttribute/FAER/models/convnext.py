# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2024/1/2 15:46
# Author     ：XuJ1E
# version    ：python 3.8
# File       : convnext.py
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath


class CCAttention(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.GELU(),
            nn.Linear(dim // 8, dim)
        )
        self.value = nn.Linear(dim, dim)
        self.depthwise = nn.Conv2d(dim, dim,
                                   kernel_size=(kernel_size, kernel_size),
                                   padding=kernel_size // 2, groups=4)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        short = x.clone()
        perm = x.permute(0, 2, 3, 1)
        value = self.value(perm).reshape(b, -1, c)
        pool = self.pool(x).permute(0, 2, 3, 1)
        key = self.mlp(perm) * pool
        key = key.reshape(b, -1, c)
        attn = self.depthwise(x).permute(0, 2, 3, 1)
        attn = self.norm(self.act(attn)).reshape(b, -1, c)
        attn = self.softmax(attn + key) * value
        attn = attn.reshape(b, h, w, c)
        attn = attn.permute(0, 3, 1, 2)
        return attn + short


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.attention = CCAttention(dim=dims[-1], kernel_size=7)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.attention(x)
        return x
        # x0 = self.downsample_layers[0](x)
        # x0 = self.stages[0](x0)   # [1, 128, 56, 56]
        #
        # x1 = self.downsample_layers[1](x0)
        # x1 = self.stages[1](x1)  # [1, 256, 28, 28]
        #
        # x2 = self.downsample_layers[2](x1)
        # x2 = self.stages[2](x2)  # [1, 512, 14, 14]
        #
        # x3 = self.downsample_layers[3](x2)
        # x3 = self.stages[3](x3)  # [1, 1024, 7, 7]
        # return x1, x2, x3

    def forward(self, x):
        feature = self.forward_features(x)
        feature = feature.mean([-2, -1])
        x = self.norm(feature)  # global average pooling, (N, C, H, W) -> (N, C)
        x = self.head(x)
        return feature, x

    def large_parameters(self):
        result = []
        result += [p for p in self.attention.parameters()]
        result += [p for p in self.head.parameters()]
        return result

    def base_parameters(self):
        result = []
        result += [p for p in self.downsample_layers.parameters()]
        result += [p for p in self.stages.parameters()]
        return result


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def repeat_checkpoint(checkpoint, model):
    out_dict = {}
    model_dict = model.state_dict()
    for k, v in checkpoint.items():
        k = k.replace('net.stem.', 'downsample_layers.0.')
        k = re.sub(r'net.stages.([0-9]+).blocks.([0-9]+)', r'stages.\1.\2', k)
        k = re.sub(r'net.stages.([0-9]+).downsample.([0-9]+)', r'downsample_layers.\1.\2', k)
        k = k.replace('conv_dw', 'dwconv')
        k = k.replace('mlp.fc', 'pwconv')
        k = k.replace('net.head.fc.', 'head.')
        if k.startswith('net.head.norm.'):
            k = k.replace('net.head.norm', 'norm')
        # if v.ndim == 2 and 'head' not in k:
        #     model_shape = model.state_dict()[k].shape
        #     v = v.reshape(model_shape)
        out_dict[k] = v
    for k in out_dict.keys():
        if out_dict[k].shape != model_dict[k].shape:
            print('Drop the layer:', str(k))
            continue
        if k in model_dict:
            model_dict[k] = out_dict[k].clone().to(model_dict[k].device)
    return model_dict


def convnext_base(pretrained=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        checkpoint = torch.load(r'.\models\pretrain\convnext_base.pth', map_location='cpu')['state_dict']
        model_dict = repeat_checkpoint(checkpoint, model)
        model.load_state_dict(model_dict, strict=False)
        print('ConvNext Backbone pretrained is True')
    return model


if __name__ == '__main__':
    x = torch.rand((1, 3, 224, 224))
    model = convnext_base(pretrained=True, num_classes=7, drop_path_rate=0.25)
    feature, x = model(x)
    print(model.stages[-1][-1])
    # print(feature.shape, x.shape)
    # torch.Size([1, 256, 28, 28]) torch.Size([1, 512, 14, 14]) torch.Size([1, 1024, 7, 7])
