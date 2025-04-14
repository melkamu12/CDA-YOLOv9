import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter

class ShuffleAttention(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        print(f"🔀 ShuffleAttention Forward | Input shape: {x.shape}")
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        print(f"🔀 ShuffleAttention Forward | Output shape: {out.shape}")
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        #2*h*w
        x = self.conv(x)
        #1*h*w
        return self.sigmoid(x)


class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        print(f"🧠 CBAM Forward | Input shape: {x.shape}")
        out = self.channel_attention(x) * x
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(out) * out
        print(f"🔀 ShuffleAttention Forward | Output shape: {out.shape}")
        return out
    
######################################################################################

import logging
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import create_model
try:
    from timm.layers import create_conv2d, create_pool2d, get_act_layer
except ImportError:
    from timm.models.layers import create_conv2d, create_pool2d, get_act_layer
from .anchors import get_feat_sizes
from .config import get_fpn_config, set_config_writeable, set_config_readonly

_USE_SCALE = False
_ACT_LAYER = get_act_layer('silu')

class SequentialList(nn.Sequential):
    """ This module exists to work around torchscript typing issues list -> list"""
    def __init__(self, *args):
        super(SequentialList, self).__init__(*args)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        for module in self:
            x = module(x)
        return x

class Interpolate2d(nn.Module):

    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']
    name: str
    size: Optional[Union[int, Tuple[int, int]]]
    scale_factor: Optional[Union[float, Tuple[float, float]]]
    mode: str
    align_corners: Optional[bool]

    def __init__(
            self,
            size: Optional[Union[int, Tuple[int, int]]] = None,
            scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
            mode: str = 'nearest',
            align_corners: bool = False,
    ) -> None:
        super(Interpolate2d, self).__init__()
        self.name = type(self).__name__
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = None if mode == 'nearest' else align_corners

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
            recompute_scale_factor=False,
        )
    
class ConvBnAct2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            padding='',
            bias=False,
            norm_layer=nn.BatchNorm2d,
            act_layer=_ACT_LAYER,
    ):
        super(ConvBnAct2d, self).__init__()
        self.conv = create_conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            bias=bias,
        )
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class SeparableConv2d(nn.Module):
    """ Separable Conv
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            dilation=1,
            padding='',
            bias=False,
            channel_multiplier=1.0,
            pw_kernel_size=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=_ACT_LAYER,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv_dw = create_conv2d(
            in_channels,
            int(in_channels * channel_multiplier),
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding,
            depthwise=True,
        )
        self.conv_pw = create_conv2d(
            int(in_channels * channel_multiplier),
            out_channels,
            pw_kernel_size,
            padding=padding,
            bias=bias,
        )
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class ResampleFeatureMap(nn.Sequential):

    def __init__(
            self,
            in_channels,
            out_channels,
            input_size,
            output_size,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=nn.BatchNorm2d,
            apply_bn=False,
            redundant_bias=False,
    ):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.output_size = output_size

        if in_channels != out_channels:
            self.add_module(
                'conv',
                ConvBnAct2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    padding=pad_type,
                    norm_layer=norm_layer if apply_bn else None,
                    bias=not apply_bn or redundant_bias,
                    act_layer=None,
                )
            )

        if input_size[0] > output_size[0] and input_size[1] > output_size[1]:
            if downsample in ('max', 'avg'):
                stride_size_h = int((input_size[0] - 1) // output_size[0] + 1)
                stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                if stride_size_h == stride_size_w:
                    kernel_size = stride_size_h + 1
                    stride = stride_size_h
                else:
                    # FIXME need to support tuple kernel / stride input to padding fns
                    kernel_size = (stride_size_h + 1, stride_size_w + 1)
                    stride = (stride_size_h, stride_size_w)
                down_inst = create_pool2d(downsample, kernel_size=kernel_size, stride=stride, padding=pad_type)
            else:
                if _USE_SCALE:  # FIXME not sure if scale vs size is better, leaving both in to test for now
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    down_inst = Interpolate2d(scale_factor=scale, mode=downsample)
                else:
                    down_inst = Interpolate2d(size=output_size, mode=downsample)
            self.add_module('downsample', down_inst)
        else:
            if input_size[0] < output_size[0] or input_size[1] < output_size[1]:
                if _USE_SCALE:
                    scale = (output_size[0] / input_size[0], output_size[1] / input_size[1])
                    self.add_module('upsample', Interpolate2d(scale_factor=scale, mode=upsample))
                else:
                    self.add_module('upsample', Interpolate2d(size=output_size, mode=upsample))


class FpnCombine(nn.Module):
    def __init__(
            self,
            feature_info,
            fpn_channels,
            inputs_offsets,
            output_size,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=nn.BatchNorm2d,
            apply_resample_bn=False,
            redundant_bias=False,
            weight_method='attn',
    ):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            self.resample[str(offset)] = ResampleFeatureMap(
                feature_info[offset]['num_chs'],
                fpn_channels,
                input_size=feature_info[offset]['size'],
                output_size=output_size,
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
            )

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out


class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class BiFpnLayer(nn.Module):
    def __init__(
            self,
            feature_info,
            feat_sizes,
            fpn_config,
            fpn_channels,
            num_levels=5,
            pad_type='',
            downsample=None,
            upsample=None,
            norm_layer=nn.BatchNorm2d,
            act_layer=_ACT_LAYER,
            apply_resample_bn=False,
            pre_act=True,
            separable_conv=True,
            redundant_bias=False,
    ):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        # fill feature info for all FPN nodes (chs and feat size) before creating FPN nodes
        fpn_feature_info = feature_info + [
            dict(num_chs=fpn_channels, size=feat_sizes[fc['feat_level']]) for fc in fpn_config.nodes]

        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            combine = FpnCombine(
                fpn_feature_info,
                fpn_channels,
                tuple(fnode_cfg['inputs_offsets']),
                output_size=feat_sizes[fnode_cfg['feat_level']],
                pad_type=pad_type,
                downsample=downsample,
                upsample=upsample,
                norm_layer=norm_layer,
                apply_resample_bn=apply_resample_bn,
                redundant_bias=redundant_bias,
                weight_method=fnode_cfg['weight_method'],
            )

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels,
                out_channels=fpn_channels,
                kernel_size=3,
                padding=pad_type,
                bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            if pre_act:
                conv_kwargs['bias'] = redundant_bias
                conv_kwargs['act_layer'] = None
                after_combine.add_module('act', act_layer(inplace=True))
            after_combine.add_module(
                'conv', SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))
            
        # Final feature_info used for next BiFPN or Detect head
        self.feature_info = fpn_feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels::]


class BiFpn(nn.Module):

    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config (
            config.fpn_name, min_level=config.min_level, max_level=config.max_level)

        feat_sizes = get_feat_sizes(config.image_size, max_level=config.max_level)
        prev_feat_size = feat_sizes[config.min_level]
        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            feat_size = feat_sizes[level + config.min_level]
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                feature_info[level]['size'] = feat_size
            else:
                # Adds a coarser level by downsampling the last feature map
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    input_size=prev_feat_size,
                    output_size=feat_size,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    apply_bn=config.apply_resample_bn,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                feature_info.append(dict(num_chs=in_chs, size=feat_size))
            prev_feat_size = feat_size

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                feat_sizes=feat_sizes,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                pre_act=not config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x

