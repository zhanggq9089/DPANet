import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.conv import Conv, autopad

__all__ = ("DynamicConv", "Dynamic_HGBlock", "DynamicAlignFusion")


class DynamicConvSingle(nn.Module):
    """Per-sample conditional convolution used by DPANet's light HG blocks."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        num_experts=4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.routing = nn.Linear(in_channels, num_experts)
        self.weight = nn.Parameter(
            torch.empty(num_experts, out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(num_experts, out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1] // self.groups
            bound = fan_in**-0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b, c, h, w = x.shape
        routing_weights = torch.sigmoid(self.routing(F.adaptive_avg_pool2d(x, 1).flatten(1)))
        weight = torch.einsum("be,eocij->bocij", routing_weights, self.weight)
        weight = weight.reshape(b * self.out_channels, c // self.groups, *self.kernel_size)
        bias = None
        if self.bias is not None:
            bias = torch.einsum("be,eo->bo", routing_weights, self.bias).reshape(-1)

        x = x.reshape(1, b * c, h, w)
        y = F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=b * self.groups,
        )
        return y.reshape(b, self.out_channels, y.shape[-2], y.shape[-1])


class DynamicConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, num_experts=4):
        super().__init__()
        self.conv = nn.Sequential(
            DynamicConvSingle(
                c1,
                c2,
                kernel_size=k,
                stride=s,
                padding=autopad(k, p, d),
                dilation=d,
                groups=g,
                num_experts=num_experts,
            ),
            nn.BatchNorm2d(c2),
            self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)


class Dynamic_HGBlock(nn.Module):
    """PP-HGNet style block with the dynamic-conv light branch used by DPANet."""

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=True):
        super().__init__()
        block = DynamicConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class DynamicAlignFusion(nn.Module):
    """Two-input adaptive alignment and fusion block used by DPANet."""

    def __init__(self, inc, ouc):
        super().__init__()
        self.conv_align1 = Conv(inc[0], ouc, 1)
        self.conv_align2 = Conv(inc[1], ouc, 1)
        self.conv_concat = Conv(ouc * 2, ouc * 2, 3)
        self.sigmoid = nn.Sigmoid()
        self.x1_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5)
        self.x2_param = nn.Parameter(torch.ones((1, ouc, 1, 1)) * 0.5)
        self.conv_final = Conv(ouc, ouc, 1)

    def forward(self, x):
        self._clamp_abs(self.x1_param.data, 1.0)
        self._clamp_abs(self.x2_param.data, 1.0)

        x1, x2 = x
        x1, x2 = self.conv_align1(x1), self.conv_align2(x2)
        x_concat = self.sigmoid(self.conv_concat(torch.cat([x1, x2], dim=1)))
        x1_weight, x2_weight = torch.chunk(x_concat, 2, dim=1)
        x1, x2 = x1 * x1_weight, x2 * x2_weight
        return self.conv_final(x1 * self.x1_param + x2 * self.x2_param)

    @staticmethod
    def _clamp_abs(data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
