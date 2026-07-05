import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.block import C2PSA, PSABlock

__all__ = ("DynamicTanh", "C2TSSA_DYT_Mona")


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x
        x = (self.conv1(x) + self.conv2(x) + self.conv3(x)) / 3.0 + identity
        return x + self.projector(x)


class Mona(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.project1 = nn.Conv2d(in_dim, 64, 1)
        self.project2 = nn.Conv2d(64, in_dim, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.adapter_conv = MonaOp(64)
        self.norm = LayerNorm2d(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.norm(x) * self.gamma + x * self.gammax
        x = self.project1(x)
        x = self.adapter_conv(x)
        x = self.dropout(F.gelu(x))
        return identity + self.project2(x)


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            return x * self.weight + self.bias
        return x * self.weight[:, None, None] + self.bias[:, None, None]

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"alpha_init_value={self.alpha_init_value}, channels_last={self.channels_last}"
        )


class AttentionTSSA(nn.Module):
    """Token statistics self-attention used by C2TSSA_DYT_Mona."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.heads = num_heads
        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.temp = nn.Parameter(torch.ones(num_heads, 1))
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(proj_drop))

    def forward(self, x):
        b, n, c = x.shape
        w = self.qkv(x).view(b, n, self.heads, c // self.heads).permute(0, 2, 1, 3)
        w_normed = F.normalize(w, dim=-2)
        pi = self.attend(torch.sum(w_normed**2, dim=-1) * self.temp)
        dots = torch.matmul((pi / (pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w**2)
        attn = self.attn_drop(1.0 / (1.0 + dots))
        out = -torch.mul(w.mul(pi.unsqueeze(-1)), attn)
        out = out.permute(0, 2, 1, 3).reshape(b, n, c)
        return self.to_out(out)


class TSSABlockDYTMona(PSABlock):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__(c, attn_ratio, num_heads, shortcut)
        self.dyt1 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.dyt2 = DynamicTanh(normalized_shape=c, channels_last=False)
        self.mona1 = Mona(c)
        self.mona2 = Mona(c)
        self.attn = AttentionTSSA(c, num_heads=num_heads)

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.dyt1(x).flatten(2).permute(0, 2, 1)
        y = self.attn(y).permute(0, 2, 1).view(b, c, h, w).contiguous()
        x = x + y if self.add else y
        x = self.mona1(x)
        y = self.ffn(self.dyt2(x))
        x = x + y if self.add else y
        return self.mona2(x)


class C2TSSA_DYT_Mona(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.m = nn.Sequential(
            *(TSSABlockDYTMona(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n))
        )
