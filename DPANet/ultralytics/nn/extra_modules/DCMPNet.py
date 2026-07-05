import torch
import torch.nn as nn

from ..modules.conv import Conv

__all__ = ("MFM",)


class MFM(nn.Module):
    """Multi-feature fusion block used by DPANet's detection neck."""

    def __init__(self, inc, dim, reduction=8):
        super().__init__()
        self.height = len(inc)
        hidden = max(int(dim / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden, dim * self.height, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)
        self.conv1x1 = nn.ModuleList(Conv(c, dim, 1) if c != dim else nn.Identity() for c in inc)

    def forward(self, in_feats_):
        in_feats = [layer(in_feats_[idx]) for idx, layer in enumerate(self.conv1x1)]
        b, c, h, w = in_feats[0].shape
        in_feats = torch.cat(in_feats, dim=1).view(b, self.height, c, h, w)
        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(b, self.height, c, 1, 1))
        return torch.sum(in_feats * attn, dim=1)
