import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class MSTemporalInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 定义不同尺度的卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0))
        self.conv5 = nn.Conv3d(in_channels, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0))
        self.conv7 = nn.Conv3d(in_channels, out_channels, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        # 定义池化层
        self.pool = nn.MaxPool3d(kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        # 定义融合卷积层
        self.fuse_conv = nn.Conv3d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        # 不同尺度的卷积操作
        feat1 = self.pool(self.conv1(x))
        feat3 = self.pool(self.conv3(x))
        feat5 = self.pool(self.conv5(x))
        feat7 = self.pool(self.conv7(x))
        # 特征融合
        fused = torch.cat([feat1, feat3, feat5, feat7], dim=1)
        fused = self.fuse_conv(fused)
        return fused


class FeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.view(-1, c, t)
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x))))).view(b, c, t, h, w)


class Attention(nn.Module):
    """
    dim: time period.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., apply_transform=False, transform_scale=True, knn_attention=0.7):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.apply_transform = apply_transform
        self.knn_attention = bool(knn_attention)
        self.topk = knn_attention

        if apply_transform:
            self.reatten_matrix = torch.nn.Conv2d(heads, heads, 1, 1)
            self.var_norm = torch.nn.BatchNorm2d(heads)
            self.reatten_scale = self.scale if transform_scale else 1.0

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        self.scores = None

    def forward(self, x):
        """

        """
        bb, cc, tt, hh, ww = x.shape
        x = x.view(-1, cc, tt)
        b, n, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if self.knn_attention:
            mask = torch.zeros(b, self.heads, n, n, device=x.device, requires_grad=False)
            index = torch.topk(dots, k=int(dots.size(-1)*self.topk), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            dots = torch.where(mask > 0, dots, torch.full_like(dots, float('-inf')))
        attn = dots.softmax(dim=-1)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale

        self.scores = attn
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out.reshape(bb, cc, tt, hh, ww)


class TemporalNet(nn.Module):
    def __init__(
            self,
            in_channels,
            seq_len,
            heads,
            dim_head,
            mlp_dim,
            dropout=0.,
            apply_transform=False,
            knn_attention=0.7
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, in_channels), requires_grad=True)
        self.mst = MSTemporalInception(in_channels, in_channels)
        self.attention = Attention(seq_len, heads, dim_head, dropout, apply_transform, knn_attention=knn_attention)
        self.mlp = FeedForward(seq_len, mlp_dim, dropout)

    def forward(self, x):
        b, c, t, h, w = x.shape
        # x = x.view(-1, t, c)
        # x = x + self.pos_embedding
        # x = x.view(b, c, t, h, w)
        x = self.mst(x)
        # Plan B:
        x = x.view(-1, t, c)
        x = x + self.pos_embedding
        x = x.view(b, c, t, h, w)
        res = x.clone()
        x = self.attention(x) + res
        res = x.clone()
        x = self.mlp(x) + res
        return x


if __name__ == "__main__":
    input_tensor = torch.randn(8, 48, 7, 20, 20).cuda()
    model = TemporalNet(48, 7, 8, 16, 32).cuda()
    # 前向传播
    output = model(input_tensor)
    print("输出张量形状:", output.shape)
