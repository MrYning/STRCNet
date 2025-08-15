import torch
import torch.nn as nn
from torch import einsum


class EHMModule(nn.Module):
    """
        Enhance the spatial feature from channel and sequence.
        in_dim: input channels
        heads: time sequence
    """
    def __init__(self, in_dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        # (8, 3, 10) => (8, 3, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        # Self Attention Layers
        self.q = nn.Linear(in_dim, dim_head * self.heads, bias=False)
        self.k = nn.Linear(in_dim, dim_head * self.heads, bias=False)
        self.scale = dim_head ** -0.5
        self.mergefc = nn.Linear(dim_head * self.heads, in_dim)
        self.conv = nn.Conv3d(dim_head * self.heads, in_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

        # sequence enhancement
        self.mlp = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, 32),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(32, 7),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x.shape = b c t h w
        b, c, t, h, w = x.shape
        x = x.reshape(-1, c, h, w)
        # chanel enhancement
        inp = x.clone()
        x = self.avg_pool2d(x).squeeze()
        q = self.q(x).unsqueeze(-1)
        k = self.k(x).unsqueeze(-1)
        dots = self.mergefc(einsum('a i c, a j c -> a i j', q, k) * self.scale)
        A = self.softmax(dots)
        inp = inp.view(b*t, c, -1)
        x = einsum('b i j, b c n -> b i n', A, inp)

        # sequence enhancement
        weight = self.avg_pool(x.permute(0, 2, 1)).squeeze()
        weight = self.mlp(weight.reshape(-1, t)).reshape(b, t, h*w)
        # rc_weight = weight.reshape(b, t, h*w)
        x = x.reshape(b, -1, t, h*w) * weight.unsqueeze(1)
        x = self.conv(x.reshape(b, -1, t, h, w)) + inp.reshape(b, c, t, h, w)
        return x, weight

    # def forward(self, x):
    #     # x.shape = b c t h w
    #     b, c, t, h, w = x.shape
    #     # chanel enhancement
    #     inp = x.clone()
    #     x = self.avg_pool3d(x).squeeze()
    #     q = self.q(x).unsqueeze(-1)
    #     k = self.k(x).unsqueeze(-1)
    #     dots = self.mergefc(einsum('a i c, a j c -> a i j', q, k) * self.scale)
    #     A = self.softmax(dots)
    #     inp = inp.view(b, c, -1)
    #     x = einsum('b i j, b c n -> b i n', A, inp)
    #
    #     # sequence enhancement
    #     weight = self.avg_pool(x.permute(0, 2, 1)).squeeze()
    #     weight = self.mlp(weight.reshape(-1, t)).reshape(b, t, h*w)
    #     # rc_weight = weight.reshape(b, t, h*w)
    #     x = x.reshape(b, -1, t, h*w) * weight.unsqueeze(1)
    #     x = self.conv(x.reshape(b, -1, t, h, w)) + inp.reshape(b, c, t, h, w)
    #     return x, weight


class MSSpatialInception(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 定义不同尺度的卷积层
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv5 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.conv7 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 7, 7), padding=(0, 3, 3))
        # 定义池化层
        self.pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # 定义融合卷积层
        self.fusion = nn.Conv3d(out_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        # 不同尺度的卷积操作
        feat1 = self.pool(self.conv1(x))
        feat3 = self.pool(self.conv3(x))
        feat5 = self.pool(self.conv5(x))
        feat7 = self.pool(self.conv7(x))
        # 特征融合
        fused = torch.cat([feat1, feat3, feat5, feat7], dim=1)
        fused = self.fusion(fused)
        return fused


class SpatialNet(nn.Module):
    def __init__(self, in_channels, hid_channels, heads=7):
        super().__init__()
        self.mssi = MSSpatialInception(in_channels, hid_channels)
        self.ehm = EHMModule(hid_channels, heads, hid_channels//2)

    def forward(self, x, flag):
        if flag == 'stack':
            x, weight = self.ehm(x)
        else:
            x = self.mssi(x)
            x, weight = self.ehm(x)
        return x, weight


if __name__ == "__main__":
    input_tensor = torch.randn(8, 32, 7, 20, 20).cuda()
    model = SpatialNet(32, 32).cuda()
    # 前向传播
    output, w = model(input_tensor, 'stack')
    print("输出张量形状:", output.shape)
    print("输出权重形状:", w.shape)
