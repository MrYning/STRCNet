import torch
import torch.nn as nn


class PreNet(nn.Module):
    def __init__(self, batch, seq_len, grid_len, hid_channel):
        super().__init__()
        self.b = batch
        self.c = hid_channel
        self.t = seq_len
        self.h = grid_len

        self.c_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.t_pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.pool = nn.AdaptiveAvgPool3d((1, 1, None))

        self.conv1 = nn.Conv3d(hid_channel, 1, 1)
        self.conv2 = nn.Conv2d(seq_len, 1, 1)

        # self.net = nn.Sequential(
        #     # nn.LayerNorm(self.h ** 2),
        #     nn.Linear(self.h ** 2, self.h ** 2),
        #     # nn.Dropout(0.2),
        #     # nn.Sigmoid(),     # best
        #     nn.Softplus(),    # good
        #     # nn.GELU(),       # not bad
        #     # nn.ReLU()
        # )
        self.linearMap = nn.Sequential(
            nn.LayerNorm(self.h ** 2),
            nn.Linear(self.h ** 2, self.h ** 2),
            nn.Softplus()
        )
        # self.linearMap1 = nn.Sequential(
        #     nn.BatchNorm2d(1),
        #     nn.Linear(self.h, self.h),
        #     nn.Softplus()
        # )
        # self.linearMap2 = nn.Sequential(
        #     nn.LayerNorm(self.h ** 2),
        #     nn.Linear(self.h ** 2, self.h ** 2),
        #     nn.ReLU()
        # )

    def forward(self, x):
        # x = x.view(self.b, self.c, self.t, self.h, self.h).contiguous()
        # # Plan A
        # x = self.c_pool(self.t_pool(x).squeeze()).squeeze()
        # # out = self.net(x.view(self.b, -1).contiguous())
        #
        # # Plan B
        # # x = self.conv2(self.conv1(x).squeeze()).squeeze()
        # out = self.linearMap(x.view(self.b, -1))

        # Plan C
        x = x.view(self.b, self.c, self.t, -1).contiguous()
        out = self.linearMap(self.pool(x).squeeze())

        # Plan D
        # x = x.view(self.b, self.c, self.t, -1).contiguous()
        # out = self.linearMap1(self.pool(x).squeeze().unsqueeze(1).contiguous().view(self.b, 1, self.h, self.h))
        # return out

        return out.unsqueeze(1).view(self.b, 1, self.h, self.h)


if __name__ == "__main__":
    input_tensor = torch.randn(8, 48, 7, 20, 20).cuda()
    model = PreNet(8, 7, 20, 48).cuda()
    # 前向传播
    output = model(input_tensor)
    print("输出张量形状:", output.shape)
