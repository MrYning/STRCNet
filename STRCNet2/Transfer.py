import torch.nn as nn
import torch
from torch import einsum
import numpy as np


class CrossTransfer(nn.Module):
    def __init__(
            self,
            hid_ch,
            trans_file=r'C:\Users\_zimo\Desktop\lw\STRCNet\STRCNet\Trans.csv'
    ):
        super().__init__()
        trans = np.genfromtxt(trans_file, delimiter=',').astype(np.float32)
        self.trans = torch.tensor(trans).cuda()
        self.trans_f2c = self.trans.t().cuda()
        self.act = nn.ReLU()
        self.W_c2f = nn.Parameter(torch.randn(400, 400), requires_grad=True)
        self.W_f2c = nn.Parameter(torch.randn(100, 100), requires_grad=True)
        self.conv2d = nn.Conv2d(hid_ch*2, hid_ch, 1)
        self.hid_ch = hid_ch

    def forward(self, fg, cg):
        temp_cg = cg.view(-1).view(-1, 100)
        temp = einsum('a b, b c -> a c', self.trans, self.W_c2f)
        trans_feature = einsum('a b, b c -> a c', temp_cg, temp).view(fg.shape)
        conv_temp = torch.cat((fg.contiguous().view(-1, self.hid_ch, 20, 20),
                               trans_feature.contiguous().view(-1, self.hid_ch, 20, 20)), dim=1)
        out_fg = self.conv2d(conv_temp).view(fg.shape)

        temp_fg = fg.view(-1).view(-1, 400)
        temp = einsum('a b, b c -> a c', self.trans_f2c, self.W_f2c)
        trans_feature = einsum('a b, b c -> a c', temp_fg, temp).view(cg.shape)
        conv_temp = torch.cat((cg.contiguous().view(-1, self.hid_ch, 10, 10),
                               trans_feature.contiguous().view(-1, self.hid_ch, 10, 10)), dim=1)
        out_cg = self.conv2d(conv_temp).view(cg.shape)

        return out_fg, out_cg


class FusionTransfer(nn.Module):
    def __init__(
            self,
            seq_len,
            hid_ch,
            trans_file=r'C:\Users\_zimo\Desktop\lw\STRCNet\STRCNet\Trans.csv'
    ):
        super().__init__()
        trans = np.genfromtxt(trans_file, delimiter=',').astype(np.float32)
        self.trans = torch.tensor(trans).cuda()
        self.W = nn.Parameter(torch.randn(400, 400), requires_grad=True)
        self.conv3d = nn.Conv3d(hid_ch*2, hid_ch, 1)
        self.hid_ch = hid_ch
        self.seq_len = seq_len

    def forward(self, fg, cg):
        cg = cg.view(-1).view(-1, 100)
        temp = einsum('a b, b c -> a c', self.trans, self.W)
        trans_feature = einsum('a b, b c -> a c', cg, temp).view(fg.shape)
        conv_temp = torch.cat(
            (
                fg.contiguous().view(-1, self.hid_ch, self.seq_len, 20, 20),
                trans_feature.contiguous().view(-1, self.hid_ch, self.seq_len, 20, 20)
            ),
            dim=1
        )
        out = self.conv3d(conv_temp).view(fg.shape)

        return out


if __name__ == "__main__":
    input_tensor = torch.randn(8, 48, 7, 20, 20).cuda()
    input_tensor2 = torch.randn(8, 48, 7, 10, 10).cuda()
    model = CrossTransfer(48).cuda()
    # 前向传播
    output, w = model(input_tensor, input_tensor2)
    print("输出张量形状:", output.shape)
    print("输出张量形状:", w.shape)

    model = FusionTransfer(7, 48).cuda()
    output = model(input_tensor, input_tensor2)
    print("输出张量形状:", output.shape)
