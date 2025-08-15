import torch
import torch.nn as nn
from .SpatialNet import SpatialNet
from .TemporalNet import TemporalNet
from .Transfer import CrossTransfer, FusionTransfer
from .PreNet import PreNet
import torch.nn.functional as F


class KLDivLoss2(nn.Module):
    def __init__(self, hid_channels, temp=4.0):
        super().__init__()
        self.temp = temp
        self.linearMap = nn.Linear(1, hid_channels)

    def forward(self, spatial_weight, temporal_out):
        spatial_ = self.linearMap(spatial_weight.unsqueeze(3)).view(temporal_out.shape)
        soft = F.softmax(temporal_out / self.temp, dim=-1)
        log_soft = F.log_softmax(spatial_ / self.temp, dim=-1)
        loss = F.kl_div(log_soft, soft, reduction='sum') * (self.temp ** 2)
        # return loss, [soft, log_soft]
        return loss, [soft, F.softmax(spatial_ / self.temp, dim=-1)]


class KLDivLoss(nn.Module):
    def __init__(self, hid_channels, temp=4.0):
        super().__init__()
        self.temp = temp
        self.conv = nn.Conv3d(hid_channels, 1, 1)

    def forward(self, spatial_weight, temporal_out):
        temporal_weight = self.conv(temporal_out).squeeze().view(spatial_weight.shape)
        soft = F.softmax(temporal_weight / self.temp, dim=-1)
        log_soft = F.log_softmax(spatial_weight / self.temp, dim=-1)
        loss = F.kl_div(log_soft, soft, reduction='sum') * (self.temp ** 2)
        return loss


class STRCNet(nn.Module):
    def __init__(
            self,
            batch,
            in_channels,
            hid_channels,
            seq_len,
            att_heads,
            dim_head,
            mlp_dim,
            temp=4.0,
            grid_len=20
    ):
        super().__init__()
        self.LN = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hid_channels),
            nn.GELU()
        )

        self.SNet = SpatialNet(hid_channels, hid_channels)
        self.TNet = TemporalNet(hid_channels, seq_len, att_heads, dim_head, mlp_dim)
        self.CTNet = CrossTransfer(hid_channels)
        self.FusionNet = FusionTransfer(seq_len, hid_channels)
        self.PNet = PreNet(batch, seq_len, grid_len, hid_channels)

        self.convf = nn.Conv3d(hid_channels*2, hid_channels, 1)
        self.convc = nn.Conv3d(hid_channels*2, hid_channels, 1)

        self.distill_loss_calculater = KLDivLoss2(hid_channels, temp)

    def forward(self, xf, xc):
        # nyc input shape: b t c h w
        xf = xf.permute(0, 1, 3, 4, 2)
        xc = xc.permute(0, 1, 3, 4, 2)
        xf = self.LN(xf).permute(0, 4, 1, 2, 3)
        xc = self.LN(xc).permute(0, 4, 1, 2, 3)

        # input shape: b c t h w
        for i in range(3):
            flag = 'stack'
            if i == 0:
                flag = 'no_stack'
            xf, wf = self.SNet(xf, flag)
            view_spatial = xf
            xc, wc = self.SNet(xc, flag)
        xf1, xc1 = self.CTNet(xf, xc)
        for j in range(1):
            xf = self.TNet(xf)
            xc = self.TNet(xc)

        """
            Calculate distill loss!!!
        """
        view_temporal = xf
        distill_loss_f, view_map_ls_f = self.distill_loss_calculater(wf, xf)
        distill_loss_c, view_map_ls_c = self.distill_loss_calculater(wc, xc)

        xf = self.convf(torch.cat((xf1, xf), dim=1))
        xc = self.convf(torch.cat((xc1, xc), dim=1))

        out = self.FusionNet(xf, xc)
        out = self.PNet(out)
        return out, distill_loss_f + distill_loss_c, [view_map_ls_f, [view_spatial, view_temporal]]


if __name__ == "__main__":
    input_tensor = torch.randn(8, 7, 48, 20, 20).cuda()
    input_tensor2 = torch.randn(8, 7, 48, 10, 10).cuda()
    model = STRCNet(8, 48, 128, 7, 8, 16, 16).cuda()
    # 前向传播
    output, loss = model(input_tensor, input_tensor2)
    print("输出张量形状:", output.shape)
    print("loss:", loss)
