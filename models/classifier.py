import torch
from torch import nn
from torch.nn import functional as F


class ConvActBatNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvActBatNorm, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Residual, self).__init__()
        self.local_conv = nn.Sequential(
            ConvActBatNorm(in_channel, 4, kernel_size=kernel_size, padding=1),
            nn.MaxPool2d(3, 2)
        )
        self.residual = nn.Sequential(
            ConvActBatNorm(4, 8, kernel_size=1),
            ConvActBatNorm(8, 8, kernel_size=3),
            ConvActBatNorm(8, out_channel, kernel_size=1)
        )

    def forward(self, x):
        x = self.local_conv(x)
        x = self.residual(x)
        return x


class GlobalBranch(nn.Module):
    def __init__(self):
        super(GlobalBranch, self).__init__()
        self.branch1x1 = ConvActBatNorm(8, 4, kernel_size=1)
        self.branch5x5 = nn.Sequential(
            ConvActBatNorm(8, 8, kernel_size=1),
            ConvActBatNorm(8, 4, kernel_size=5, padding=2)
        )

        self.conv = nn.Sequential(
            ConvActBatNorm(8, 16, kernel_size=3),
            ConvActBatNorm(16, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(3, 2)
        )

    def forward(self, x):
        x1 = self.branch1x1(x)
        x2 = self.branch5x5(x)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)

        return out


class AttentionBlock(nn.Module):
    def __init__(self, sample):
        super(AttentionBlock, self).__init__()
        self.name = 'AttentionBlock'
        # Global Feature and Global Score
        self.input_conv = nn.Sequential(
            ConvActBatNorm(3, 4, kernel_size=3, padding=1),
            ConvActBatNorm(4, 8, kernel_size=3, padding=1),
            ConvActBatNorm(8, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(3, 2)
        )

        self.branch = nn.Sequential(
            GlobalBranch(),
            # GlobalBranch(),
            # GlobalBranch(),
        )

        self.Global_features = ConvActBatNorm(8, 4, kernel_size=1, padding=1)
        self.Global_Score = ConvActBatNorm(4, 2, kernel_size=1, padding=1)

        # local features and local score****************************
        self.residual = nn.Sequential(
            Residual(3, 4, kernel_size=3),
            Residual(4, 4, kernel_size=5),
            # Residual(4, 4, kernel_size=3),
            # Residual(4, 4, kernel_size=5),
        )

        self.local_score = nn.Sequential(
            ConvActBatNorm(4, 4, kernel_size=3, padding=1),
            ConvActBatNorm(4, 4, kernel_size=5, padding=2),
            ConvActBatNorm(4, 2, kernel_size=7, padding=3)
        )

    def forward(self, x):
        # Global
        out1 = self.input_conv(x)

        branch = self.branch(out1)
        global_score = self.Global_Score(self.Global_features(branch))

        # Local
        residual = self.residual(x)
        local_score = self.local_score(residual)
        local_score = F.interpolate(local_score, size=(global_score.data[0].shape[1], global_score.data[0].shape[2]),
                                    mode='bilinear', align_corners=False)

        score = local_score + global_score
        out = torch.mean(score, dim=(1, 2))

        return out
