from torch import nn
import torch


class ConvActBatNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super(ConvActBatNorm, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class ConvTActBatNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)):
        super().__init__()

        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class MSCNN(nn.Module):
    def __init__(self, in_channels, out_channel, features=8):
        super(MSCNN, self).__init__()

        self.conv3 = ConvActBatNorm(in_channels, features, kernel_size=(1, 1), padding=(0, 0))
        self.conv4 = ConvActBatNorm(in_channels, features, kernel_size=(3, 3), padding=(1, 1))
        self.conv5 = ConvActBatNorm(in_channels, features, kernel_size=(5, 5), padding=(2, 2))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv = ConvActBatNorm(3* features, out_channel, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        conv1 = self.conv3(x)
        conv3 = self.conv4(x)
        conv5 = self.conv5(x)
        cat = torch.cat([conv1, conv3, conv5], dim=1)
        return self.pool(self.conv(self.pool(cat)))
