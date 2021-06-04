import torch
from torch import nn
from .modules import ConvTActBatNorm, ConvActBatNorm, MSCNN


class VAEClsConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "VAECls"
        self.encoder = nn.Sequential(
            ConvActBatNorm(3, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
            ConvActBatNorm(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.Sigmoid()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(8),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            ConvTActBatNorm(8, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            ConvTActBatNorm(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            ConvTActBatNorm(64, 3, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Sigmoid()
        )

        self.classifier_mu = nn.Sequential(
            MSCNN(8, 2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),  # 128 in features for 32 * 32 size and 2 channel
            nn.ReLU()
        )
        self.classifier_logvar = nn.Sequential(
            MSCNN(8, 2),
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(128, 2),  # 128 in features for 32 * 32 size and 2 channel
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=1)

        self.logscale = nn.Parameter(torch.Tensor([0.0]))

    def encode(self, x):
        x = self.encoder(x)
        return self.conv1(x), self.conv2(x)

    def reparametrize(self, mu, logvar: torch.Tensor):
        std = torch.exp(logvar / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return z, std

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z, std = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        cls_mu = self.classifier_mu(mu)
        cls_logvar = self.classifier_logvar(z)
        classes = self.softmax(cls_mu + cls_logvar)
        return recon, mu, z, std, self.logscale, classes
