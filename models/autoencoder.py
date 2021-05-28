import torch
from torch import nn
from torch.nn import functional as F
from .modules import ConvActBatNorm, ConvTActBatNorm


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        self.name = 'Simple_AE'
        '''--------------Encoder--------------'''
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5, 5), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 4, kernel_size=(5, 5), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        '''--------------Decoder--------------'''
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, kernel_size=(5, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=(4, 4), stride=(2, 2)),
            nn.Sigmoid()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = 'VAE'
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar: torch.Tensor):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()

        return eps.mul(std).add_(mu)

    def decode(self, z: torch.Tensor):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "VAEConv"
        self.encoder = nn.Sequential(
            ConvActBatNorm(3, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
            ConvActBatNorm(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv1 = ConvActBatNorm(32, 8, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = ConvActBatNorm(32, 8, (3, 3), stride=(1, 1), padding=(1, 1))

        self.decoder = nn.Sequential(
            ConvTActBatNorm(8, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            ConvTActBatNorm(32, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.Upsample(scale_factor=2),
            ConvTActBatNorm(64, 3, (3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        return self.conv1(x), self.conv2(x)

    def reparametrize(self, mu, logvar: torch.Tensor):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()

        return eps.mul(std).add_(mu)

    def decode(self, x):
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        # recon = F.interpolate(recon, size=(28, 28))
        return recon, mu, logvar


