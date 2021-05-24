import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from collections import OrderedDict


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


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = self._block(in_channels, features, name='encoder1')
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2, name='encoder2')
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4, name='encoder3')
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self.encoder4(features * 4, features * 8, name='encoder4')
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=(2, 2), stride=(2, 2))
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="decoder4")

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=(2, 2), stride=(2, 2))
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="decoder3")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=(2, 2), stride=(2, 2))
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="decoder2")

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=(2, 2), stride=(2, 2))
        self.decoder1 = self._block(features * 2, features, name="decoder1")

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "conv1",
                     nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=(3, 3), padding=(1, 1),
                               bias=False),),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "conv2",
                     nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=(3, 3), padding=(1, 1),
                               bias=False, ),),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
