import torch
from torch import nn


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
