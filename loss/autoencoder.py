import torch
from torch import nn


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstruction_function = nn.MSELoss(size_average=False)

    def forward(self, recon_x, x, mu, logvar):
        """

        :param recon_x:  Generated Image.
        :param x: Original Image.
        :param mu: Latent Mean.
        :param logvar: Latent Log Variance.
        :return: BCE + KLD loss
        """
        bce = self.reconstruction_function(recon_x, x)
        kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        kld = torch.sum(kld_element).mul_(-0.5)

        return bce + kld
