import torch
from torch import nn


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstruction_function = nn.MSELoss(reduction='sum')

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


class VAEClsLoss(nn.Module):
    def __init__(self):
        super(VAEClsLoss, self).__init__()
        self.vae_loss = VAELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, recon_x, x, mu, logvar, pred_classes, labels):
        vae_loss = self.vae_loss(recon_x, x, mu, logvar)
        cls_loss = self.cross_entropy(pred_classes, labels)

        return vae_loss + cls_loss, vae_loss, cls_loss
