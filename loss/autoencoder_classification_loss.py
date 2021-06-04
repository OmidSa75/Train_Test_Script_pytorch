import torch
from torch import nn


def gaussian_likelihood(recon_x, logscale, x):
    scale = torch.exp(logscale)
    mean = recon_x
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))


def kl_divergence(z, mu, std):
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum((1, 2, 3))
    return kl


class VAEClsLoss(nn.Module):
    def __init__(self):
        super(VAEClsLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, recon_x, x, z, mu, std, logscale, pred_classes, labels):
        cls_loss = self.cross_entropy(pred_classes, labels)

        recon_loss = gaussian_likelihood(recon_x, logscale, x)
        kl = kl_divergence(z, mu, std)

        # elbo
        elbo = kl - recon_loss
        elbo = elbo.mean()

        return elbo + cls_loss, elbo, cls_loss
