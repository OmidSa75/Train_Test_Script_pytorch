from torchvision.datasets import ImageFolder
import os
from models import ConvAutoEncoder, VAE
from utils import Utils, TransForms
from traintest.traintest_autoencoder import VAETrainTest
from config import config
from torchvision.datasets import MNIST

if __name__ == '__main__':
    tfms = TransForms(config.img_size)
    utils = Utils()
    # train_dataset = ImageFolder(os.path.join('breast_cancer', 'train'), transform=tfms.train_tfms)
    # test_dataset = ImageFolder(os.path.join('breast_cancer', 'test'), transform=tfms.test_tfms)
    train_dataset = MNIST('MNIST', train=True, transform=tfms.simple_tfms, download=True)
    test_dataset = MNIST('MNIST', train=False, transform=tfms.simple_tfms, download=True)
    model = VAE()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of train images: {}\nNumber of test images: {}\nNumber of model trainable parameters: {}".format(
        len(train_dataset.data), len(test_dataset.data), num_params
    ))
    train_test = VAETrainTest(config, model, train_dataset, test_dataset, utils)
    train_test.train()
