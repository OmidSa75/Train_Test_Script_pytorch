from torchvision.datasets import ImageFolder
import os
from models import VAEClsConv
from utils import Utils, TransForms
from traintest import VAEClsTrainTest
from config import config

if __name__ == '__main__':
    tfms = TransForms(config.img_size)
    utils = Utils()
    train_dataset = ImageFolder(os.path.join('breast_cancer', 'train'), transform=tfms.train_tfms)
    test_dataset = ImageFolder(os.path.join('breast_cancer', 'test'), transform=tfms.test_tfms)
    model = VAEClsConv()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of train images: {}\nNumber of test images: {}\nNumber of model trainable parameters: {}".format(
        len(train_dataset.imgs), len(test_dataset.imgs), num_params
    ))
    train_test = VAEClsTrainTest(config, model, train_dataset, test_dataset, utils)
    train_test.train()
