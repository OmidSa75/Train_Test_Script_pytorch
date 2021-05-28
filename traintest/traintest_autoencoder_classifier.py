import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.utils import save_image
import os
from tqdm import tqdm

from loss import VAEClsLoss


class VAEClsTrainTest:
    def __init__(self, args, model: torch.nn.Module, train_dataset: Dataset, test_dataset: Dataset, utils):
        self.utils = utils
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.batch_size = self.args.batch_size
        self.img_size = self.args.img_size

        self.model = model.to(self.device)

        os.makedirs(os.path.join(self.args.ckpt_dir, self.model.name), exist_ok=True)
        os.makedirs(self.args.save_gen_images_dir, exist_ok=True)

        ''' optimizer '''

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        '''dataset and dataloader'''
        self.train_dataset = train_dataset
        weights = self.utils.make_weights_for_balanced_classes(self.train_dataset.imgs, len(self.train_dataset.classes))
        weights = torch.DoubleTensor(weights)
        sampler = WeightedRandomSampler(weights, len(weights))

        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size,
                                           num_workers=args.num_worker, sampler=sampler,
                                           pin_memory=True)

        self.test_dataset = test_dataset
        self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, num_workers=args.num_worker,
                                          pin_memory=True)

        '''loss function'''
        self.criterion = VAEClsLoss().to(self.device)

        '''scheduler'''
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def train(self):
        num_epoch = self.args.num_epochs

        for epoch in range(1, num_epoch + 1):
            self.model.train(True)

            train_loss = 0.0
            train_acc = 0.0
            train_cls_loss = 0.0
            train_vae_loss = 0.0

            for data, labels in tqdm(self.train_dataloader, desc="Training"):
                data, labels = data.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                recon_batch, mu, logvar, pred_classes = self.model(data)
                total_loss, vae_loss, cls_loss = self.criterion(recon_batch, data, mu, logvar, pred_classes, labels)
                total_loss.backward()
                self.optimizer.step()

                train_loss += total_loss.data
                train_acc += self.utils.calc_acc(pred_classes, labels)
                train_vae_loss += vae_loss.data
                train_cls_loss += cls_loss.data

            epoch_loss = train_loss / len(self.train_dataloader)
            epoch_acc = train_acc / len(self.train_dataloader)
            epoch_vae_loss = train_vae_loss / len(self.train_dataloader)
            epoch_cls_loss = train_cls_loss / len(self.train_dataloader)

            print("\n\033[0;32mEpoch: {} [Train Loss: {:.4f}] [Train Acc: {:.4f}]"
                  "\n[VAE Loss: {:.4f}] [Cls Loss: {:.4f}]\033[0;0m".format(epoch, epoch_loss, epoch_acc, epoch_vae_loss, epoch_cls_loss))

            if epoch % self.args.save_gen_images == 0:
                save_imgs = self.utils.to_img(recon_batch.cpu().data, self.args.img_size)
                save_image(save_imgs, os.path.join(self.args.save_gen_images_dir, f'epoch_{epoch}.jpg'))

            if epoch % self.args.save_iteration == 0:
                torch.save(self.model.state_dict(),
                           os.path.join(self.args.ckpt_dir, self.model.name, f'ckpt_{epoch}.pth'))

            if epoch % self.args.test_iteration == 0:
                self.test()

            self.scheduler.step(epoch_loss)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_vae_loss = 0.0
        test_cls_loss = 0.0

        for data, labels in tqdm(self.test_dataloader, desc="---Testing---"):
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            recon_batch, mu, logvar, pred_classes = self.model(data)
            total_loss, vae_loss, cls_loss = self.criterion(recon_batch, data, mu, logvar, pred_classes, labels)

            test_loss += total_loss.data
            test_acc += self.utils.calc_acc(pred_classes, labels)
            test_vae_loss += vae_loss.data
            test_cls_loss += cls_loss.data

        epoch_loss = test_loss / len(self.test_dataloader)
        epoch_acc = test_acc / len(self.test_dataloader)
        epoch_vae_loss = test_vae_loss / len(self.test_dataloader)
        epoch_cls_loss = test_cls_loss / len(self.test_dataloader)

        print("\n\033[1;34m** Test: [Test Loss: {:.4f}] [Test Acc: {:.4f}]"
              "\n[VAE Loss: {:.4f}] [Cls Loss: {:.4f}]**\033[0;0m".format(epoch_loss, epoch_acc, epoch_vae_loss, epoch_cls_loss))
