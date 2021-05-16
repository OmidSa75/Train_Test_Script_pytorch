import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
from tqdm import tqdm


class TrainTest:
    def __init__(self, args, model: torch.nn.Module, train_dataset: Dataset, test_dataset: Dataset, utils):
        self.utils = utils
        self.args = args
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.batch_size = self.args.batch_size
        self.img_size = self.args.img_size

        self.model = model.to(self.device)

        os.makedirs(os.path.join(self.args.ckpt_dir, self.model.name), exist_ok=True)

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
        self.criterion = torch.nn.BCELoss().to(self.device)

        '''scheduler'''
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3)

    def train(self):
        num_epoch = self.args.num_epochs

        for epoch in range(1, num_epoch + 1):
            self.model.train(True)

            train_loss = 0.0

            for data, _ in tqdm(self.train_dataloader, desc="Training"):
                data = data.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(data)
                loss = self.criterion(outputs, data)
                loss.backward()
                self.optimizer.step()

                train_loss += loss

            epoch_loss = train_loss / len(self.train_dataloader)

            print("\n\033[0;32mEpoch: {} [Train Loss: {:.4f}\033[0;0m".format(epoch, epoch_loss))

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

        for data, _ in tqdm(self.test_dataloader, desc="*Testing*"):
            data = data.to(self.device)
            outputs = self.model(data)
            loss = self.criterion(outputs, data)

            test_loss += loss

        total_loss = test_loss / len(self.test_dataloader)

        print("\n\033[1;34m** Test: [Test Loss: {:.4f}]**\033[0;0m".format(total_loss))
