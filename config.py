from easydict import EasyDict
import os

config = EasyDict()
config.lr = 0.01
config.batch_size = 32
config.num_epochs = 200
config.num_worker = 2
config.save_iteration = 20
config.ckpt_dir = os.path.join('checkpoints')
config.img_size = (200, 350)
config.test_iteration = 5
