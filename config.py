from easydict import EasyDict
import os

config = EasyDict()
config.lr = 0.1
config.batch_size = 16
config.num_epochs = 200
config.num_worker = 4
config.save_iteration = 20
config.ckpt_dir = os.path.join('checkpoints')
config.img_size = (128, 128)
config.test_iteration = 5

'''AutoEncoder Setting'''
config.save_gen_images_dir = 'generated_images'
config.save_gen_images = 10  # after this epochs , save the generated images.
