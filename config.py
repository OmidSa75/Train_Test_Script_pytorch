from easydict import EasyDict
import os

config = EasyDict()
config.lr = 0.01
config.batch_size = 32
config.num_epochs = 200
config.num_worker = 0
config.save_iteration = 20
config.ckpt_dir = os.path.join('checkpoints')
config.img_size = (28, 28)
config.test_iteration = 5

'''AutoEncoder Setting'''
config.save_gen_images_dir = 'generated_images'
config.save_gen_images = 10  # after this epochs , save the generated images.
