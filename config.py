import string

width = 120
height = 50
label_len = 6

characters = '0123456789qwertyuiopasdfghjklzxcvbnm'
label_classes = len(characters)+1

# save checkpoint path
cp_save_path = './models/weights.{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5' 

# the model for predicting
base_model_path = './models/weights_for_predict.hdf5'  

# TensorBoard save path, Optional
tb_log_dir = './tensorboard'  

load_model_path = ''
learning_rate = 0.01
initial_epoch = 0
is_training = True

img_dir = './data/train'
description_path = './data/train.csv'

# img_dir = './data/public_test'
# description_path = './data/public_test.csv'

# img_dir = './data/private_test'
# description_path = './data/private_test.csv'

N_EPOCHS = 30
BATCH_SIZE = 32
downsample_factor = 4