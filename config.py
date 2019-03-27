import string

width = 150
height = 35
max_label_len = 7

characters = '0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
label_classes = len(characters)+1

# save checkpoint path
cp_save_path = './models/weights.{epoch:02d}-{loss:.3f}-{val_loss:.3f}.hdf5' 

# save model for predicting
base_model_path = './models/weights_for_predict.hdf5'  

# TensorBoard save path
tb_log_dir = './tensorboard'  

load_model_path = ''
learning_rate = 0.01

img_dir = './data/captcha'
description_path = './data/captcha.csv'

N_EPOCHS = 2
BATCH_SIZE = 32
downsample_factor = 4