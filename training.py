import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras import backend as K
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.optimizers import Adadelta, RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from config import *
from model import create_model
from data_generator import CapchaDataGenerator


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_data(description):
    list_files = [] 
    list_labels = []
    f = open(description, "r")
    for i, line in enumerate(f):
        if i > 0:
            fname, label = line[:-1].split(",")
            list_files.append(fname)
            list_labels.append(label)
    return list_files, list_labels

list_files, list_labels = get_data(description_path)
X_train, X_val, y_train, y_val = train_test_split(list_files, list_labels, test_size=0.1, random_state=9)

N_TRAIN_SAMPLES = len(X_train)
N_TEST_SAMPLES = len(X_val)

print("Number of training set:", N_TRAIN_SAMPLES)
print("Number of validation set:", N_TEST_SAMPLES)

# datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=3,  # randomly rotate images in the range (degrees, 0 to 180)
#         # zoom_range=0.03, # Randomly zoom image 
#         # width_shift_range=0.08,  # randomly shift images horizontally (fraction of total width)
#         # height_shift_range=0.01,  # randomly shift images vertically (fraction of total height)
#         # fill_mode="constant",
#         # cval=250
#     )  

training_generator = CapchaDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, is_testing=False)
validation_generator = CapchaDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, is_testing=False)

model, base_model = create_model(is_training=True)

# clipnorm seems to speeds up convergence
optimizer = SGD(lr=learning_rate, decay=1e-6, momentum=0.8, nesterov=True, clipnorm=5)
# optimizer = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
model.summary()  # print a summary representation of your model.
# plot_model(model, to_file='CRNN-CTC-loss.png', show_shapes=True)  # save a image which is the architecture of the model 

checkpoint = ModelCheckpoint(cp_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir=tb_log_dir, write_graph=True, write_images=True)
# lr_reduction = ReduceLROnPlateau(monitor='loss', 
#                                 patience=1, 
#                                 verbose=1, 
#                                 factor=0.5, 
#                                 min_lr=0.001)
if load_model_path != '':
    model.load_weights(load_model_path)

#try your own fit_generator() settings, you may get a better result
model.fit_generator(
        training_generator,
        steps_per_epoch=N_TRAIN_SAMPLES // BATCH_SIZE,
        initial_epoch=0,
        epochs=N_EPOCHS,
        validation_data=validation_generator,
        validation_steps=N_TEST_SAMPLES // BATCH_SIZE,
        verbose=1,
        callbacks=[checkpoint],
        max_queue_size=40,
        workers=20,
        use_multiprocessing=True,
    )

base_model.save(base_model_path)