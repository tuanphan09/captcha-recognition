from keras import backend as K
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.optimizers import Adadelta, RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from config import *


def ctc_lambda_func(args):
    iy_pred, ilabels, iinput_length, ilabel_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    iy_pred = iy_pred[:, 2:, :]  # no such influence
    return K.ctc_batch_cost(ilabels, iy_pred, iinput_length, ilabel_length)


def CRNN_model(is_training=True):
    inputShape = Input((width, height, 1))  # base on Tensorflow backend
    conv_1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputShape)
    conv_2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv_1)
    batchnorm_2 = BatchNormalization()(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(batchnorm_2)

    conv_3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool_2)
    conv_4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_3)
    batchnorm_4 = BatchNormalization()(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(batchnorm_4)

    conv_5 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool_4)
    conv_6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv_5)
    batchnorm_6 = BatchNormalization()(conv_6)

    bn_shape = batchnorm_6.get_shape()  


    print(bn_shape) 

    x_reshape = Reshape(target_shape=(int(bn_shape[1]), int(bn_shape[2] * bn_shape[3])))(batchnorm_6)
    drop_reshape = Dropout(0.25)(x_reshape)
    fc_1 = Dense(256, activation='relu')(drop_reshape)  

    print(x_reshape.get_shape())  
    print(fc_1.get_shape()) 

    bi_LSTM_1 = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal'), merge_mode='sum')(fc_1)
    bi_LSTM_2 = Bidirectional(LSTM(128, return_sequences=True, kernel_initializer='he_normal'), merge_mode='concat')(bi_LSTM_1)

    drop_rnn = Dropout(0.3)(bi_LSTM_2)

    fc_2 = Dense(label_classes, kernel_initializer='he_normal', activation='softmax')(drop_rnn)

    base_model = Model(inputs=[inputShape], outputs=fc_2) 

    labels = Input(name='the_labels', shape=[label_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([fc_2, labels, input_length, label_length])

    if is_training:
        return Model(inputs=[inputShape, labels, input_length, label_length], outputs=[loss_out]), base_model
    else:
        return base_model