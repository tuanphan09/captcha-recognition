import numpy as np
import cv2
import os
import keras
from config import *

def labels_to_text(labels):    
    return ""

def text_to_labels(text):     
    # padding     
    # index of 'blank' = label_classes - 1
    return [characters.find(c) for c in text] + [label_classes - 1] * (label_len-len(text))
    # return [characters.find(c) for c in text]


class CapchaDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_files, list_labels, datagen=None, batch_size=64, is_testing=False):
        'Initialization'
        
        self.list_files = list_files
        self.list_labels = list_labels
        self.datagen = datagen
        self.batch_size = batch_size
        self.is_testing = is_testing
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.is_testing:
            return int(np.floor((len(self.list_files)-1) / self.batch_size + 1))
        return int(np.floor(len(self.list_files) / self.batch_size))
        

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_files_batch = [self.list_files[k] for k in indexes]
        list_labels_batch = [self.list_labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_files_batch, list_labels_batch)
        if self.is_testing:
            return X, y
        return [X, y, np.ones(self.batch_size) * int(width / downsample_factor - 2), np.ones(self.batch_size) * label_len], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.is_testing == False:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_files_batch, list_labels_batch):
        'Generates data containing batch_size samples' 
        
        X = []
        y = []
        for i in range(len(list_files_batch)):
            img = cv2.imread(os.path.join(img_dir, list_files_batch[i]))
            if img is None:
                print("Cannot read image:", list_files_batch[i])
                exit()
            if img.shape[0] != height or img.shape[1] != width:
                print("{} image shape is not the same as input", list_files_batch[i])
                exit()
            if self.datagen is not None and i < 0.6 * len(list_files_batch):
                img = self.datagen.random_transform(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.reshape(img, (*img.shape, 1))
            X.append(img.transpose([1, 0, 2]))

            label = list_labels_batch[i]
            y.append(text_to_labels(label))

        X = np.array(X, dtype=np.float32)
        X /= 255
        y = np.array(y, dtype=np.float32)
        return X, y