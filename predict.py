import os
import itertools
import numpy as np
import pandas as pd
import operator
import math
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
from keras import backend as K
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.optimizers import Adadelta, RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from config import *
from model import *
from data_gen import CapchaDataGenerator
import cv2 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

 
# beam search
def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			for j in range(len(row)):
				candidate = [seq + [j], score * row[j]]
				all_candidates.append(candidate)
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse=True)
		# select k best
		sequences = ordered[:k]
	return sequences

def greedy_search_decoder(data):
        return np.argmax(data, axis=1)

def get_label(idxes):
        result = ''
        for i in range(len(idxes)):
                if (i == 0 or idxes[i] != idxes[i-1]) and idxes[i] < len(characters):
                        result += characters[idxes[i]]
        return result

def decode_label_greedy(data):
        idxes = greedy_search_decoder(data)
        return get_label(idxes)

def decode_label_beam(data, top=10):
        top_result = beam_search_decoder(data, top)
        final_result = {get_label(top_result[0][0]) : 0}
        for res in top_result: 
                idxes, score = res
                label = get_label(idxes)
                if len(label) == label_len:
                        final_result[label] = final_result.get(label, 0) + score

        return max(final_result.items(), key=operator.itemgetter(1))[0]



# load data
list_files = [] 
list_labels = []
f = open(description_path, "r")
for i, line in enumerate(f):
    if i > 0:
        fname, label = line[:-1].split(",")
        list_files.append(fname)
        list_labels.append(label)

testing_generator = CapchaDataGenerator(list_files, list_labels, batch_size=BATCH_SIZE, is_testing=True)

# load model
base_model = CRNN_model(is_training=False)
base_model.load_weights(base_model_path)
print("Done loaded model!")


# image example
img_path = 'data/train/1850b7bd60887ad59144f384ecdcce80.png'

img = cv2.imread(img_path)  # (height, width)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert rgb to gray
img = np.reshape(img, (*img.shape, 1))  # (height, width, 1)
img = img.transpose([1, 0, 2])    # (width, height, 1)

X = np.array([img], dtype=np.float32)  # X.shape = (number of images, width, height, 1)
X /= 255  # normalize

y_pred = base_model.predict(X)  

text = decode_label_greedy(y_pred[0, 2:])
print("Text:", text)