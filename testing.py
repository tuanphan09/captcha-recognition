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
        print(idxes)
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
print("Testing data in {}".format(description_path))
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

# predict
y_pred = base_model.predict_generator(testing_generator, max_queue_size=20, verbose=1)
print(y_pred.shape)

# decode multi-process
start = time.time()
p = Pool(8)
label_pred_greedy = p.map(decode_label_greedy, y_pred[:, 2:])
label_pred_beam = p.map(decode_label_beam, y_pred[:, 2:])
print("Decoding time:", time.time()-start)


beam = 0
greedy = 0
for i in range(len(y_pred)):
        greedy_label = label_pred_greedy[i]
        beam_label = label_pred_beam[i]
        true_label = list_labels[i]

        if greedy_label == true_label:
                greedy += 1
        if beam_label == true_label:
                beam += 1

print("Greed {}, beam {}".format(greedy, beam))
print("ACC greedy: ", 1.0*greedy/len(y_pred))
print("ACC beam: ", 1.0*beam/len(y_pred))