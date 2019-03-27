import os
import itertools
import numpy as np
import pandas as pd
import operator
import math
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.callbacks import *
from keras.layers import *
from keras.models import *
from keras.utils import *
from keras.optimizers import Adadelta, RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

from config import *
from model import create_model
from data_gen import CapchaDataGenerator

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

 
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
                if 5 <= len(label) and len(label) <= max_label_len:
                        final_result[label] = final_result.get(label, 0) + score

        return max(final_result.items(), key=operator.itemgetter(1))[0]


list_files = [] 
list_labels = []
f = open(description_path, "r")
for i, line in enumerate(f):
    if i > 0:
        fname, label = line[:-1].split(",")
        list_files.append(fname)
        list_labels.append(label)

X_train, X_val, y_train, y_val = train_test_split(list_files, list_labels, test_size=0.1, random_state=9)

testing_generator = CapchaDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, is_testing=True)

base_model = create_model(is_training=False)
base_model.load_weights(base_model_path)
print("Done loaded model!")

y_pred = base_model.predict_generator(testing_generator, max_queue_size=20, verbose=1)
print(y_pred.shape)
cnt = 0
f = open("wrong_prediction.csv", "w")
f.write("ImageId,Label,Predict\n")
beam = 0
greedy = 0
# print(y_pred[0])
for i in range(len(y_pred)):
#     if i % 1000 == 0:
#         print(int(100.0*i/len(y_pred)))
#     pred_label_greedy = decode_label_greedy(y_pred[i, 2:])
#     pred_label = decode_label_beam(y_pred[i, 2:])
#     if pred_label_greedy != pred_label:
#         if pred_label == list_labels[i]:
#             beam += 1
#         if pred_label_greedy == list_labels[i]:
#             greedy += 1
    pred_label = decode_label_greedy(y_pred[i, 2:])
    if pred_label == list_labels[i]:
        cnt += 1
    else:
        f.write("{},{},{}\n".format(list_files[i], list_labels[i], pred_label))
print("Greed {}, beam {}".format(greedy, beam))
print("ACC: ", 1.0*cnt/len(y_pred))
        
        