import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv


FILE_NAME = 'hiv1_protease.tsv'

fn = FILE_NAME.split('.')

NUM_EPOCHS = 100
NUM_VAL = 0.25
BATCH_SIZE = 8
DROPOUT_RATE = 0.9
LR = 0.002

SAVE_NAME = fn[0] + '-e:'+ str(NUM_EPOCHS) + '-bs:' + str(BATCH_SIZE) + '-dropout'


def get_labels(FILE_NAME):
    labels = np.asarray(pd.read_csv(FILE_NAME, delimiter='\t', header=(0), usecols=[3,4], encoding='utf-8'))
    return labels

def get_smiles(FILE_NAME):
    smiles = np.genfromtxt(FILE_NAME, delimiter = '', skip_header=1, usecols=[1], dtype=str)
    smiles.astype(str)
    return smiles

def convert_smiles():
  smiles_bin = []
  for smile in smiles:
    seq_bin = []
    if len(smile) < MAX_LEN:
      for bit in str(smile):
        key = (ord(bit)-32)
        seq_bin.append(key)
      for i in range(MAX_LEN-len(seq_bin)):
        seq_bin.append(0)
      smiles_bin.append(seq_bin)
    elif len(smile) == MAX_LEN:
      for bit in str(smile):
        key = (ord(bit)-32)
        seq_bin.append(key)
      smiles_bin.append(seq_bin)
  return smiles_bin

def hot_smiles_img(NUM_SMILES, MAX_LEN, smiles_bin):
    X = np.zeros([NUM_SMILES, MAX_LEN, MAX_LEN])
    identity = np.eye(MAX_LEN)
    for i in range(NUM_SMILES):
        s = str(i)
        x_smile = smiles_bin[i]
        x_hot = identity[x_smile]
        X[i, ...] = x_hot
        if i == (NUM_SMILES+1):
            break
    return X

def hot_labels(labels):
    labels = labels.astype(np.int32)
    ident = np.eye(2)
    Y = ident[labels[:,1]]
    return Y

labels = get_labels(FILE_NAME)
smiles = get_smiles(FILE_NAME)
MAX_LEN = max([len(x) for x in smiles])
NUM_SMILES = len(smiles)
smiles_bin = convert_smiles()
Y = hot_labels(labels)
# X = hot_smiles_img(NUM_SMILES, MAX_LEN, smiles_bin)
X = smiles_bin



import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences


class DataAugmentation(object):
    def __init__(self):
        self.methods = []
        self.args = []
    def apply(self, batch):
        for i, m in enumerate(self.methods):
            if self.args[i]:
                batch = m(batch, *self.args[i])
            else:
                batch = m(batch)
        return batch

class ImageAugmentation(DataAugmentation):
    def __init__(self):
        super(ImageAugmentation, self).__init__()
    def add_random_flip_leftright(self):
        self.methods.append(self._random_flip_leftright)
        self.args.append(None)
    def _random_flip_leftright(self, batch):
        for i in range(len(batch)):
                batch[i] = np.roll(batch[i],np.random.randint(max_length),axis=2)
        return batch

img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()

tf.reset_default_graph()

network = input_data(shape=[None, MAX_LEN], data_augmentation=img_aug)
network = fully_connected(network, 200 , activation='tanh')
network = dropout(network, DROPOUT_RATE)
network = fully_connected(network, 200, activation='tanh')
network = dropout(network, DROPOUT_RATE)
network = fully_connected(network, 200, activation='tanh')
network = dropout(network, DROPOUT_RATE)
network = fully_connected(network, 200, activation='tanh')
network = dropout(network, DROPOUT_RATE)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=LR)

# Training
model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir='./Graph')
model.fit(X,Y, n_epoch=NUM_EPOCHS, validation_set=NUM_VAL, snapshot_step=10, batch_size=BATCH_SIZE, show_metric=True, run_id=SAVE_NAME)
model.save(run_id + '.tflearn')









#
