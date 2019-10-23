# !pip install -q tflearn
# !pip install -q urllib3
# !pip install autograd
get_ipython().magic(u'matplotlib inline')
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.misc import logsumexp
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt
from google.colab import drive
import os
import pandas as pd
import csv

#names:
d = 'hiv1_protease'
e = '652067'
f = '1053197'
#files
a = 'hiv1_protease.tsv'
b = '652067.tsv'
c = '1053197.tsv'

NUM_SAMPLES = 10000
MAX_LEN = 256
TEST = int((.2*NUM_SAMPLES))
data_path = 'drive/My Drive/DeepChem/binding_datasets'
dataset = c




if dataset == a:
  folder = d
elif dataset == b:
  folder = e
else:
  folder = f

  def get_labels(dataset):
    if dataset == a:
      labels = np.asarray(pd.read_csv(dataset, delimiter='\t', header=(0), usecols=[3,4], encoding='utf-8'))
    else:
      labels = np.asarray(pd.read_csv(dataset, delimiter='\t', header=(0), usecols=[3,4], encoding='utf-8'))
    return labels

def get_smiles():
  if dataset == a:
    smiles = np.genfromtxt(dataset, delimiter = '', skip_header=1, usecols=[1], dtype=str)
  else:
    smiles = np.asarray(pd.read_csv(dataset, delimiter='\t', header=(0), usecols=[1], encoding='utf-8'))
  return smiles


smiles = get_smiles()
if dataset != a:
  smiles.astype(str)
labels = get_labels(dataset)

y0p = []
y1 = []
for i,n in enumerate(labels[:, 1]):
  if n == 1:
    y1.append(labels[i, 1])
  else:
    y0p.append(labels[i, 1])
y0p = np.asarray(y0p)
y1 = np.asarray(y1)


non_num = len(y0p)
bind_num = len(y1)
if bind_num < NUM_SAMPLES:
    seq_num = bind_num
else:
    seq_num = NUM_SAMPLES
snum = np.random.choice(non_num, bind_num, replace=False)
print(seq_num, non_num, bind_num)

if dataset == a:
  ident = np.eye(2)
  y1 = ident[np.asarray(y1)]
  y0ip = ident[np.asarray(y0p)]
  y0 = y0ip[snum]
else:
  y0p = []
  y1 = []
  for i,z in enumerate(labels[:, 1]):
    if z == 1:
      y1.append(labels[i, 0])
    else:
      y0p.append(labels[i, 0])
  y0p = np.asarray(y0p)
  y1 = np.asarray(y1)
  y0 = y0p[snum]

  smiles_bin = []
for smile in smiles:
    seq_bin = []
    for bit in smile:
        c = ord(bit)
        n = c - 32
        seq_bin.append(n)
    if len(seq_bin) < MAX_LEN:
        for i in range(MAX_LEN-len(seq_bin)):
            seq_bin.append(0)
        smiles_bin.append(seq_bin)
    elif len(seq_bin) == MAX_LEN:
        smiles_bin.append(seq_bin)

x0p = smiles_bin[:non_num]
x1 = smiles_bin[non_num:]
x0p = np.asarray(x0p)
x1 = (np.asarray(x1))
x0 = x0p[snum]


X = np.append(x0, x1, axis=0)
Y = np.append(y0, y1, axis=0)
if dataset != a:
  Y = np.expand_dims(Y, axis=0).T
X -= 95


import tflearn
import tensorflow as tf
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

tf.reset_default_graph()
tbc=TensorBoardColab()

p = 0.9


import tensorflow as tf
tf.reset_default_graph()

if dataset == a:
  network = input_data(shape=[None, MAX_LEN])
  network = fully_connected(network, 100, activation='relu')
  network = dropout(network, p)
  network = fully_connected(network, 100, activation='relu')
  network = dropout(network, p)
  network = fully_connected(network, 100, activation='relu')
  network = dropout(network, p)
  network = fully_connected(network, 2, activation='softmax')
  network = regression(network, optimizer='SGD', loss='categorical_crossentropy', learning_rate=0.001)
else:
  network = input_data(shape=[None, MAX_LEN])
  network = fully_connected(network, 100, activation='relu')
  network = dropout(network, p)
  network = fully_connected(network, 100, activation='relu')
  network = dropout(network, p)
  network = fully_connected(network, 100, activation='relu')
  network = dropout(network, p)
  network = fully_connected(network, 1, activation='linear')
  network = regression(network, optimizer='SGD', loss=tf.losses.mean_squared_error, learning_rate=0.001)


# Training
model = tflearn.DNN(network, tensorboard_verbose=2, tensorboard_dir='./Graph')
model.fit(X,Y, n_epoch=100, validation_set=0.1, snapshot_step=10, batch_size=120, show_metric=True, run_id=(folder+ '100e,.1V, 120BS'))

















#
