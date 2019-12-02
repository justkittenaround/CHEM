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
    X = np.zeros([NUM_SMILES, np.max(np.asarray(smiles_bin)), MAX_LEN])
    identity = np.eye(MAX_LEN)
    for i in range(NUM_SMILES):
        s = str(i)
        x_smile = smiles_bin[i]
        x_hot = identity[x_smile]
        x = x_hot[:85, :]
        X[i, ...] = x
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
smiles_bin = np.asarray(convert_smiles())
Y = hot_labels(labels)
X = np.expand_dims(hot_smiles_img(NUM_SMILES, MAX_LEN, smiles_bin), axis=3)
# smiles_bin = np.asarray(smiles_bin)
