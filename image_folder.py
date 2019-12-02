import numpy as np
from PIL import Image
import pandas as pd

SAVE_PATH = 'data/train'

FILE_NAME = 'hiv1_protease.tsv'

def get_labels(FILE_NAME):
    labels = np.asarray(pd.read_csv(FILE_NAME, delimiter='\t', header=(0), usecols=[3,4], encoding='utf-8'))
    # labels = np.asarray(pd.read_csv(FILE_NAME, delimiter='\t', header=(0), encoding='utf-8'))
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
    X = np.zeros([NUM_SMILES,MAX_LEN, MAX_LEN])
    identity = np.eye(MAX_LEN)
    for i in range(NUM_SMILES):
        s = str(i)
        x_smile = smiles_bin[i]
        x = identity[x_smile]
        X[i, ...] = x
        if i == (NUM_SMILES+1):
            break
    return X

def hot_labels(labels):
    labels = labels.astype(np.int32)
    ident = np.eye(2)
    Y = ident[labels[:,1]]
    return Y

def save_img(imgs, lab, SAVE_PATH, split):
    for idx, x in enumerate(imgs):
        img = Image.fromarray(x)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if lab[idx] == 1:
            fol = 'bind'
        else:
            fol = 'not_bind'
        save_name = SAVE_PATH + '/' + split + '/' + fol + '/' + str(idx) + '.png'
        img.save(save_name, 'png')

labels = get_labels(FILE_NAME)
smiles = get_smiles(FILE_NAME)
MAX_LEN = max([len(x) for x in smiles])
NUM_SMILES = len(smiles)
smiles_bin = np.asarray(convert_smiles())
Y = np.asarray(labels[:, 1])

val_num = int(X.shape[0]*.20)
val_idx = np.random.choice(X.shape[0], val_num, replace=False)
X_val = X[val_idx, ...]
X_train = X[[i for i in range(X.shape[0]) if i not in val_idx], ...]
Y_val = Y[val_idx, ...]
Y_train = Y[[i for i in range(Y.shape[0]) if i not in val_idx], ...]



save_img(X_val, Y_val, SAVE_PATH, 'val')
save_img(X_train, Y_train, SAVE_PATH, 'train')




















#
