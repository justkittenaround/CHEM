import numpy as np
from PIL import Image
import pandas as pd

SAVE_PATH = 'data/test/neut'

FILE_NAME = 'neutralized_reactions.tsv'

def get_labels(FILE_NAME):
    labels = np.asarray(pd.read_csv(FILE_NAME, delimiter='\t', header=(0), encoding='utf-8'))
    return labels

def convert_smiles(smiles):
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

def save_img(imgs, SAVE_PATH, fol):
    for idx, x in enumerate(imgs):
        img = Image.fromarray(x)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        save_name = SAVE_PATH + '/' + fol + '/' + str(idx) + '.png'
        img.save(save_name, 'png')

smiles = get_labels(FILE_NAME)
og = smiles[:,0]
neut = smiles[:,0]
MAX_LENog = max([len(x) for x in smiles[:,0]])
MAX_LENn = max([len(x) for x in smiles[:,1]])
MAX_LEN =  max(MAX_LENog, MAX_LENn)
NUM_SMILES = smiles.shape[0]
smiles_bin_og = np.asarray(convert_smiles(og))
smiles_bin_n = np.asarray(convert_smiles(neut))
X_og = hot_smiles_img(NUM_SMILES, MAX_LEN, smiles_bin_og)
X_n = hot_smiles_img(NUM_SMILES, MAX_LEN, smiles_bin_n)


save_img(X_og, SAVE_PATH, 'og')
save_img(X_n, SAVE_PATH, 'n')



#
