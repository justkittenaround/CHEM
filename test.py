
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, models, transforms
import torch.nn.functional as f
import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
import csv

LOAD_PATH = 'results/CNN-torch-150e-32bs-0.9490150637311703.pt'
SAVE_PATH = 'results/tests/'
FILE_NAME = 'deletion_reactions.tsv'
LABEL_NAME = 'hiv1_protease.tsv'

BATCH_SIZE = 8
INPUT_SIZE_R = 541
INPUT_SIZE_C = 86
ft_size = ((round(INPUT_SIZE_R/4)) +1) * ((round(INPUT_SIZE_C/4)) +1) * 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvNet(nn.Module):
    def __init__(self, NUM_CLASSES=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(ft_size, 2)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = torch.load(LOAD_PATH)
model.eval()
model = model.to(device)

#DATA###########################################################################

gps = np.genfromtxt(FILE_NAME, delimiter = '', skip_header=1, usecols=[0], dtype=str).astype(int)
og_smiles = np.genfromtxt(FILE_NAME, delimiter = '', skip_header=1, usecols=[1], dtype=str).astype(str)
neut_smiles = np.asarray(pd.read_csv(FILE_NAME, delimiter='\t', header=(0), usecols=[2], encoding='utf-8')).astype('str')

def convert_smiles(smiles, max_len):
    smiles_bin = []
    for smile in smiles:
        seq_bin = []
        if len(smile) < max_len:
            for bit in str(smile):
                key = (ord(bit)-32)
                seq_bin.append(key)
            for i in range(max_len-len(seq_bin)):
                seq_bin.append(0)
            smiles_bin.append(seq_bin)
        elif len(smile) == max_len:
            for bit in str(smile):
                key = (ord(bit)-32)
                seq_bin.append(key)
            smiles_bin.append(seq_bin)
    return smiles_bin

def hot_smiles_img(max_len, smiles_bin):
    print('Loading Data...')
    X = np.zeros((gps.shape[0], max_len, 86))
    identity = np.eye(max_len, 86)
    for i in range(gps.shape[0]):
        s = str(i)
        x_smile = smiles_bin[i]
        x = identity[x_smile]
        X[i, ...] = x
        if i == (gps.shape[0]+1):
            break
    return X

max_len = 541

labels = np.asarray(pd.read_csv(LABEL_NAME, delimiter='\t', header=(0), usecols=[4], encoding='utf-8')).astype('float64')[:,0]
labels = labels[gps, ...]

og_smiles_bin = np.asarray(convert_smiles(og_smiles, max_len))
n_smiles_bin = np.asarray(convert_smiles(neut_smiles, max_len))

og_X = hot_smiles_img(max_len, og_smiles_bin)
n_X = hot_smiles_img(max_len, n_smiles_bin)
og_X = torch.from_numpy(og_X).unsqueeze(1).type(torch.FloatTensor)
n_X = torch.from_numpy(n_X).unsqueeze(1).type(torch.FloatTensor)

og_dl = DataLoader(og_X, batch_size=BATCH_SIZE, num_workers=4)
n_dl = DataLoader(n_X, batch_size=BATCH_SIZE, num_workers=4)

data = {'og':og_dl, 'neut':n_dl}

#RUN AND SAVE###################################################################
ins_OG = np.zeros((og_X.shape[0], og_X.shape[2], og_X.shape[3]))
preds_OG = np.zeros((og_X.shape[0], 2))
ins_N = np.zeros((og_X.shape[0], og_X.shape[2], og_X.shape[3]))
preds_N = np.zeros((og_X.shape[0], 2))
for phase in ['og', 'neut']:
    n_p = 0
    n_i = 0
    for inputs in data[phase]:
        inputs = inputs.to(device)
        predictions = f.softmax(model(inputs)).detach().cpu().numpy()
        if phase == 'og':
            for ins in inputs:
                ins_OG[n_i, ...] = ins.detach().cpu().numpy()
                n_i += 1
            for preds in predictions:
                preds_OG[n_p, ...] = preds
                n_p += 1
        else:
            for ins in inputs:
                ins_N[n_i, ...] = ins.detach().cpu().numpy()
                n_i += 1
            for preds in predictions:
                preds_N[n_p, ...] = preds
                n_p += 1


og_save = []
for og in ins_OG:
    seq_bin = []
    m = np.argmax(og, axis=1)
    for bit in m:
        key = (chr(bit+32))
        if key != ' ':
            seq_bin.append(key)
    seq_bin = "".join(seq_bin)
    og_save.append([seq_bin])

n_save = []
for n in ins_N:
    seq_bin = []
    m = np.argmax(n, axis=1)
    for bit in m:
        key = (chr(bit+32))
        if key != ' ':
            seq_bin.append(key)
    seq_bin = "".join(seq_bin)
    n_save.append([seq_bin])

magna = np.subtract(preds_OG,preds_N)

a = FILE_NAME.split('.')
b = a[0]
SAVE_NAME = SAVE_PATH + b + '-tests'

with open((SAVE_NAME + '.csv'), 'w') as csvfile:
        s = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
        s.writerow(og_save)
        s.writerow(preds_OG)
        s.writerow(n_save)
        s.writerow(preds_N)
        s.writerow(magna)


###
#first node is binding









#
