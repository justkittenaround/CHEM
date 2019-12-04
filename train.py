import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import visdom
import pandas as pd
from PIL import Image
import progressbar

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# class DataParallelModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.block1 = nn.Linear(10, 20)
#         # wrap block2 in DataParallel
#         self.block2 = nn.Linear(20, 20)
#         self.block2 = nn.DataParallel(self.block2)
#         self.block3 = nn.Linear(20, 20)
#     def forward(self, x):
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         return x

vis = visdom.Visdom()

PATH = 'results'

RESULTS = 'results/graphs'

SAVE_PATH = 'data/train'

FILE_NAME = 'hiv1_protease.tsv'

MODEL_NAME = 'CNN'

NUM_CLASSES = 2

BATCH_SIZE = 32

NUM_EPOCHS = 150

INPUT_SIZE_R = 541

INPUT_SIZE_C = 86

LR = .0005

EXTRA_NAME = '-torch-'+str(NUM_EPOCHS)+'e-'+str(BATCH_SIZE)+'bs-'

ft_size = ((round(INPUT_SIZE_R/4)) +1) * ((round(INPUT_SIZE_C/4)) +1) * 32

#DEFINE TRAINING PROCEDURE ################################
def train_model(model, DATA, criterion, optimizer, NUM_EPOCHS):
    print('Beginning training!')

    since = time.time()
    val_acc_history = []
    train_acc = []
    train_loss = []
    val_loss = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in progressbar.progressbar(range(NUM_EPOCHS)):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if (epoch % 2) != 0:
                vis.image(inputs[1,...], win='inputs', opts=dict(title=MODEL_NAME + '-inputs'))
                vis.text(str(labels[1,...]), win='labels', opts=dict(title=MODEL_NAME + '-labels'))

            if phase == 'train':
                train_acc.append(epoch_acc.cpu().numpy())
                vis.line(train_acc, win='train_acc', opts=dict(title= MODEL_NAME + '-train_acc'))
                train_loss.append(epoch_loss)
                vis.line(train_loss, win='train_loss', opts=dict(title= MODEL_NAME + '-train_loss'))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc.cpu().numpy())
                vis.line(val_acc_history, win='val_acc', opts=dict(title= MODEL_NAME + '-val_acc'))
                val_loss.append(epoch_loss)
                vis.line(val_loss, win='val_loss', opts=dict(title= MODEL_NAME + '-val_loss'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)

    return model, val_acc_history, best_acc, val_loss, time_elapsed


#DEFINE MODEL #############################################################
if MODEL_NAME == 'resnet':
    model = models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

else:
    class ConvNet(nn.Module):
        def __init__(self, NUM_CLASSES, BATCH_SIZE):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Linear(ft_size, NUM_CLASSES)
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
            return out

    model = ConvNet(NUM_CLASSES, BATCH_SIZE)


# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     model = nn.DataParallel(model)

model = model.to(device)

params_to_update = model.parameters()
optimizer = optim.Adam(params_to_update, lr=LR)
# optimizer = optim.SGD(params_to_update, lr=LR)
criterion = nn.CrossEntropyLoss()

#DATA #####################
def get_labels(FILE_NAME):
    labels = np.asarray(pd.read_csv(FILE_NAME, delimiter='\t', header=(0), usecols=[4], encoding='utf-8')).astype('float64')
    return labels

def get_smiles(FILE_NAME):
    smiles = np.genfromtxt(FILE_NAME, delimiter = '', skip_header=1, usecols=[1], dtype=str).astype(str)
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

def hot_smiles_img(MAX_LEN, smiles_bin):
    print('Loading Data...')
    X = np.zeros((smiles.shape[0], MAX_LEN, 86))
    identity = np.eye(MAX_LEN, 86)
    for i in range(smiles.shape[0]):
        s = str(i)
        x_smile = smiles_bin[i]
        x = identity[x_smile]
        X[i, ...] = x
        if i == (smiles.shape[0]+1):
            break
    return X

labels = get_labels(FILE_NAME)[:,0]
smiles = get_smiles(FILE_NAME)
MAX_LEN = max([len(x) for x in smiles])
NUM_SMILES = len(smiles)
num_0 = np.argmax(labels)
num_1 = NUM_SMILES - num_0
select = np.random.choice(num_0, int(num_1), replace=False)
Y0 = labels[select, ...]
Y1 = labels[num_0:, ...]
Y = torch.from_numpy(np.append(Y0, Y1)).type(torch.FloatTensor)
smiles0 = smiles[select, ...]
smiles1 = smiles[num_0:, ...]
smiles = np.append(smiles0, smiles1)
smiles_bin = np.asarray(convert_smiles())
X = hot_smiles_img(MAX_LEN, smiles_bin)
X = torch.from_numpy(X).unsqueeze(1).type(torch.FloatTensor)
val_num = int(X.shape[0]*.20)
val_idx = np.random.choice(smiles.shape[0], val_num, replace=False)
X_val = X[val_idx, ...]
X_train = X[[i for i in range(smiles.shape[0]) if i not in val_idx], ...]
Y_val = Y[val_idx, ...]
Y_train = Y[[i for i in range(smiles.shape[0]) if i not in val_idx], ...]

train_ds = TensorDataset(X_train, Y_train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=4)
val_ds = TensorDataset(X_val, Y_val)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2, num_workers=4)
dataloaders = {'train': train_dl, 'val': val_dl}


#RUN AND SAVE ###########################################
model, val_acc, best_acc, val_loss, time_elapsed = train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS)

val_loss_plt = plt.figure()
plt.plot(val_loss)
val_loss_plt.savefig(RESULTS + '/' + MODEL_NAME + EXTRA_NAME + '_val-loss.jpg')
val_acc_plt = plt.figure()
plt.plot(val_acc)
val_acc_plt.savefig(RESULTS + '/' + MODEL_NAME + EXTRA_NAME + '_val-acc.jpg')

save_name = PATH + '/' + MODEL_NAME + EXTRA_NAME + str(best_acc.detach().cpu().numpy()) + '.pt'
torch.save(model, save_name)

print(model.weights().type())
