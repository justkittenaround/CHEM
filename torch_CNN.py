#utils
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms, utils
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import visdom
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
from PIL import Image

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
vis = visdom.Visdom()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DataParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Linear(10, 20)
        # wrap block2 in DataParallel
        self.block2 = nn.Linear(20, 20)
        self.block2 = nn.DataParallel(self.block2)
        self.block3 = nn.Linear(20, 20)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

data_dir = 'data'

PATH = 'results'

RESULTS = 'results/graphs'

NUM_CLASSES = 2

BATCH_SIZE = 16

NUM_EPOCHS = 100

INPUT_SIZE = 541

MODEL_NAME = 'CNN'

EXTRA_NAME = 'torch-100e-16bs-cnn'

PRETRAIN = True

ft_size = ((round(INPUT_SIZE/4)) +1) * ((round(INPUT_SIZE/4)) +1) * 32
##############################################################################
model_best = []
def train_model(model, dataloaders, criterion, optimizer, NUM_EPOCHS=NUM_EPOCHS, is_inception=False):
    since = time.time()
    val_acc_history = []
    train_acc = []
    train_loss = []
    val_loss = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(epoch, NUM_EPOCHS - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if (epoch % 2) != 0:
                continue
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
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
    return model, val_acc_history, best_acc, val_loss

##SET MODEL PARAMETERS WITH GRAD################################################

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}

print("Initializing Datasets and Dataloaders...")
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'val']}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load_model#############################################################
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
        self.fc = nn.Linear(ft_size, NUM_CLASSES)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def initialize_model(MODEL_NAME, NUM_CLASSES, use_pretrained=PRETRAIN):
    model_ft = None
    INPUT_SIZE = 0
    if MODEL_NAME == "resnet":
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        INPUT_SIZE = 541
    elif MODEL_NAME == "CNN":
        model_ft = ConvNet(NUM_CLASSES)
    return model_ft, INPUT_SIZE
ft_model, INPUT_SIZE = initialize_model(MODEL_NAME, NUM_CLASSES, use_pretrained=PRETRAIN)

if torch.cuda.device_count() > 1:
              print("Let's use", torch.cuda.device_count(), "GPUs!")
              # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
              ft_model = nn.DataParallel(ft_model)

ft_model= ft_model.to(device)
params_to_update = ft_model.parameters()
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

ft_model, hist, best_acc, val_loss = train_model(ft_model, dataloaders_dict, criterion, optimizer_ft, NUM_EPOCHS=NUM_EPOCHS)

model_best.append(MODEL_NAME)
model_best.append(best_acc.cpu().item())
print('model bests:', model_best)
save_name = PATH + '/' + MODEL_NAME + EXTRA_NAME + '.pt'
val_loss_plt = plt.figure()
plt.plot(val_loss)
val_loss_plt.savefig(RESULTS + '/' + MODEL_NAME + EXTRA_NAME + '_val-loss.jpg')
val_acc_plt = plt.figure()
plt.plot(hist)
val_acc_plt.savefig(RESULTS + '/' + MODEL_NAME + EXTRA_NAME + '_val-acc.jpg')

##SAVE MODEL
torch.save(ft_model, save_name)
    # torch.save(ft_model.state_dict(), save_name + '.ckpt')

#Beep when done
import os
duration = 3
freq = 333
os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
